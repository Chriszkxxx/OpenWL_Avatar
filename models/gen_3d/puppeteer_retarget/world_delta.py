#!/usr/bin/env python3
"""World-delta Mixamo -> Puppeteer skeleton retargeting (Blender / bpy).

Recommended retarget method. Applies each source bone's world-space rotation
delta to the destination rest pose:

    delta_ws   = src_pose_ws @ src_rest_ws^-1
    target_ws  = delta_ws @ dst_rest_ws

This is axis-frame independent: it does not depend on the destination bone's
local roll, which avoids the arm-drift / backward-knee artifacts produced by
local-frame delta methods. Because only world-space rotations are transferred,
the *source* can be a Mixamo FBX or a BVH (e.g. MoMask) interchangeably — the
importer is chosen by file extension and the root bones come from the mapping
JSON's `root_bones`.

Run as a module so relative imports resolve::

    # Mixamo FBX
    python -m models.gen_3d.puppeteer_retarget.world_delta \\
        --glb char.glb --rig char.txt --source-anim run.fbx \\
        --mapping mappings/luffi_puppeteer_ue_mixamo_mapping.json \\
        --output out.fbx

    # BVH (direct, no Mixamo intermediate)
    python -m models.gen_3d.puppeteer_retarget.world_delta \\
        --glb char.glb --rig char.txt --source-anim motion.bvh \\
        --mapping mappings/momask_bvh_to_puppeteer_mapping.json \\
        --output out.fbx --fps 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import bpy
from mathutils import Quaternion, Vector

from .rig_io import (
    assign_vertex_weights,
    build_rigged_mesh_objects,
    clear_bpy_data,
    import_glb,
    load_rigged_mesh_data,
    parent_to_armature,
)

# Default root bones (overridable via the mapping JSON or CLI).
ROOT_MIX = "mixamorig:Hips"
ROOT_PUP = "joint23"


def load_bone_map(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["bone_map"]


def load_root_bones(
    path: str,
    default_src: str = "mixamorig:Hips",
    default_dst: str = "joint23",
) -> Tuple[str, str]:
    """Read source/target root bone names from a mapping JSON's `root_bones`.

    Accepts either {"source": ..., "puppeteer": ...} (BVH map) or
    {"mixamo": ..., "puppeteer": ...} (Mixamo map).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    roots = data.get("root_bones", {})
    src = roots.get("source") or roots.get("mixamo") or default_src
    dst = roots.get("puppeteer") or roots.get("target") or default_dst
    return src, dst


def stabilize(q: Quaternion, prev: Quaternion | None) -> Quaternion:
    """Flip quaternion sign to stay on the same hemisphere as the previous frame."""
    q = q.normalized()
    if prev is not None and q.dot(prev) < 0.0:
        q = Quaternion((-q.w, -q.x, -q.y, -q.z))
    return q


def clamp_delta_deg(q: Quaternion, max_deg: float) -> Quaternion:
    """Clamp a rotation delta magnitude to avoid ~180-degree flips."""
    if max_deg <= 0:
        return q
    q = q.normalized()
    w = max(-1.0, min(1.0, q.w))
    theta = 2.0 * math.acos(abs(w))
    max_rad = math.radians(max_deg)
    if theta <= max_rad:
        return q
    s = max_rad / theta
    return Quaternion((1, 0, 0, 0)).slerp(q, s).normalized()


def import_source_animation(
    path: str,
    global_scale: float = 1.0,
) -> Tuple[bpy.types.Object, bpy.types.Action]:
    """Import a source animation armature from FBX or BVH (dispatch by extension).

    BVH (e.g. MoMask) can be retargeted directly: the world-delta math only
    reads per-bone world rotations, so it is independent of the source bone
    naming / roll. `global_scale` lets you reconcile BVH units with the rig.
    """
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext == ".bvh":
        bpy.ops.import_anim.bvh(
            filepath=path,
            axis_forward="-Z",
            axis_up="Y",
            global_scale=global_scale,
        )
    else:
        bpy.ops.import_scene.fbx(filepath=path)
    new_objs = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new_objs if o.type == "ARMATURE")
    for o in new_objs:
        if o.type == "MESH":
            o.hide_viewport = True
            o.hide_render = True
    arm.name = "AnimSource"
    if not arm.animation_data or not arm.animation_data.action:
        raise RuntimeError(f"No animation action found in {path}")
    return arm, arm.animation_data.action


def import_mixamo_animation(path: str) -> Tuple[bpy.types.Object, bpy.types.Action]:
    """Backwards-compatible alias for FBX/BVH source import."""
    return import_source_animation(path)


def build_puppeteer_rig(glb_path: str, rig_path: str) -> Tuple[bpy.types.Object, bpy.types.Object]:
    """Import the textured GLB and bind it to the Puppeteer armature + weights."""
    textured = import_glb(glb_path)
    data = load_rigged_mesh_data(glb_path, rig_path, apply_coord_rot=True)
    proxy_mesh, arm_obj = build_rigged_mesh_objects(
        data, mesh_name="RigProxy", collection_name="RigProxy_collection"
    )
    for name in data["names"]:
        if name not in textured.vertex_groups:
            textured.vertex_groups.new(name=name)
    assign_vertex_weights(textured, data["names"], data["vertex_group"])
    bpy.data.objects.remove(proxy_mesh, do_unlink=True)
    parent_to_armature(textured, arm_obj)

    arm_obj.name = "PuppeteerArmature"
    textured.name = "PuppeteerMesh"
    for mod in textured.modifiers:
        if mod.type == "ARMATURE":
            mod.object = arm_obj
    textured.parent = arm_obj
    return textured, arm_obj


def cache_rest_world_quats(arm: bpy.types.Object) -> Dict[str, Quaternion]:
    arm_q = arm.matrix_world.to_quaternion()
    out: Dict[str, Quaternion] = {}
    for b in arm.data.bones:
        local_q = b.matrix_local.to_quaternion()
        out[b.name] = (arm_q @ local_q).normalized()
    return out


def cache_local_rest_quats(arm: bpy.types.Object) -> Dict[str, Quaternion]:
    out: Dict[str, Quaternion] = {}
    for b in arm.data.bones:
        if b.parent is not None:
            m = b.parent.matrix_local.inverted() @ b.matrix_local
        else:
            m = b.matrix_local
        out[b.name] = m.to_quaternion().normalized()
    return out


def order_mapping_parent_first(
    mapping: Dict[str, str], dst_arm: bpy.types.Object
) -> List[Tuple[str, str]]:
    depth: Dict[str, int] = {}

    def d(name: str) -> int:
        if name in depth:
            return depth[name]
        b = dst_arm.data.bones[name]
        depth[name] = 0 if b.parent is None else d(b.parent.name) + 1
        return depth[name]

    items = list(mapping.items())
    items.sort(key=lambda kv: d(kv[1]))
    return items


def reset_dest_pose(arm: bpy.types.Object) -> None:
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.location = Vector((0.0, 0.0, 0.0))
        pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
        pb.scale = Vector((1.0, 1.0, 1.0))


def retarget_frame(
    src_arm: bpy.types.Object,
    dst_arm: bpy.types.Object,
    mapping_ordered: List[Tuple[str, str]],
    src_rest_ws: Dict[str, Quaternion],
    dst_rest_ws: Dict[str, Quaternion],
    rest_local_q: Dict[str, Quaternion],
    max_delta_deg: float,
    correction: Dict[str, Quaternion],
) -> None:
    src_arm_q = src_arm.matrix_world.to_quaternion()
    dst_arm_q = dst_arm.matrix_world.to_quaternion()

    for src_name, dst_name in mapping_ordered:
        src_pb = src_arm.pose.bones[src_name]
        dst_pb = dst_arm.pose.bones[dst_name]
        dst_bone = dst_arm.data.bones[dst_name]

        src_pose_ws = (src_arm_q @ src_pb.matrix.to_quaternion()).normalized()
        delta_ws = (src_pose_ws @ src_rest_ws[src_name].inverted()).normalized()
        delta_ws = clamp_delta_deg(delta_ws, max_delta_deg)

        target_ws = (delta_ws @ dst_rest_ws[dst_name]).normalized()
        corr = correction.get(dst_name)
        if corr is not None:
            target_ws = (target_ws @ corr).normalized()

        if dst_bone.parent is not None:
            parent_pb = dst_arm.pose.bones[dst_bone.parent.name]
            parent_pose_ws = (dst_arm_q @ parent_pb.matrix.to_quaternion()).normalized()
        else:
            parent_pose_ws = dst_arm_q
        no_self_ws = (parent_pose_ws @ rest_local_q[dst_name]).normalized()
        local_q = (no_self_ws.inverted() @ target_ws).normalized()

        dst_pb.rotation_quaternion = local_q
        bpy.context.view_layer.update()


def apply_root_translation(
    src_arm: bpy.types.Object,
    dst_arm: bpy.types.Object,
    bake_to_bone: bool,
    root_scale: float,
) -> Vector:
    src_root_pose = src_arm.matrix_world @ src_arm.pose.bones[ROOT_MIX].matrix
    src_root_rest = src_arm.matrix_world @ src_arm.data.bones[ROOT_MIX].matrix_local
    offset = (src_root_pose.translation - src_root_rest.translation) * root_scale

    if bake_to_bone:
        dst_arm.location = Vector((0, 0, 0))
        dst_arm.pose.bones[ROOT_PUP].location = offset
    else:
        dst_arm.location = offset
        dst_arm.pose.bones[ROOT_PUP].location = Vector((0, 0, 0))
    return offset


def bake_action(
    src_arm: bpy.types.Object,
    dst_arm: bpy.types.Object,
    mapping: Dict[str, str],
    action_name: str,
    frame_start: int,
    frame_end: int,
    bake_root_to_bone: bool,
    root_scale: float,
    max_delta_deg: float,
    correction: Dict[str, Quaternion],
) -> bpy.types.Action:
    src_rest_ws = cache_rest_world_quats(src_arm)
    dst_rest_ws = cache_rest_world_quats(dst_arm)
    rest_local_q = cache_local_rest_quats(dst_arm)
    mapping_ordered = order_mapping_parent_first(mapping, dst_arm)

    action = bpy.data.actions.new(action_name)
    if dst_arm.animation_data is None:
        dst_arm.animation_data_create()
    dst_arm.animation_data.action = action

    scene = bpy.context.scene
    bpy.context.view_layer.objects.active = dst_arm
    bpy.ops.object.mode_set(mode="POSE")

    prev_q: Dict[str, Quaternion] = {}
    all_dst_bones = [b.name for b in dst_arm.data.bones]

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        reset_dest_pose(dst_arm)
        bpy.context.view_layer.update()
        retarget_frame(
            src_arm,
            dst_arm,
            mapping_ordered,
            src_rest_ws,
            dst_rest_ws,
            rest_local_q,
            max_delta_deg,
            correction,
        )
        apply_root_translation(src_arm, dst_arm, bake_root_to_bone, root_scale)

        if not bake_root_to_bone:
            dst_arm.keyframe_insert(data_path="location", frame=frame)

        for name in all_dst_bones:
            pb = dst_arm.pose.bones[name]
            q = stabilize(pb.rotation_quaternion.copy(), prev_q.get(name))
            pb.rotation_quaternion = q
            prev_q[name] = q
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            if bake_root_to_bone and name == ROOT_PUP:
                pb.keyframe_insert(data_path="location", frame=frame)

    bpy.ops.object.mode_set(mode="OBJECT")
    return action


def export_animated_fbx(
    path: str,
    fps: int,
    frame_start: int,
    frame_end: int,
    action_name: str = "Take 001",
    anim_only: bool = False,
) -> None:
    """Export FBX with explicit frame range and a UE-friendly take name."""
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = frame_start
    scene.frame_end = frame_end

    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE" and not o.hide_viewport)
    if arm.animation_data and arm.animation_data.action:
        arm.animation_data.action.name = action_name

    bpy.ops.object.select_all(action="DESELECT")
    for o in bpy.data.objects:
        if o.type == "ARMATURE" and not o.hide_viewport:
            o.select_set(True)
        elif o.type == "MESH" and not o.hide_viewport and not anim_only:
            o.select_set(True)

    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(
        filepath=path,
        check_existing=False,
        use_selection=True,
        add_leaf_bones=False,
        path_mode="COPY",
        embed_textures=not anim_only,
        mesh_smooth_type="FACE",
        use_mesh_modifiers=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_step=1,
        bake_anim_simplify_factor=0.0,
        bake_anim_force_startend_keying=True,
        apply_scale_options="FBX_SCALE_ALL",
        axis_forward="-Z",
        axis_up="Y",
        bake_space_transform=False,
        use_armature_deform_only=True,
        object_types={"ARMATURE"} if anim_only else {"ARMATURE", "MESH"},
    )


def run(args: argparse.Namespace) -> None:
    global ROOT_MIX, ROOT_PUP

    clear_bpy_data()
    mapping = load_bone_map(args.mapping)

    # Resolve root bones: CLI override > mapping JSON > module defaults.
    map_src, map_dst = load_root_bones(args.mapping, ROOT_MIX, ROOT_PUP)
    ROOT_MIX = args.src_root or map_src
    ROOT_PUP = args.dst_root or map_dst

    print("[1/5] Building Puppeteer mesh + armature + weights...")
    mesh_obj, dst_arm = build_puppeteer_rig(args.glb, args.rig)
    weighted = sum(1 for v in mesh_obj.data.vertices if len(v.groups) > 0)
    print(
        f"  verts={len(mesh_obj.data.vertices)}, "
        f"groups={len(mesh_obj.vertex_groups)}, weighted={weighted}"
    )

    ext = os.path.splitext(args.source_anim)[1].lower()
    print(f"[2/5] Importing source animation ({ext or 'fbx'}): {args.source_anim}")
    print(f"  roots: src={ROOT_MIX} -> dst={ROOT_PUP}")
    src_arm, src_action = import_source_animation(args.source_anim, global_scale=args.global_scale)
    fs = int(args.frame_start if args.frame_start >= 0 else src_action.frame_range[0])
    fe = int(args.frame_end if args.frame_end >= 0 else src_action.frame_range[1])
    print(f"  frames {fs}-{fe} ({fe - fs + 1} frames)")

    correction: Dict[str, Quaternion] = {}
    if args.correction_json:
        with open(args.correction_json, encoding="utf-8") as f:
            for name, q in json.load(f).items():
                correction[name] = Quaternion(q).normalized()
        print(f"  loaded {len(correction)} per-bone correction quats")

    print("[3/5] Baking (world-delta retargeting, axis-frame independent)...")
    action_name = args.action_name or os.path.splitext(os.path.basename(args.source_anim))[0]
    action = bake_action(
        src_arm,
        dst_arm,
        mapping,
        action_name=action_name,
        frame_start=fs,
        frame_end=fe,
        bake_root_to_bone=args.bake_root_to_bone,
        root_scale=args.root_scale,
        max_delta_deg=args.max_delta_deg,
        correction=correction,
    )
    print(f"  action '{action.name}' fcurves={len(action.fcurves)}")

    bpy.data.objects.remove(src_arm, do_unlink=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    ue_action = args.action_name or "Take 001"
    print(f"[4/5] Exporting animated FBX: {args.output}")
    if args.anim_only:
        print("  mode: anim-only (armature, no mesh — for UE Existing Skeleton)")
    export_animated_fbx(
        args.output,
        fps=args.fps,
        frame_start=fs,
        frame_end=fe,
        action_name=ue_action,
        anim_only=args.anim_only,
    )

    meta = {
        "source_animation": args.source_anim,
        "source_type": ext.lstrip(".") or "fbx",
        "target_skeleton": args.glb,
        "rig": args.rig,
        "output": args.output,
        "action_name": action.name,
        "frame_start": fs,
        "frame_end": fe,
        "fps": args.fps,
        "retarget_method": "world_conjugation_delta",
        "src_root": ROOT_MIX,
        "dst_root": ROOT_PUP,
        "global_scale": args.global_scale,
        "bake_root_to_bone": args.bake_root_to_bone,
        "root_scale": args.root_scale,
        "max_delta_deg": args.max_delta_deg,
        "bone_map": mapping,
    }
    meta_path = os.path.splitext(args.output)[0] + "_bake_info.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[5/5] Metadata: {meta_path}")
    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="World-delta Mixamo -> Puppeteer retarget."
    )
    p.add_argument("--glb", required=True, help="Target character GLB (textures preserved).")
    p.add_argument("--rig", required=True, help="Puppeteer rig txt (skeleton + skin weights).")
    p.add_argument("--source-anim", "--mixamo-anim", dest="source_anim", required=True,
                   help="Source animation. FBX (Mixamo) or BVH (e.g. MoMask) — "
                        "dispatched by file extension.")
    p.add_argument("--mapping", required=True, help="Source->Puppeteer bone_map JSON.")
    p.add_argument("--output", required=True, help="Output animated FBX path.")
    p.add_argument("--action-name", default="")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frame-start", type=int, default=-1)
    p.add_argument("--frame-end", type=int, default=-1)
    p.add_argument("--src-root", default="",
                   help="Override source root bone (else read from mapping/root_bones).")
    p.add_argument("--dst-root", default="",
                   help="Override Puppeteer root bone (else read from mapping/root_bones).")
    p.add_argument("--global-scale", type=float, default=1.0,
                   help="BVH import scale (reconcile BVH units with the rig).")
    p.add_argument("--root-scale", type=float, default=1.0,
                   help="Scale for root translation (0 disables).")
    p.add_argument("--max-delta-deg", type=float, default=170.0,
                   help="Clamp source per-bone rotation magnitude (deg). 0 disables.")
    p.add_argument("--bake-root-to-bone", action="store_true",
                   help="Bake root motion into the root joint instead of armature object.")
    p.add_argument("--correction-json", default="",
                   help="Optional per-bone correction quats: {dst_bone: [w,x,y,z]}.")
    p.add_argument("--anim-only", action="store_true",
                   help="Export armature animation only (for UE Existing Skeleton import).")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
