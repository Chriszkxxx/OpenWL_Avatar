#!/usr/bin/env python3
"""Convert MoMask BVH motion to a Mixamo-format FBX (MoMask / Keemap workflow).

MoMask BVH skeletons don't share Mixamo's bone naming or root transform, so a
direct BVH -> Puppeteer retarget tends to misalign the root height and drop the
animation on UE import. This script does the first half of the recommended
two-step pipeline:

  1. Import the Mixamo bind rig (mixamorig:* bones) as the destination.
  2. Import the BVH motion (Hips, LeftArm, ...) as the source.
  3. Retarget BVH -> Mixamo via the world-delta math (`world_delta.bake_action`)
     and export an anim-only Mixamo FBX, ground-aligned.

Feed the output FBX to `world_delta` (Mixamo -> Puppeteer) for the final clip.

Run as a module::

    python -m models.gen_3d.puppeteer_retarget.bvh_to_mixamo \\
        --bvh motion.bvh --mixamo-ref char_mixamo.fbx \\
        --mapping mappings/momask_bvh_to_mixamo_mapping.json \\
        --output out_mixamo.fbx
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import bpy
from mathutils import Quaternion, Vector

from . import world_delta as _bake
from .rig_io import clear_bpy_data


def load_mapping(path: str) -> Tuple[Dict[str, str], str, str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    roots = data.get("root_bones", {})
    root_src = roots.get("source", "Hips")
    root_dst = roots.get("mixamo", roots.get("puppeteer", "mixamorig:Hips"))
    return data["bone_map"], root_src, root_dst


def load_momask_keemap_corrections(path: str) -> Dict[str, Quaternion]:
    """Optional per-bone quaternions from MoMask assets/mapping.json."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Quaternion] = {}
    for b in data.get("bones", []):
        dst = b.get("DestinationBoneName")
        if not dst:
            continue
        q = Quaternion((
            b.get("QuatCorrectionFactorw", 1.0),
            b.get("QuatCorrectionFactorx", 0.0),
            b.get("QuatCorrectionFactory", 0.0),
            b.get("QuatCorrectionFactorz", 0.0),
        )).normalized()
        if abs(q.angle - 0.0) > 1e-4:
            out[dst] = q
    return out


def import_bvh_animation(path: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.import_anim.bvh(
        filepath=path,
        axis_forward="-Z",
        axis_up="Y",
        global_scale=1.0,
    )
    arm = next(o for o in bpy.data.objects if o not in before and o.type == "ARMATURE")
    arm.name = "BVHSource"
    if not arm.animation_data or not arm.animation_data.action:
        raise RuntimeError(f"No animation in BVH: {path}")
    return arm


def import_mixamo_bind_rig(path: str) -> bpy.types.Object:
    """Import Mixamo character FBX as bind-pose destination rig (hide mesh)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    new_objs = [o for o in bpy.data.objects if o not in before]
    arm = next(o for o in new_objs if o.type == "ARMATURE")
    for o in new_objs:
        if o.type == "MESH":
            o.hide_viewport = True
            o.hide_render = True
    arm.name = "MixamoDest"
    if arm.animation_data:
        arm.animation_data.action = None
    _bake.reset_dest_pose(arm)
    arm.location = Vector((0.0, 0.0, 0.0))
    arm.rotation_euler = (0.0, 0.0, 0.0)
    arm.scale = Vector((1.0, 1.0, 1.0))
    return arm


def apply_root_translation(
    src_arm: bpy.types.Object,
    dst_arm: bpy.types.Object,
    root_src: str,
    root_dst: str,
    root_scale: float,
) -> None:
    src_root_pose = src_arm.matrix_world @ src_arm.pose.bones[root_src].matrix
    src_root_rest = src_arm.matrix_world @ src_arm.data.bones[root_src].matrix_local
    offset = (src_root_pose.translation - src_root_rest.translation) * root_scale
    dst_arm.location = offset
    dst_arm.pose.bones[root_dst].location = Vector((0.0, 0.0, 0.0))


def export_mixamo_anim_fbx(
    path: str,
    arm: bpy.types.Object,
    fps: int,
    frame_start: int,
    frame_end: int,
) -> None:
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(
        filepath=path,
        check_existing=False,
        use_selection=True,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_step=1,
        bake_anim_simplify_factor=0.0,
        apply_scale_options="FBX_SCALE_ALL",
        axis_forward="-Z",
        axis_up="Y",
        bake_space_transform=False,
        object_types={"ARMATURE"},
    )


def run(args: argparse.Namespace) -> None:
    clear_bpy_data()
    mapping, root_src, root_dst = load_mapping(args.mapping)

    # Patch the world-delta core's root handling for BVH (src=Hips, dst=mixamorig:Hips).
    def _root_apply(src_arm, dst_arm, bake_to_bone, root_scale):
        apply_root_translation(src_arm, dst_arm, root_src, root_dst, root_scale)

    _bake.apply_root_translation = _root_apply  # type: ignore[assignment]
    _bake.ROOT_MIX = root_src
    _bake.ROOT_PUP = root_dst

    correction: Dict[str, Quaternion] = {}
    if args.momask_keemap_json:
        correction = load_momask_keemap_corrections(args.momask_keemap_json)
        print(f"  Keemap corrections: {len(correction)} bones")

    print(f"[1/4] Import Mixamo bind rig: {args.mixamo_ref}")
    dst_arm = import_mixamo_bind_rig(args.mixamo_ref)
    print(f"  bones={len(dst_arm.data.bones)}")

    print(f"[2/4] Import MoMask BVH: {args.bvh}")
    src_arm = import_bvh_animation(args.bvh)
    act = src_arm.animation_data.action
    fs = int(args.frame_start if args.frame_start >= 0 else act.frame_range[0])
    fe = int(args.frame_end if args.frame_end >= 0 else act.frame_range[1])
    fps = args.fps if args.fps > 0 else 20
    print(f"  frames {fs}-{fe} @ {fps}fps")

    print("[3/4] Retarget BVH -> Mixamo (world-delta)...")
    action_name = args.action_name or Path(args.bvh).stem
    action = _bake.bake_action(
        src_arm,
        dst_arm,
        mapping,
        action_name=action_name,
        frame_start=fs,
        frame_end=fe,
        bake_root_to_bone=False,
        root_scale=args.root_scale,
        max_delta_deg=args.max_delta_deg,
        correction=correction,
    )
    print(f"  action '{action.name}' fcurves={len(action.fcurves)}")

    bpy.data.objects.remove(src_arm, do_unlink=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    print(f"[4/4] Export Mixamo FBX: {args.output}")
    export_mixamo_anim_fbx(args.output, dst_arm, fps=fps, frame_start=fs, frame_end=fe)

    meta = {
        "source_bvh": args.bvh,
        "mixamo_ref": args.mixamo_ref,
        "output": args.output,
        "action_name": action.name,
        "frame_start": fs,
        "frame_end": fe,
        "fps": fps,
        "bone_map": mapping,
        "pipeline": "momask_bvh_to_mixamo_fbx",
    }
    meta_path = os.path.splitext(args.output)[0] + "_info.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  metadata: {meta_path}")
    print("Done.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoMask BVH -> Mixamo FBX (Keemap-style).")
    p.add_argument("--bvh", required=True)
    p.add_argument("--mixamo-ref", required=True,
                   help="Mixamo character FBX (T-pose / bind rig with mixamorig:* bones).")
    p.add_argument("--mapping", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--momask-keemap-json", default="",
                   help="Optional MoMask assets/mapping.json for QuatCorrection factors.")
    p.add_argument("--action-name", default="")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--frame-start", type=int, default=-1)
    p.add_argument("--frame-end", type=int, default=-1)
    p.add_argument("--root-scale", type=float, default=1.0)
    p.add_argument("--max-delta-deg", type=float, default=170.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
