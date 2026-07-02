#!/usr/bin/env python3
"""Auto-generate a Mixamo -> Puppeteer bone mapping from skeleton topology + geometry.

No hardcoded joint indices. Both armatures are inspected structurally:

  * HIPS  = multi-child bone with the biggest subtree (pelvis hub)
  * legs vs spine = HIPS children split by world-Z relative to HIPS
  * chest = first multi-child junction walking up the spine
  * neck/arms = chest children, neck is highest-Z, the other two are shoulders
  * left/right = resolved by world-X sign AFTER applying Mixamo's 180-deg Z
    facing-fix, so both rigs live in one shared frame
  * limb chains = walk single-child descendants until we hit a branch
    (hand stops naturally at 5-finger fan-out; foot stops at toe)

Vendored from Puppeteer's `skeleton_retarget_refine_code5/tools/`. Requires a
`bpy`-capable interpreter (`pip install bpy==4.2.0`) or Blender's bundled
Python. Run as a package module so it resolves inside OpenWL-Avatar::

    python -m models.gen_3d.puppeteer_retarget.mappings.generate_mapping_auto \\
        --puppeteer-fbx examples/luffi_clear/luffi_clear_puppeteer_ue.fbx \\
        --mixamo-fbx    examples/luffi_clear/any_mixamo_char.fbx \\
        --output        models/gen_3d/puppeteer_retarget/mappings/my_mapping.json

or via `PuppeteerModel.generate_mapping(...)`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import bpy
from mathutils import Matrix, Vector

# NOTE: Despite the old hardcoded mapping's comment claiming the two rigs
# face opposite directions and need a 180-deg Z fix, real data shows BOTH
# Puppeteer (joint17/34 toe tails at Y<0) and Mixamo (toe tails at Y<0) face
# -Y in world space. Applying any flip is what caused the original leg-mirror
# and arm-swap bugs. We compare raw world-X directly.
FRAME_FIX = Matrix.Identity(4)

# ---------------------------------------------------------------- bpy utils ---


def clear_scene() -> None:
    for d in (bpy.data.objects, bpy.data.armatures, bpy.data.meshes,
              bpy.data.actions, bpy.data.materials):
        for x in list(d):
            d.remove(x, do_unlink=True)


def import_fbx_armature(path: str, name: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=path)
    arm = next(o for o in bpy.data.objects if o not in before and o.type == "ARMATURE")
    arm.name = name
    return arm


def whead(arm: bpy.types.Object, name: str) -> Vector:
    return arm.matrix_world @ arm.data.bones[name].head_local


# ----------------------------------------------------- topology helpers ---


def subtree_size(arm: bpy.types.Object, name: str, cache: Dict[str, int]) -> int:
    if name in cache:
        return cache[name]
    s = 1 + sum(subtree_size(arm, c.name, cache) for c in arm.data.bones[name].children)
    cache[name] = s
    return s


def find_hips(arm: bpy.types.Object) -> str:
    """Largest subtree among multi-child junctions; falls back to bone with most children."""
    cache: Dict[str, int] = {}
    multi = [b.name for b in arm.data.bones if len(b.children) >= 3]
    if not multi:
        multi = [b.name for b in arm.data.bones if len(b.children) >= 2]
    if not multi:
        # Pathological; pick the root.
        return next(b.name for b in arm.data.bones if b.parent is None)
    return max(multi, key=lambda n: subtree_size(arm, n, cache))


def walk_chain(arm: bpy.types.Object, start: str, max_len: int = 64) -> List[str]:
    """Follow single-child descendants from `start` until branch or leaf."""
    chain = [start]
    cur = arm.data.bones[start]
    while len(chain) < max_len and len(cur.children) == 1:
        cur = cur.children[0]
        chain.append(cur.name)
    return chain


def first_branch(arm: bpy.types.Object, start: str) -> Optional[str]:
    """First descendant (inclusive) with >= 2 children. None if leaf reached."""
    cur = arm.data.bones[start]
    while True:
        if len(cur.children) >= 2:
            return cur.name
        if not cur.children:
            return None
        cur = cur.children[0]


# ---------------------------------------------------- structural classify ---


def classify(arm: bpy.types.Object, fix: Matrix) -> Dict[str, object]:
    """Return roles for one armature:
        hips:   str
        spine:  List[str]   (hips ... head/neck-tip)
        legs:   Tuple[List[str], List[str]]
        arms:   Tuple[List[str], List[str]]

    NOTE: For hub bones (joint23-style) all children share the same head,
    so we classify by each child's *chain tip* head position, not by the
    child's own head.
    """
    bones = arm.data.bones

    def fpos(n: str) -> Vector:
        return fix @ whead(arm, n)

    def tip_of(start: str) -> str:
        """End of the single-child chain starting at `start` (or first branch)."""
        return walk_chain(arm, start)[-1]

    def tip_pos(start: str) -> Vector:
        return fpos(tip_of(start))

    hips = find_hips(arm)
    hp = fpos(hips)
    hc = [c.name for c in bones[hips].children]

    # Split hips children by direction of their CHAIN TIP relative to hips:
    #   tip above hips -> spine root
    #   tip below hips -> leg root
    legs_root = [n for n in hc if tip_pos(n).z < hp.z]
    spine_cands = [n for n in hc if tip_pos(n).z >= hp.z]
    if not spine_cands:
        spine_cands = [max(hc, key=lambda n: tip_pos(n).z)]
        legs_root = [n for n in hc if n not in spine_cands]
    # If multiple "spine candidates" (e.g., extra helper bones), pick the one
    # whose chain tip is highest.
    spine_root = max(spine_cands, key=lambda n: tip_pos(n).z)

    # Walk spine until first branch (chest).
    chest = first_branch(arm, spine_root)
    pre_chest: List[str] = []
    cur = bones[spine_root]
    while cur.name != (chest or cur.name):
        pre_chest.append(cur.name)
        cur = cur.children[0]
    if chest:
        pre_chest.append(chest)

    arms_root: List[str] = []
    neck_chain: List[str] = []
    if chest:
        cc = [c.name for c in bones[chest].children]
        # Chest children share head too -> classify by chain tip:
        #   neck/head tip is highest in Z; arms tips spread out in |X|.
        neck_root = max(cc, key=lambda n: tip_pos(n).z)
        arms_root = [n for n in cc if n != neck_root]
        neck_chain = walk_chain(arm, neck_root)
    else:
        neck_chain = walk_chain(arm, spine_root)
        pre_chest = []

    # Sort arms/legs by chain-tip X (not root X, which is shared on hub).
    legs_root.sort(key=lambda n: tip_pos(n).x)
    arms_root.sort(key=lambda n: tip_pos(n).x)

    leg_chains = tuple(walk_chain(arm, r)[:4] for r in legs_root[:2])
    arm_chains = tuple(walk_chain(arm, r)[:4] for r in arms_root[:2])

    spine_full = [hips] + pre_chest + neck_chain
    return {
        "hips": hips,
        "spine": spine_full,
        "legs": leg_chains,
        "arms": arm_chains,
    }


# ---------------------------------------------------------- mapping build ---


def left_sign_from_mixamo(mix_arm: bpy.types.Object) -> int:
    """Return +1 or -1: world-X sign that corresponds to character-LEFT,
    derived from Mixamo's named bones (LeftUpLeg is at the LEFT side)."""
    hips = whead(mix_arm, "mixamorig:Hips")
    lup = whead(mix_arm, "mixamorig:LeftUpLeg")
    return 1 if (lup.x - hips.x) >= 0 else -1


def split_lr(chains: Tuple[List[str], List[str]],
             arm: bpy.types.Object, fix: Matrix,
             left_sign: int) -> Tuple[List[str], List[str]]:
    """Return (left_chain, right_chain). Use the CHAIN TIP X (not the root X,
    which on a hub like joint23 is shared between both sides)."""
    if not chains:
        return [], []
    a, b = chains[0], chains[1] if len(chains) > 1 else []
    if not b:
        return (a, []) if left_sign > 0 else ([], a)
    ax = (fix @ whead(arm, a[-1])).x
    bx = (fix @ whead(arm, b[-1])).x
    return (a, b) if ax * left_sign > bx * left_sign else (b, a)


def build_mapping(mix_arm: bpy.types.Object,
                  pup_arm: bpy.types.Object) -> Tuple[Dict[str, str], Dict[str, dict]]:
    # Both rigs are inspected in raw world space (no facing flip).
    mix = classify(mix_arm, Matrix.Identity(4))
    pup = classify(pup_arm, Matrix.Identity(4))

    left_sign = left_sign_from_mixamo(mix_arm)
    mL, mR = split_lr(mix["legs"], mix_arm, Matrix.Identity(4), left_sign)
    pL, pR = split_lr(pup["legs"], pup_arm, Matrix.Identity(4), left_sign)
    maL, maR = split_lr(mix["arms"], mix_arm, Matrix.Identity(4), left_sign)
    paL, paR = split_lr(pup["arms"], pup_arm, Matrix.Identity(4), left_sign)

    def zip_chain(src: List[str], dst: List[str]) -> List[Tuple[str, str]]:
        n = min(len(src), len(dst))
        return list(zip(src[:n], dst[:n]))

    pairs: List[Tuple[str, str]] = []
    pairs += zip_chain(mix["spine"], pup["spine"])
    pairs += zip_chain(maL, paL)
    pairs += zip_chain(maR, paR)
    pairs += zip_chain(mL, pL)
    pairs += zip_chain(mR, pR)

    bone_map: Dict[str, str] = {}
    for s, d in pairs:
        bone_map.setdefault(s, d)

    chains = {
        "spine":     {"mixamo": mix["spine"], "puppeteer": pup["spine"]},
        "left_arm":  {"mixamo": maL,          "puppeteer": paL},
        "right_arm": {"mixamo": maR,          "puppeteer": paR},
        "left_leg":  {"mixamo": mL,           "puppeteer": pL},
        "right_leg": {"mixamo": mR,           "puppeteer": pR},
    }
    # Trim chains to the shortest of each pair so JSON is consistent with bone_map.
    for v in chains.values():
        n = min(len(v["mixamo"]), len(v["puppeteer"]))
        v["mixamo"] = v["mixamo"][:n]
        v["puppeteer"] = v["puppeteer"][:n]

    return bone_map, chains


# -------------------------------------------------------------- output ---


def write_json(path: str, bone_map: Dict[str, str], chains: Dict[str, dict],
               mix_root: str, pup_root: str) -> None:
    payload = {
        "description": "Mixamo (source) -> Puppeteer (target). Auto-generated from FBX topology.",
        "source_skeleton": "Mixamo",
        "target_skeleton": "Puppeteer",
        "root_bones": {"mixamo": mix_root, "puppeteer": pup_root},
        "bone_map": bone_map,
        "retarget_chains": chains,
        "notes": [
            "Mapping inferred by walking hierarchies + world-pose geometry; no joint indices hardcoded.",
            "Left/right sides determined via world-X sign after Mixamo 180-deg Z facing-fix.",
            "Chain length = min(mixamo, puppeteer) so bone_map and retarget_chains stay consistent.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------- entry ---


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--puppeteer-fbx", required=True)
    p.add_argument("--mixamo-fbx", required=True,
                   help="Mixamo FBX (animation or T-pose); first frame is used as bind reference.")
    p.add_argument("--output", required=True)
    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    clear_scene()
    print(f"[1/3] Importing Puppeteer: {args.puppeteer_fbx}")
    pup = import_fbx_armature(args.puppeteer_fbx, "Puppeteer")
    print(f"[2/3] Importing Mixamo:    {args.mixamo_fbx}")
    mix = import_fbx_armature(args.mixamo_fbx, "Mixamo")
    bpy.context.scene.frame_set(int(bpy.context.scene.frame_start))

    print("[3/3] Inferring mapping from topology + geometry...")
    bone_map, chains = build_mapping(mix, pup)

    print(f"  matched {len(bone_map)} bones across "
          f"{sum(1 for v in chains.values() if v['mixamo'])} chains")
    for k, v in chains.items():
        print(f"    {k:10s} ({len(v['mixamo']):d}): "
              f"{v['mixamo']}  ->  {v['puppeteer']}")

    write_json(args.output, bone_map, chains,
               mix_root="mixamorig:Hips", pup_root=bone_map.get("mixamorig:Hips", ""))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)
