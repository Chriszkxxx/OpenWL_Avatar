#!/usr/bin/env python3
"""Dump full hierarchy + world-space head/tail of one or more FBX armatures
to a single text file. Used to design / debug mapping rules from real data.

Vendored from Puppeteer's `skeleton_retarget_refine_code5/tools/`. Requires a
`bpy`-capable interpreter. Run as a package module::

    python -m models.gen_3d.puppeteer_retarget.mappings.inspect_skeleton \\
        --output output/skeletons_dump.txt \\
        char_puppeteer_ue.fbx any_mixamo_char.fbx
"""
from __future__ import annotations
import argparse
import sys

import bpy


def clear():
    for d in (bpy.data.objects, bpy.data.armatures, bpy.data.meshes,
              bpy.data.actions, bpy.data.materials):
        for x in list(d):
            d.remove(x, do_unlink=True)


def dump_tree(arm, name, depth, out):
    b = arm.data.bones[name]
    h = arm.matrix_world @ b.head_local
    t = arm.matrix_world @ b.tail_local
    out.append(
        f"{'  ' * depth}- {name}  "
        f"head=({h.x:+.3f},{h.y:+.3f},{h.z:+.3f})  "
        f"tail=({t.x:+.3f},{t.y:+.3f},{t.z:+.3f})  "
        f"len={(t-h).length:.3f}  children={len(b.children)}"
    )
    for c in b.children:
        dump_tree(arm, c.name, depth + 1, out)


def dump_fbx(path, out):
    clear()
    bpy.ops.import_scene.fbx(filepath=path)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    out.append(f"\n========== {path} ==========")
    out.append(f"# Armature: {arm.name}  bones={len(arm.data.bones)}")
    out.append(f"# matrix_world Euler XYZ = "
               f"{[round(x,3) for x in arm.rotation_euler]}")
    for r in [b.name for b in arm.data.bones if b.parent is None]:
        dump_tree(arm, r, 0, out)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("fbx", nargs="+")
    args = p.parse_args(argv)

    lines = []
    for f in args.fbx:
        dump_fbx(f, lines)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    main(argv)
