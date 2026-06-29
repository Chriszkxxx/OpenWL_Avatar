"""
Test: Avatar auto-rigging (Puppeteer skeleton + skinning)

Pipeline:
    1. PuppeteerModel.rig: 3D mesh (.glb) -> skeleton (skeleton GPT)
       -> skin weights (skinning net) -> rig `.txt`
    2. (optional) export a bind-pose FBX (mesh + armature + weights) for UE.

Requires GPU + Puppeteer checkpoints (see scripts/installing/install_puppeteer.sh)
and a bpy-capable interpreter for the FBX export step.

Override any path via env var, e.g.
    PUPPETEER_ROOT=/path/to/Puppeteer MESH=/path/to/char.glb python test/test_rigging.py
"""

import os
import sys
sys.path.insert(0, ".")

from operators.gen_ue_avatar.operator import UEAvatarOperator

CFG = {
    # Puppeteer source root (cloned by install_puppeteer.sh).
    "puppeteer_root": os.environ.get(
        "PUPPETEER_ROOT", "models/gen_3d/Puppeteer_main"
    ),
    "device": "cuda",
    "gpu": int(os.environ.get("GPU", "0")),
    # Optional: separate interpreters for torch rigging vs bpy export.
    "rigging_python": os.environ.get("RIGGING_PYTHON"),
    "bpy_python": os.environ.get("BPY_PYTHON"),
}

MESH = os.environ.get(
    "MESH", "models/gen_3d/Puppeteer_main/examples/luffi_clear/luffi_clear.glb"
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/rigging")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    op = UEAvatarOperator(CFG)

    # Mesh -> Puppeteer rig (skeleton + skin weights) + bind-pose FBX.
    rig = op.rig_avatar(MESH, output_dir=OUTPUT_DIR, export_fbx=True)

    print(f"Rig file (skeleton + skin) : {rig['rig_txt']}")
    print(f"Skeleton file              : {rig['skeleton_txt']}")
    print(f"Working OBJ                : {rig['mesh_obj']}")
    if "rigged_fbx" in rig:
        print(f"Bind-pose FBX              : {rig['rigged_fbx']}")
