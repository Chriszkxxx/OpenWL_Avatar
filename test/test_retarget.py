"""
Test: Motion retarget onto a Puppeteer-rigged avatar (world-delta, bpy)

Pipeline:
    1. Reuse a Puppeteer rig `.txt` (from test_rigging.py) + the textured GLB.
    2. PuppeteerModel.retarget:
       - source="mixamo": Mixamo FBX -> animated FBX (mesh+anim and anim-only)
       - source="bvh"   : BVH (e.g. MoMask) -> animated FBX, *directly* (no
                          Mixamo intermediate; world-delta is roll-independent)

Only needs a bpy-capable interpreter (no GPU). Set:
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

Override any path via env var, e.g.
    GLB=... RIG=... MIXAMO_ANIM=... python test/test_retarget.py
    SOURCE=bvh BVH=... python test/test_retarget.py
"""

import os
import sys
sys.path.insert(0, ".")

from models.gen_3d.puppeteer import PuppeteerModel

_EX = "models/gen_3d/Puppeteer_main/examples/luffi_clear"

CFG = {
    "puppeteer_root": os.environ.get("PUPPETEER_ROOT", "models/gen_3d/Puppeteer_main"),
    "device": "cpu",  # retarget is bpy-only
    "bpy_python": os.environ.get("BPY_PYTHON"),
}

GLB = os.environ.get("GLB", f"{_EX}/luffi_clear.glb")
RIG = os.environ.get("RIG", "output/rigging/luffi_clear/skinning/generate/luffi_clear_skin.txt")
SOURCE = os.environ.get("SOURCE", "mixamo")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/motion")

# Mixamo path
MIXAMO_ANIM = os.environ.get("MIXAMO_ANIM", f"{_EX}/Slow Run.fbx")
# BVH path (e.g. MoMask) — retargeted directly, no Mixamo reference needed.
BVH = os.environ.get("BVH", "")
GLOBAL_SCALE = float(os.environ.get("GLOBAL_SCALE", "1.0"))
FPS = int(os.environ.get("FPS", "30" if SOURCE == "mixamo" else "20"))

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = PuppeteerModel(
        model_path=CFG["puppeteer_root"],
        device=CFG["device"],
        bpy_python=CFG["bpy_python"],
    )

    if SOURCE == "bvh":
        assert BVH, "source=bvh needs the BVH env var (path to a .bvh file)."
        motion = BVH
        out = os.path.join(OUTPUT_DIR, "momask_on_luffi_clear.fbx")
        kwargs = dict(source="bvh", global_scale=GLOBAL_SCALE)
    else:
        motion = MIXAMO_ANIM
        out = os.path.join(OUTPUT_DIR, "slow_run_on_luffi_clear.fbx")
        kwargs = dict(source="mixamo")

    from operators.gen_ue_avatar.funcs.retarget_motion import retarget_motion

    result = retarget_motion(
        glb_path=GLB,
        rig_txt=RIG,
        motion_path=motion,
        model=model,
        output_path=out,
        fps=FPS,
        action_name="Take 001",
        export_anim_only=True,
        **kwargs,
    )

    print(f"Animated FBX (mesh+anim) : {result['output']}")
    if result.get("intermediate"):
        print(f"Intermediate Mixamo FBX  : {result['intermediate']}")
    if result.get("anim_only"):
        print(f"Anim-only FBX (UE)       : {result['anim_only']}")
