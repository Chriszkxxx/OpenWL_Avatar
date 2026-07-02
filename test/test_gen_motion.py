"""
Test: Text-to-motion generation (MoMask) + optional retarget onto a rig

Pipeline:
    1. MoMaskModel.generate: text prompt -> HumanML3D BVH clip (20 fps).
    2. (optional) if a Puppeteer rig `.txt` + textured GLB are given, retarget
       the generated BVH directly onto the rigged avatar (world-delta, bpy),
       producing an animated FBX (mesh+anim and anim-only for UE).

Generation needs the MoMask checkpoints (see scripts/installing/install_momask.sh).
The optional retarget step additionally needs a bpy-capable interpreter; set:
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

Override any path/prompt via env var, e.g.
    PROMPT="a person jumps" python test/test_gen_motion.py
    GLB=... RIG=... python test/test_gen_motion.py   # also retarget onto the rig
"""

import os
import sys
sys.path.insert(0, ".")

from models.gen_3d.momask import MoMaskModel
from operators.gen_ue_avatar.funcs.gen_motion import gen_motion

CFG = {
    # MoMask source root (cloned by install_momask.sh).
    "momask_root": os.environ.get("MOMASK_ROOT", "models/gen_3d/momask_main"),
    "device": os.environ.get("DEVICE", "cuda"),
    "gpu": int(os.environ.get("GPU", "0")),
    # Optional: separate interpreters for MoMask (torch) vs bpy retarget.
    "motion_python": os.environ.get("MOTION_PYTHON"),
    "bpy_python": os.environ.get("BPY_PYTHON"),
}

PROMPT = os.environ.get("PROMPT", "walk forward, then perform a gear attack")
MOTION_LENGTH = int(os.environ.get("MOTION_LENGTH", "0"))  # 0 = MoMask estimates
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output/motion")

# Optional retarget target — set both to also apply the motion onto a rig
# (rig `.txt` from test_rigging.py + the textured GLB).
GLB = os.environ.get("GLB", "")
RIG = os.environ.get("RIG", "")

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    momask = MoMaskModel(
        model_path=CFG["momask_root"],
        device=CFG["device"],
        gpu=CFG["gpu"],
        python_bin=CFG["motion_python"],
    )

    # Retarget onto a rig only when both GLB + RIG are provided.
    puppeteer = None
    if GLB and RIG:
        from models.gen_3d.puppeteer import PuppeteerModel

        puppeteer = PuppeteerModel(
            model_path=os.environ.get("PUPPETEER_ROOT", "models/gen_3d/Puppeteer_main"),
            device="cpu",  # retarget is bpy-only
            bpy_python=CFG["bpy_python"],
        )

    result = gen_motion(
        PROMPT,
        momask,
        output_dir=OUTPUT_DIR,
        motion_length=MOTION_LENGTH,
        glb_path=GLB or None,
        rig_txt=RIG or None,
        puppeteer_model=puppeteer,
    )

    print(f"Generated BVH (20 fps)   : {result['bvh']}")
    if result.get("mp4"):
        print(f"Preview MP4              : {result['mp4']}")
    if result.get("npy"):
        print(f"Joints .npy              : {result['npy']}")
    if result.get("output"):
        print(f"Animated FBX (mesh+anim) : {result['output']}")
    if result.get("anim_only"):
        print(f"Anim-only FBX (UE)       : {result['anim_only']}")
