"""
Test: 3D avatar generation (TRELLIS.2 image-to-3D + PBR video preview)

Pipeline:
    1. Directly load an RGBA T-pose image from disk
       (produced by `test/test_tpose_gen.py`).
    2. TrellisModel:
       - image-to-3D → simplified voxel mesh
       - PBR preview video via HDR envmap                  (saved as *.mp4)
       - export to GLB via o_voxel.postprocess.to_glb      (saved as *.glb)
"""

import os
import sys
sys.path.insert(0, ".")

from PIL import Image
from models.gen_3d.trellis import TrellisModel
from operators.gen_ue_avatar.funcs.gen_3d_avatar import gen_3d_avatar

CFG = {
    "gen_3d_model": "microsoft/TRELLIS.2-4B",
    "envmap_path":  "assets/hdri/forest.exr",
    "device":       "cuda",
}

OUTPUT_DIR = "output"
TPOSE_FG_PATH = os.path.join(OUTPUT_DIR, "luffy_tpose_fg.png")  # from test_tpose_gen.py

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tpose_rgba = Image.open(TPOSE_FG_PATH)   # RGBA 1024x1024

    trellis = TrellisModel(
        CFG["gen_3d_model"],
        device=CFG["device"],
        envmap_path=CFG["envmap_path"],
    )

    glb_path   = os.path.join(OUTPUT_DIR, "luffy_tpose.glb")
    video_path = os.path.join(OUTPUT_DIR, "luffy_tpose.mp4")

    # return_intermediate=True also gives back the in-memory mesh object.
    result = gen_3d_avatar(
        tpose_rgba,
        trellis,
        output_path=glb_path,
        save_video=True,
        video_path=video_path,
        fps=15,
        return_intermediate=True,
    )

    print(f"3D mesh (GLB) saved   : {result['glb_path']}")
    print(f"PBR preview video     : {result['video_path']}")
    print(f"Mesh object           : {type(result['mesh']).__name__}")
