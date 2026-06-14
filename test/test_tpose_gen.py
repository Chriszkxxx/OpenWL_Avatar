"""
Test: T-pose generation (Qwen-Image-Edit + RMBG-1.4)

Pipeline:
    1. UEAvatarOperator.gen_tpose: reference image → game-CG T-pose
       - Qwen-Image-Edit  → white-bg T-pose CG render          (saved as *_tpose.png)
       - RMBG-1.4         → foreground mask → transparent RGBA
       - tight-crop + 1024x1024 square canvas                  (saved as *_tpose_fg.png)
You can change to depth_anything by modifying the CFG to test the legacy depth-based matting branch, but RMBG-1.4 is recommended for better quality and robustness.
"""

import os
import sys
sys.path.insert(0, ".")

from PIL import Image
from operators.gen_ue_avatar.operator import UEAvatarOperator

CFG = {
    "gen_image_model": "Qwen/Qwen-Image-Edit-2511",
    "gen_3d_model":    "microsoft/TRELLIS.2-4B",
    "rmbg_model":      "briaai/RMBG-1.4",
    "device":          "cuda",
}

OUTPUT_DIR = "output"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ref_image   = Image.open("assets/luffy.jpg")
    description = "Monkey D. Luffy, the main protagonist of the One Piece anime series, wearing his iconic red vest and straw hat, with a cheerful expression and his signature stretched rubber body pose. I want him to use basketball as weapon."

    op = UEAvatarOperator(CFG)

    # T-pose generation; return_intermediate=True also gives the white-bg RGB image.
    result = op.gen_tpose(ref_image, description=description, return_intermediate=True)
    tpose_rgb  = result["tpose_rgb"]
    tpose_rgba = result["tpose_rgba"]

    rgb_path  = os.path.join(OUTPUT_DIR, "luffy_tpose.png")
    rgba_path = os.path.join(OUTPUT_DIR, "luffy_tpose_fg.png")

    tpose_rgb.save(rgb_path)
    tpose_rgba.save(rgba_path)

    print(f"White-bg T-pose saved : {rgb_path}  size={tpose_rgb.size}  mode={tpose_rgb.mode}")
    print(f"Transparent T-pose    : {rgba_path}  size={tpose_rgba.size}  mode={tpose_rgba.mode}")
