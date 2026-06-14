"""
gen_tpose.py — Generate a game-CG T-pose RGBA image from a reference character image.

Pipeline:
  1. Use a GenImageModel (Qwen-Image-Edit) to turn the reference image into
     a game-CG T-pose render on a pure white background.
  2. Use a *mask model* to extract the foreground character. Two flavors are
     supported transparently — the right branch is selected automatically:
       - RMBGModel        : returns a clean foreground probability mask.
       - DepthAnythingModel: returns a depth map, threshold + white-bg
                             suppression are applied to derive the mask.
  3. Post-process: validate alpha, tight-crop the foreground bbox, pad to a
     square canvas and resize to a fixed target size (1024 by default) so the
     result is directly consumable by downstream 3D pipelines (e.g. TRELLIS).
"""

from typing import Optional

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

TPOSE_PROMPT = (
    "Transform this character into a high-quality game CG T-pose reference sheet. "
    "The character should stand upright with arms extended horizontally at shoulder height (T-pose), "
    "facing directly forward, on a pure solid white background (RGB 255,255,255). "
    "Keep the character's original appearance, costume, colors, and style. "
    "Render in a clean game-art / anime-CG style with clear outlines. "
    "No background scenery, no shadows on the floor. "
    "The background must be a flat, uniform, solid white color only — no gradients, no shadows cast on background."
)


# ---------------------------------------------------------------------------
# Step 1: Generate T-pose CG image
# ---------------------------------------------------------------------------

def _generate_tpose_rgb(
    gen_model,
    ref_image: Image.Image,
    description: str,
    seed: int,
    steps: int,
) -> Image.Image:
    """Call the image-edit model to produce the T-pose CG image (white-bg RGB)."""
    prompt = TPOSE_PROMPT
    if description:
        prompt = f"Character description: {description}. " + prompt
    return gen_model.edit(ref_image, prompt=prompt, seed=seed, steps=steps)


# ---------------------------------------------------------------------------
# Step 2: Foreground extraction
# ---------------------------------------------------------------------------

def _is_depth_model(model) -> bool:
    """Detect whether the supplied tool model is a depth estimator."""
    cls = type(model).__name__.lower()
    if "depth" in cls:
        return True
    if "rmbg" in cls or "matting" in cls or "segment" in cls:
        return False
    # Fall back to attribute heuristic.
    return False


def _extract_foreground_via_depth(
    image: Image.Image,
    depth_model,
    white_thresh: int = 235,
    depth_quantile: float = 0.35,
) -> Image.Image:
    """Segment foreground using depth cues + white-bg suppression."""
    img = np.array(image.convert("RGB"))

    depth = depth_model.predict(image)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    white_mask = (
        (img[..., 0] >= white_thresh)
        & (img[..., 1] >= white_thresh)
        & (img[..., 2] >= white_thresh)
    )

    fg = binary_fill_holes(depth_norm >= depth_quantile) & ~white_mask
    fg = binary_dilation(binary_erosion(fg, iterations=2), iterations=3)

    rgba = np.array(image.convert("RGBA"))
    rgba[..., 3] = (fg * 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _extract_foreground_via_mask(
    image: Image.Image,
    mask_model,
    threshold: float = 0.5,
) -> Image.Image:
    """Segment foreground using a direct alpha-mask model (e.g. RMBG-1.4).

    The mask model is expected to return either a HxW float array in [0, 1]
    (foreground probability) or an RGBA PIL image. The continuous mask is
    used as-is for the alpha channel; a binary version is also computed for
    light morphological cleanup so post-processing can find a stable bbox.
    """
    out = mask_model.predict(image)

    if isinstance(out, Image.Image):
        rgba = np.array(out.convert("RGBA"))
        return Image.fromarray(rgba, "RGBA")

    mask = np.asarray(out, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[..., 0]
    # Normalize to [0, 1] just in case.
    mn, mx = float(mask.min()), float(mask.max())
    if mx > mn:
        mask = (mask - mn) / (mx - mn)

    # Light cleanup on the binary mask only used to validate / crop bbox;
    # the final alpha keeps the soft RMBG mask for nicer edges.
    binary = mask >= threshold
    binary = binary_dilation(binary_erosion(binary, iterations=1), iterations=1)
    soft_alpha = mask.copy()
    soft_alpha[~binary] = 0.0

    rgba = np.array(image.convert("RGBA"))
    rgba[..., 3] = np.clip(soft_alpha * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _extract_foreground(image: Image.Image, mask_model) -> Image.Image:
    """Dispatch to the correct extraction routine based on the model type."""
    if _is_depth_model(mask_model):
        return _extract_foreground_via_depth(image, mask_model)
    return _extract_foreground_via_mask(image, mask_model)


# ---------------------------------------------------------------------------
# Step 3: Post-process for downstream 3D
# ---------------------------------------------------------------------------

def _postprocess_for_trellis(
    rgba: Image.Image,
    size: int = 1024,
    white_thresh: int = 240,
    padding: float = 0.05,
) -> Image.Image:
    """Validate alpha, tight-crop, pad to square, resize."""
    arr = np.array(rgba).copy()

    fg_pixel_count = (arr[..., 3] > 0).sum()
    total_pixels = arr.shape[0] * arr.shape[1]
    if fg_pixel_count < total_pixels * 0.01:
        print("[warn] Mask alpha nearly empty; falling back to white-bg removal.")
        rgb = arr[..., :3]
        white = (
            (rgb[..., 0] >= white_thresh)
            & (rgb[..., 1] >= white_thresh)
            & (rgb[..., 2] >= white_thresh)
        )
        arr[white, 3] = 0
        arr[~white, 3] = 255

    alpha = arr[..., 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        h_box = y_max - y_min
        w_box = x_max - x_min
        pad_y = int(h_box * padding)
        pad_x = int(w_box * padding)

        img_h, img_w = alpha.shape
        y_min = max(0, y_min - pad_y)
        y_max = min(img_h, y_max + pad_y)
        x_min = max(0, x_min - pad_x)
        x_max = min(img_w, x_max + pad_x)

        arr = arr[y_min:y_max, x_min:x_max]
        print(f"[crop] Foreground bbox: ({x_min},{y_min})-({x_max},{y_max}), "
              f"cropped size: {arr.shape[1]}x{arr.shape[0]}")
    else:
        print("[warn] No foreground pixels found after alpha fix; skipping crop.")

    rgba = Image.fromarray(arr, "RGBA")

    w, h = rgba.size
    side = max(w, h)
    square = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    square.paste(rgba, ((side - w) // 2, (side - h) // 2))
    return square.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gen_tpose(
    ref_image: Image.Image,
    description: str,
    gen_model,
    mask_model=None,
    depth_model=None,  # backwards-compat alias
    seed: int = 42,
    steps: int = 40,
    target_size: int = 1024,
    return_intermediate: bool = False,
):
    """
    Generate a T-pose RGBA image from a reference character image.

    Args:
        ref_image:           Reference PIL image of the character.
        description:         Short text description of the character (can be empty).
        gen_model:           Loaded GenImageModel (e.g. QwenEditModel).
        mask_model:          Loaded foreground / matting model — `RMBGModel`
                             or `DepthAnythingModel`. The branch is selected
                             automatically based on the class name.
        depth_model:         Backwards-compatible alias for `mask_model`.
        seed:                Random seed for the image-edit model.
        steps:               Diffusion steps for the image-edit model.
        target_size:         Output canvas size (square).
        return_intermediate: If True, also return the white-bg RGB T-pose.

    Returns:
        RGBA PIL image (target_size x target_size, transparent background).
        If `return_intermediate=True`, returns a dict with keys
        {"tpose_rgb", "tpose_rgba"}.
    """
    if mask_model is None:
        mask_model = depth_model

    # Step 1: image edit → white-bg T-pose
    tpose_rgb = _generate_tpose_rgb(gen_model, ref_image, description, seed, steps)

    # Step 2: foreground extraction
    if mask_model is not None:
        fg_rgba = _extract_foreground(tpose_rgb, mask_model)
    else:
        # Direct fallback: postprocess will do white-bg removal.
        fg_rgba = tpose_rgb.convert("RGBA")

    # Step 3: post-process for downstream 3D
    fg_rgba = _postprocess_for_trellis(fg_rgba, size=target_size)

    if return_intermediate:
        return {"tpose_rgb": tpose_rgb, "tpose_rgba": fg_rgba}
    return fg_rgba
