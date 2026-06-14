"""
gen_3d_avatar.py — Generate a 3D avatar mesh from a T-pose RGBA image.

Pipeline:
  1. Feed the T-pose RGBA image into a Gen3D model (Trellis2) to obtain a
     voxel-based mesh.
  2. Simplify the mesh, optionally render a PBR preview video using an HDR
     environment map.
  3. Post-process and export the mesh as a `.glb` file via `o_voxel`.

Input:
    tpose_image : PIL.Image (RGBA)  — T-pose character with transparent background
    model       : TrellisModel      — loaded image-to-3D model

Output:
    mesh_path   : str               — path to the exported `.glb` mesh file
"""

import os
from typing import Optional

from PIL import Image


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR    = "output"
DEFAULT_GLB_NAME      = "avatar.glb"
DEFAULT_DECIMATION    = 1_000_000
DEFAULT_TEXTURE_SIZE  = 4096
DEFAULT_VIDEO_FPS     = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_output_paths(
    output_path: Optional[str],
    save_video: bool,
    video_path: Optional[str],
):
    """Resolve final glb / video output paths and ensure the directory exists."""
    glb_path = output_path or os.path.join(DEFAULT_OUTPUT_DIR, DEFAULT_GLB_NAME)
    os.makedirs(os.path.dirname(os.path.abspath(glb_path)) or ".", exist_ok=True)

    vid_path = None
    if save_video:
        vid_path = video_path or os.path.splitext(glb_path)[0] + ".mp4"
        os.makedirs(os.path.dirname(os.path.abspath(vid_path)) or ".", exist_ok=True)
    return glb_path, vid_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gen_3d_avatar(
    tpose_image: Image.Image,
    model,
    output_path: Optional[str] = None,
    save_video: bool = False,
    video_path: Optional[str] = None,
    fps: int = DEFAULT_VIDEO_FPS,
    decimation_target: int = DEFAULT_DECIMATION,
    texture_size: int = DEFAULT_TEXTURE_SIZE,
    return_intermediate: bool = False,
):
    """
    Generate a 3D avatar mesh from a T-pose RGBA image.

    Args:
        tpose_image:         RGBA PIL image of the character in T-pose.
        model:               Loaded `TrellisModel` instance.
        output_path:         Destination `.glb` file path. If None, defaults to
                             `output/avatar.glb`.
        save_video:          If True, render and save a PBR preview video
                             (requires `model.envmap` to be available).
        video_path:          Destination `.mp4` file path. If None and
                             `save_video=True`, falls back to `<glb_stem>.mp4`.
        fps:                 FPS for the preview video.
        decimation_target:   Triangle budget for the exported mesh.
        texture_size:        Baked texture resolution.
        return_intermediate: If True, return a dict containing both the mesh
                             object and the exported paths.

    Returns:
        Path to the saved `.glb` mesh (default), or a dict with keys
        {"mesh", "glb_path", "video_path"} if `return_intermediate=True`.
    """
    glb_path, vid_path = _resolve_output_paths(output_path, save_video, video_path)

    # Step 1: image-to-3D inference (+ simplify)
    mesh = model.run(tpose_image)

    # Step 2 (optional): render PBR preview video
    if save_video:
        rendered = model.render_video(mesh, vid_path, fps=fps)
        if rendered is None:
            print("[warn] save_video=True but model has no envmap; skipping video.")
            vid_path = None

    # Step 3: export GLB
    model.export_glb(
        mesh,
        glb_path,
        decimation_target=decimation_target,
        texture_size=texture_size,
    )
    print(f"[gen_3d_avatar] mesh exported: {glb_path}")
    if vid_path:
        print(f"[gen_3d_avatar] preview video: {vid_path}")

    if return_intermediate:
        return {"mesh": mesh, "glb_path": glb_path, "video_path": vid_path}
    return glb_path
