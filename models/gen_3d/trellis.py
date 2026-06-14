"""
TrellisModel — wrapper for the Trellis2 image-to-3D pipeline.

Mirrors the style of `models.gen_image.qwen_edit.QwenEditModel`: a thin class
that loads the underlying pipeline once and exposes a single `image_to_3d`
method that runs inference, optionally renders a preview video, and exports a
`.glb` mesh.

Reference: see `models/gen_3d/example.py`.
"""

import os
import sys

# Must be set before importing cv2 / trellis2.
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# `trellis2` and `o_voxel` live under TRELLIS.2-main/ next to this file
# (models/gen_3d/TRELLIS.2-main/{trellis2,o_voxel}/...).
# Add that directory to sys.path so plain `import trellis2` / `import o_voxel` works.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRELLIS_ROOT = os.path.join(_HERE, "TRELLIS2_main")
if _TRELLIS_ROOT not in sys.path:
    sys.path.insert(0, _TRELLIS_ROOT)

from typing import Optional

import cv2
import imageio
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils

import o_voxel


class TrellisModel:
    """Thin wrapper around `Trellis2ImageTo3DPipeline`."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        envmap_path: Optional[str] = None,
        simplify_faces: int = 16_777_216,  # nvdiffrast limit
    ):
        """
        Args:
            model_path:     Local path or HF hub id for TRELLIS.2.
            device:         "cuda" / "cpu".
            envmap_path:    Optional HDR env map (.exr) used for video preview
                            and PBR rendering. If None, video rendering is skipped.
            simplify_faces: Face budget for `mesh.simplify(...)`.
        """
        self.model_path = model_path
        self.device = device
        self.simplify_faces = simplify_faces

        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model_path)
        if device == "cuda":
            self.pipeline.cuda()

        self.envmap = None
        if envmap_path and os.path.exists(envmap_path):
            hdr = cv2.cvtColor(
                cv2.imread(envmap_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
            )
            self.envmap = EnvMap(torch.tensor(hdr, dtype=torch.float32, device=device))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run(self, image: Image.Image):
        """Run the image-to-3D pipeline and return the simplified mesh object."""
        with torch.inference_mode():
            mesh = self.pipeline.run(image)[0]
        mesh.simplify(self.simplify_faces)
        return mesh

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def render_video(self, mesh, video_path: str, fps: int = 15) -> Optional[str]:
        """Render a PBR preview video. Requires `envmap_path` to be set."""
        if self.envmap is None:
            return None
        frames = render_utils.make_pbr_vis_frames(
            render_utils.render_video(mesh, envmap=self.envmap)
        )
        imageio.mimsave(video_path, frames, fps=fps)
        return video_path

    def export_glb(
        self,
        mesh,
        output_path: str,
        decimation_target: int = 1_000_000,
        texture_size: int = 4096,
        remesh: bool = True,
        remesh_band: int = 1,
        remesh_project: int = 0,
        verbose: bool = True,
    ) -> str:
        """Export the mesh as a `.glb` file via `o_voxel.postprocess.to_glb`."""
        # Resolve to an absolute path BEFORE running the pipeline, because
        # some downstream extensions (xatlas / cumesh / o_voxel) chdir into
        # a temp dir and may leave the process cwd dangling if it gets
        # cleaned up. Once cwd is gone, os.getcwd() raises FileNotFoundError
        # and any relative-path operation (incl. trimesh's export) breaks.
        output_path = os.path.abspath(os.path.expanduser(output_path))
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)

        glb = o_voxel.postprocess.to_glb(
            vertices          = mesh.vertices,
            faces             = mesh.faces,
            attr_volume       = mesh.attrs,
            coords            = mesh.coords,
            attr_layout       = mesh.layout,
            voxel_size        = mesh.voxel_size,
            aabb              = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target = decimation_target,
            texture_size      = texture_size,
            remesh            = remesh,
            remesh_band       = remesh_band,
            remesh_project    = remesh_project,
            verbose           = verbose,
        )

        # Guard against extensions that left cwd pointing to a deleted dir.
        try:
            os.getcwd()
        except (FileNotFoundError, OSError):
            os.chdir(out_dir)

        glb.export(output_path, extension_webp=False)
        return output_path

    # ------------------------------------------------------------------
    # High-level convenience API
    # ------------------------------------------------------------------

    def image_to_3d(
        self,
        image: Image.Image,
        output_path: str,
        video_path: Optional[str] = None,
        fps: int = 15,
        decimation_target: int = 1_000_000,
        texture_size: int = 4096,
    ) -> str:
        """
        Run image-to-3D inference and export mesh.

        Args:
            image:             RGBA PIL image (1024x1024, transparent background).
            output_path:       Path to save the output .glb mesh.
            video_path:        Optional .mp4 path; if given and an envmap is
                               available, also render a PBR preview video.
            fps:               FPS for the preview video.
            decimation_target: Triangle budget for the exported mesh.
            texture_size:      Baked texture resolution.

        Returns:
            Path to the saved .glb mesh.
        """
        mesh = self.run(image)
        if video_path:
            self.render_video(mesh, video_path, fps=fps)
        return self.export_glb(
            mesh,
            output_path,
            decimation_target=decimation_target,
            texture_size=texture_size,
        )
