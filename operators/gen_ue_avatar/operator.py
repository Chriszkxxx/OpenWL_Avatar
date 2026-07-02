"""
UEAvatarOperator — top-level operator for the UE avatar generation pipeline.

Loads all required models once and exposes high-level pipeline methods.
Internal logic is delegated to funcs/*.py (one file per capability).

UE5 import (RPC) is handled by serving/ue_client.py, not this operator.

Typical usage:
    op = UEAvatarOperator(cfg)
    tpose   = op.gen_tpose(ref_image, description)
    mesh    = op.gen_3d_avatar(tpose)
    motion  = op.gen_motion(mesh, "walk forward")
    # then call serving/ue_client.py to push to UE5
"""

from models.gen_image.qwen_edit import QwenEditModel
from models.gen_3d.trellis import TrellisModel
from models.tools.depth_anything import DepthAnythingModel
from models.tools.rmbg import RMBGModel
from operators.gen_ue_avatar.funcs.gen_tpose import gen_tpose
from operators.gen_ue_avatar.funcs.gen_3d_avatar import gen_3d_avatar
from operators.gen_ue_avatar.funcs.gen_motion import detect_skeleton, gen_motion
from operators.gen_ue_avatar.funcs.rig_avatar import rig_avatar
from operators.gen_ue_avatar.funcs.retarget_motion import retarget_motion


class UEAvatarOperator:
    """Orchestrates the UE avatar AI generation pipeline (T-pose → 3D → motion)."""

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: Config dict with model paths, e.g.
                 {
                   "gen_image_model": "/path/to/qwen-edit",
                   "gen_3d_model":    "/path/to/trellis",
                   "device":          "cuda",
                 }
        """
        self.cfg = cfg
        device = cfg.get("device", "cuda")
        self.gen_image_model = QwenEditModel(cfg["gen_image_model"], device=device)
        self.gen_3d_model = TrellisModel(
            cfg["gen_3d_model"],
            device=device,
            envmap_path=cfg.get("envmap_path"),
        )

        # Foreground / matting model for T-pose extraction.
        # Priority:
        #   1. cfg["rmbg_model"]  → RMBGModel (recommended, RMBG-1.4)
        #   2. cfg["depth_model"] → DepthAnythingModel (legacy)
        rmbg_path  = cfg.get("rmbg_model")
        depth_path = cfg.get("depth_model")
        if rmbg_path:
            self.mask_model = RMBGModel(rmbg_path, device=device)
        elif depth_path:
            self.mask_model = DepthAnythingModel(depth_path, device=device)
        else:
            self.mask_model = None
        # Keep a legacy alias for backwards compatibility.
        self.depth_model = self.mask_model

        # Rigging / retarget model (Puppeteer). Optional + lazy: only loaded
        # when a rigging config is provided, so T-pose / 3D-only runs don't pay
        # for it.
        self.rigging_model = None
        if cfg.get("rigging_model") or cfg.get("puppeteer_root"):
            from models.gen_3d.puppeteer import PuppeteerModel

            self.rigging_model = PuppeteerModel(
                model_path=cfg.get("puppeteer_root") or cfg.get("rigging_model"),
                device=device,
                gpu=cfg.get("gpu", 0),
                python_bin=cfg.get("rigging_python"),
                bpy_python=cfg.get("bpy_python"),
            )

        # Text-to-motion model (MoMask). Optional + lazy: only loaded when a
        # motion config is provided.
        self.motion_model = None
        if cfg.get("motion_model") or cfg.get("momask_root"):
            from models.gen_3d.momask import MoMaskModel

            self.motion_model = MoMaskModel(
                model_path=cfg.get("momask_root") or cfg.get("motion_model"),
                device=device,
                gpu=cfg.get("gpu", 0),
                python_bin=cfg.get("motion_python") or cfg.get("rigging_python"),
            )

    # ------------------------------------------------------------------
    # Pipeline steps (thin wrappers that call funcs/)
    # ------------------------------------------------------------------

    def gen_tpose(self, ref_image, description: str = "", **kwargs):
        """Step 1: Generate T-pose RGBA image.

        Extra kwargs (seed, steps, target_size, return_intermediate, ...) are
        forwarded to `funcs.gen_tpose.gen_tpose`.
        """
        return gen_tpose(
            ref_image,
            description,
            gen_model=self.gen_image_model,
            mask_model=self.mask_model,
            **kwargs,
        )

    def gen_3d_avatar(self, tpose_image, **kwargs):
        """Step 2: Lift T-pose image to 3D avatar mesh.

        Extra kwargs (output_path, save_video, video_path, fps,
        decimation_target, texture_size, return_intermediate, ...) are
        forwarded to `funcs.gen_3d_avatar.gen_3d_avatar`.
        """
        return gen_3d_avatar(tpose_image, self.gen_3d_model, **kwargs)

    def rig_avatar(self, mesh_path: str, **kwargs):
        """Step 3a: Auto-rig the avatar mesh (Puppeteer skeleton + skinning).

        Returns a dict with the rig `.txt` (+ optional bind-pose FBX). Extra
        kwargs (output_dir, name, export_fbx, post_filter) are forwarded to
        `funcs.rig_avatar.rig_avatar`.
        """
        if self.rigging_model is None:
            raise RuntimeError(
                "No rigging model loaded. Pass cfg['puppeteer_root'] (or "
                "cfg['rigging_model']) when constructing UEAvatarOperator."
            )
        return rig_avatar(mesh_path, self.rigging_model, **kwargs)

    def retarget_motion(self, glb_path: str, rig_txt: str, motion_path: str, **kwargs):
        """Step 3b: Retarget a motion clip (Mixamo FBX / MoMask BVH) onto the rig.

        Extra kwargs (output_path, source, mapping, fps, export_anim_only, ...)
        are forwarded to
        `funcs.retarget_motion.retarget_motion`.
        """
        if self.rigging_model is None:
            raise RuntimeError(
                "No rigging model loaded. Pass cfg['puppeteer_root'] (or "
                "cfg['rigging_model']) when constructing UEAvatarOperator."
            )
        return retarget_motion(glb_path, rig_txt, motion_path, self.rigging_model, **kwargs)

    def gen_motion(self, mesh_path: str, motion_desc: str = "", motion_path: str = "", **kwargs):
        """Step 3: Rig the avatar, then give it motion.

        Precedence:
          1. `motion_path` set → retarget that existing clip (Mixamo FBX / BVH).
          2. `motion_desc` set + a MoMask model loaded → generate a novel motion
             from text and retarget it onto the rig.

        Args:
            mesh_path:   Path to the 3D avatar mesh (`.glb`).
            motion_desc: Text description for MoMask text-to-motion generation.
            motion_path: Optional existing motion clip (Mixamo FBX / MoMask BVH)
                         to retarget onto the rigged character.
            **kwargs:    Forwarded to `retarget_motion` (when `motion_path` set)
                         or `gen_motion` (when generating from `motion_desc`).

        Returns:
            The rig dict, augmented with motion / retarget outputs.
        """
        rig = self.rig_avatar(mesh_path)
        if motion_path:
            retarget = self.retarget_motion(
                mesh_path, rig["rig_txt"], motion_path, **kwargs
            )
            rig.update(retarget)
        elif motion_desc:
            if self.motion_model is None:
                raise RuntimeError(
                    "No motion model loaded. Pass cfg['momask_root'] (or "
                    "cfg['motion_model']) when constructing UEAvatarOperator to "
                    "enable text-to-motion generation."
                )
            motion = gen_motion(
                motion_desc,
                self.motion_model,
                glb_path=mesh_path,
                rig_txt=rig["rig_txt"],
                puppeteer_model=self.rigging_model,
                **kwargs,
            )
            rig.update(motion)
        return rig

    # ------------------------------------------------------------------
    # Convenience: run full pipeline end-to-end (returns assets for serving)
    # ------------------------------------------------------------------

    def run(
        self,
        ref_image,
        description: str = "",
        motion_desc: str = "",
        motion_path: str = "",
        **motion_kwargs,
    ) -> dict:
        """
        Run the full AI generation pipeline from a reference image.

        If a rigging model is loaded, also rig the mesh and (when `motion_path`
        is given) retarget that motion clip onto it.

        Returns:
            dict with keys: tpose_image, mesh_path, rig (or None), motion (or None)
            (caller is responsible for pushing results to UE5 via serving/ue_client.py)
        """
        tpose = self.gen_tpose(ref_image, description)
        mesh = self.gen_3d_avatar(tpose)

        rig = None
        if self.rigging_model is not None:
            rig = self.gen_motion(
                mesh, motion_desc=motion_desc, motion_path=motion_path, **motion_kwargs
            )
        return {
            "tpose_image": tpose,
            "mesh_path": mesh,
            "rig": rig,
            "motion_path": (rig or {}).get("output") if rig else None,
        }
