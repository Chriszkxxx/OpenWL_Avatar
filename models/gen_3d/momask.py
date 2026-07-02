"""
MoMaskModel — wrapper for the MoMask text-to-motion generation pipeline.

Mirrors the style of `models.gen_3d.puppeteer.PuppeteerModel`: a thin class that
points at the MoMask source tree (cloned to `models/gen_3d/momask_main/` by
`scripts/installing/install_momask.sh`, just like `Puppeteer_main`) and drives
its upstream entrypoint (`gen_t2m.py`) as a subprocess.

MoMask belongs under `gen_3d` because it is the *motion generation* model
referenced in `models/gen_3d/__init__.py` ("2. 生成motion"): given a text
description it generates a 3D human motion (HumanML3D 22-joint skeleton) and
writes it out as a `.bvh` clip, which the Puppeteer world-delta retarget engine
(`source="bvh"`) transfers directly onto a rigged avatar.

Why subprocess instead of in-process import (like `puppeteer.py`, unlike
`trellis.py`): MoMask ships its own top-level `models/`, `utils/`, `options/`
and `common/` packages whose names collide with OpenWL-Avatar's own `models`
package, and its scripts assume their own working directory (checkpoints under
`./checkpoints`, results under `./generation`). Running `gen_t2m.py` in a
subprocess with `cwd` = the MoMask root keeps those namespaces and relative
paths isolated and matches how upstream is meant to run.
"""

import glob
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

# MoMask source root, cloned next to this file by install_momask.sh
# (models/gen_3d/momask_main/{gen_t2m.py,checkpoints,...}).
DEFAULT_MOMASK_ROOT = os.path.join(_HERE, "momask_main")

# Default HumanML3D checkpoint names (match MoMask's options/base_option.py +
# options/eval_option.py, i.e. what `bash prepare/download_models.sh` fetches).
DEFAULT_T2M_NAME = "t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns"
DEFAULT_RES_NAME = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"


class MoMaskModel:
    """Thin wrapper around the MoMask text-to-motion generation pipeline."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        gpu: int = 0,
        dataset_name: str = "t2m",
        name: str = DEFAULT_T2M_NAME,
        res_name: str = DEFAULT_RES_NAME,
        python_bin: Optional[str] = None,
    ):
        """
        Args:
            model_path:   MoMask source root. Defaults to
                          `models/gen_3d/momask_main`.
            device:       "cuda" / "cpu". Maps to MoMask's `--gpu_id`
                          (cpu -> -1).
            gpu:          CUDA device index passed as `--gpu_id` when device is
                          "cuda" (also sets CUDA_VISIBLE_DEVICES for the child).
            dataset_name: MoMask dataset / checkpoint family ("t2m" for
                          HumanML3D, "kit" for KIT-ML).
            name:         Masked-transformer checkpoint name (under
                          `checkpoints/<dataset_name>/<name>`).
            res_name:     Residual-transformer checkpoint name.
            python_bin:   Python interpreter for the MoMask subprocess. Defaults
                          to the current interpreter.
        """
        self.model_path = os.path.abspath(model_path or DEFAULT_MOMASK_ROOT)
        self.device = device
        self.gpu = gpu
        self.dataset_name = dataset_name
        self.name = name
        self.res_name = res_name
        self.python_bin = python_bin or sys.executable

    @property
    def gpu_id(self) -> int:
        """MoMask `--gpu_id` convention: -1 = CPU, otherwise the CUDA index."""
        return -1 if self.device == "cpu" else self.gpu

    # ------------------------------------------------------------------
    # Subprocess helper
    # ------------------------------------------------------------------

    def _run(self, cmd: List[str], cwd: str) -> None:
        env = os.environ.copy()
        if self.device != "cpu":
            env.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu))
        print(f"[momask] $ (cwd={cwd})\n  {' '.join(cmd)}")
        subprocess.run(cmd, cwd=cwd, env=env, check=True)

    # ------------------------------------------------------------------
    # Text -> motion (BVH)
    # ------------------------------------------------------------------

    def generate(
        self,
        text_prompt: str,
        output_dir: str,
        name: Optional[str] = None,
        motion_length: int = 0,
        repeat_times: int = 1,
        cond_scale: float = 4.0,
        time_steps: int = 18,
        temperature: float = 1.0,
        seed: int = 10107,
        use_ik: bool = True,
    ) -> Dict[str, str]:
        """
        Generate a novel motion clip from a text prompt and return its `.bvh`.

        Runs MoMask's `gen_t2m.py` as a subprocess (masked + residual
        transformers -> RVQ decode -> IK -> BVH), then copies the chosen BVH
        (and its companion `.npy` joints / `.mp4` preview if present) into
        `output_dir`.

        Args:
            text_prompt:   Natural-language motion description (e.g. "a person
                           walks forward and waves").
            output_dir:    Directory to copy the final motion artifacts into.
            name:          Base name for the copied outputs (defaults to a slug
                           of the prompt).
            motion_length: Number of poses to generate (0 = let MoMask's length
                           estimator decide). Motion is in 20 fps.
            repeat_times:  Number of generations for the prompt (the first,
                           repeat 0, is returned).
            cond_scale:    Classifier-free guidance scale.
            time_steps:    Mask-generate iterations for the masked transformer.
            temperature:   Sampling temperature.
            seed:          RNG seed for reproducibility.
            use_ik:        Return the foot-IK-corrected BVH (`*_ik.bvh`) when
                           available, else the raw BVH.

        Returns:
            dict with keys: {"bvh", "bvh_raw", "bvh_ik"?, "npy"?, "mp4"?,
            "name", "fps", "ext"}. `bvh` is the clip to feed into
            `PuppeteerModel.retarget(..., source="bvh", fps=20)`.
        """
        out = os.path.abspath(output_dir)
        os.makedirs(out, exist_ok=True)
        name = name or _slugify(text_prompt)

        # Unique --ext so concurrent / repeated runs don't clobber each other
        # inside MoMask's ./generation/<ext>/ folder.
        ext = f"openwl_{name}"

        cmd = [
            self.python_bin, "gen_t2m.py",
            "--gpu_id", str(self.gpu_id),
            "--dataset_name", self.dataset_name,
            "--name", self.name,
            "--res_name", self.res_name,
            "--ext", ext,
            "--text_prompt", text_prompt,
            "--motion_length", str(motion_length),
            "--repeat_times", str(repeat_times),
            "--cond_scale", str(cond_scale),
            "--time_steps", str(time_steps),
            "--temperature", str(temperature),
            "--seed", str(seed),
        ]
        self._run(cmd, cwd=self.model_path)

        # gen_t2m.py writes to ./generation/<ext>/{animations,joints}/<sample>/.
        # For a single --text_prompt the sample index is 0.
        gen_root = os.path.join(self.model_path, "generation", ext)
        anim_dir = os.path.join(gen_root, "animations", "0")
        joints_dir = os.path.join(gen_root, "joints", "0")

        raw_bvhs = sorted(
            b for b in glob.glob(os.path.join(anim_dir, "*.bvh"))
            if not b.endswith("_ik.bvh")
        )
        ik_bvhs = sorted(glob.glob(os.path.join(anim_dir, "*_ik.bvh")))
        if not raw_bvhs and not ik_bvhs:
            raise RuntimeError(
                f"MoMask produced no .bvh in {anim_dir}. Check that checkpoints "
                f"'{self.name}' / '{self.res_name}' exist under "
                f"{os.path.join(self.model_path, 'checkpoints', self.dataset_name)}."
            )

        result: Dict[str, object] = {"name": name, "fps": 20, "ext": ext}

        if raw_bvhs:
            result["bvh_raw"] = _copy(raw_bvhs[0], os.path.join(out, f"{name}.bvh"))
        if ik_bvhs:
            result["bvh_ik"] = _copy(ik_bvhs[0], os.path.join(out, f"{name}_ik.bvh"))

        # Preferred clip for downstream retarget.
        if use_ik and "bvh_ik" in result:
            result["bvh"] = result["bvh_ik"]
        else:
            result["bvh"] = result.get("bvh_raw") or result.get("bvh_ik")

        npy = _pick(glob.glob(os.path.join(joints_dir, "*.npy")), "_ik.npy", use_ik)
        if npy:
            result["npy"] = _copy(npy, os.path.join(out, f"{name}.npy"))

        mp4 = _pick(glob.glob(os.path.join(anim_dir, "*.mp4")), "_ik.mp4", use_ik)
        if mp4:
            result["mp4"] = _copy(mp4, os.path.join(out, f"{name}.mp4"))

        print(f"[momask] text-to-motion done: {result['bvh']}")
        return result


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def _slugify(text: str, max_len: int = 40) -> str:
    """Turn a text prompt into a filesystem-safe base name."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text.strip().lower()]
    slug = "".join(keep).strip("_") or "motion"
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_len].strip("_") or "motion"


def _copy(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    shutil.copy(src, dst)
    return dst


def _pick(paths: List[str], ik_suffix: str, prefer_ik: bool) -> Optional[str]:
    """Pick the IK-corrected variant when `prefer_ik`, else the raw one."""
    if not paths:
        return None
    ik = sorted(p for p in paths if p.endswith(ik_suffix))
    raw = sorted(p for p in paths if not p.endswith(ik_suffix))
    if prefer_ik and ik:
        return ik[0]
    if raw:
        return raw[0]
    return (ik or raw or [None])[0]
