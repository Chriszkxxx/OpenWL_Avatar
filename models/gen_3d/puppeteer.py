"""
PuppeteerModel — wrapper for the Puppeteer auto-rigging + retarget pipeline.

Mirrors the style of `models.gen_3d.trellis.TrellisModel`: a thin class that
points at the Puppeteer source tree (cloned to `models/gen_3d/Puppeteer_main/`
by `scripts/installing/install_puppeteer.sh`, just like `TRELLIS2_main`) and
exposes high-level methods.

Puppeteer belongs under `gen_3d` because it is the *skeleton detection +
skinning* model referenced in `models/gen_3d/__init__.py` ("监测骨骼"): given a
3D mesh it predicts a skeleton and per-vertex skin weights, producing a rigged
character ready for animation.

Two capabilities are exposed:

  1. `rig(mesh_path, output_dir)` — GPU rigging. Runs the skeleton GPT
     (`skeleton/demo.py`) then the skinning network (`skinning/main.py`) as
     subprocesses (they use `accelerate` / `torchrun` and load checkpoints via
     paths relative to their own dirs), returning the Puppeteer rig `.txt`
     (joints + bones + skin weights).

  2. `retarget(...)` — CPU/bpy motion transfer onto the rigged character. This
     uses the vendored, committed engine in
     `models/gen_3d/puppeteer_retarget/` (world-conjugation-delta retarget),
     run in a Blender/bpy interpreter.

Why subprocess instead of in-process import (unlike trellis.py): the rigging
stages are multi-process GPU jobs with their own working dirs and relative
checkpoint paths, and the retarget stage needs `bpy` (often a different env
than the torch one). Subprocess keeps each stage isolated and matches how the
upstream `demo.sh` scripts are meant to run.
"""

import glob
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))

# Puppeteer source root, cloned next to this file by install_puppeteer.sh
# (models/gen_3d/Puppeteer_main/{skeleton,skinning,...}).
DEFAULT_PUPPETEER_ROOT = os.path.join(_HERE, "Puppeteer_main")

# Vendored bpy retarget engine (committed with this repo).
RETARGET_PKG = "models.gen_3d.puppeteer_retarget"
MAPPINGS_PKG = f"{RETARGET_PKG}.mappings"
MAPPINGS_DIR = os.path.join(_HERE, "puppeteer_retarget", "mappings")

DEFAULT_SKELETON_CKPT = "skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth"
DEFAULT_SKINNING_CKPT = "skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth"


class PuppeteerModel:
    """Thin wrapper around the Puppeteer rigging + retarget pipeline."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        skeleton_ckpt: str = DEFAULT_SKELETON_CKPT,
        skinning_ckpt: str = DEFAULT_SKINNING_CKPT,
        llm: str = "facebook/opt-350m",
        input_pc_num: int = 8192,
        skinning_depth: int = 1,
        gpu: int = 0,
        python_bin: Optional[str] = None,
        bpy_python: Optional[str] = None,
        repo_root: Optional[str] = None,
    ):
        """
        Args:
            model_path:     Puppeteer source root. Defaults to
                            `models/gen_3d/Puppeteer_main`.
            device:         "cuda" / "cpu" (rigging requires CUDA).
            skeleton_ckpt:  Skeleton-GPT weights, relative to `<root>/skeleton`.
            skinning_ckpt:  Skinning-net weights, relative to `<root>/skinning`.
            llm:            LLM backbone id for the skeleton model.
            input_pc_num:   Sampled point-cloud size for skeleton inference.
            skinning_depth: Skinning transformer depth (1 or 2).
            gpu:            CUDA_VISIBLE_DEVICES index for the rigging subprocesses.
            python_bin:     Python interpreter for the torch rigging stages.
                            Defaults to the current interpreter.
            bpy_python:     Python interpreter that has `bpy` importable for the
                            retarget stage. Defaults to `python_bin`.
            repo_root:      OpenWL-Avatar repo root (for `python -m` of the
                            vendored retarget package). Auto-detected by default.
        """
        self.model_path = os.path.abspath(model_path or DEFAULT_PUPPETEER_ROOT)
        self.device = device
        self.skeleton_ckpt = skeleton_ckpt
        self.skinning_ckpt = skinning_ckpt
        self.llm = llm
        self.input_pc_num = input_pc_num
        self.skinning_depth = skinning_depth
        self.gpu = gpu
        self.python_bin = python_bin or sys.executable
        self.bpy_python = bpy_python or self.python_bin
        # repo root = three levels up from this file: models/gen_3d/ -> repo.
        self.repo_root = os.path.abspath(
            repo_root or os.path.join(_HERE, os.pardir, os.pardir)
        )

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def _run(self, cmd: List[str], cwd: str, extra_env: Optional[Dict[str, str]] = None) -> None:
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", str(self.gpu))
        if extra_env:
            env.update(extra_env)
        print(f"[puppeteer] $ (cwd={cwd})\n  {' '.join(cmd)}")
        subprocess.run(cmd, cwd=cwd, env=env, check=True)

    @staticmethod
    def _to_obj(mesh_path: str, dst_obj: str) -> str:
        """Ensure an `.obj` exists for rigging (skeleton + skinning consume OBJ)."""
        if mesh_path.lower().endswith(".obj"):
            shutil.copy(mesh_path, dst_obj)
            return dst_obj
        import trimesh

        mesh = trimesh.load(mesh_path, force="mesh", process=False, maintain_order=True)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh.export(dst_obj)
        return dst_obj

    # ------------------------------------------------------------------
    # Rigging (GPU): skeleton + skinning -> rig.txt
    # ------------------------------------------------------------------

    def rig(
        self,
        mesh_path: str,
        output_dir: str,
        name: Optional[str] = None,
        post_filter: bool = True,
        master_port: int = 10009,
    ) -> Dict[str, str]:
        """
        Predict skeleton + skin weights for a mesh and return the rig `.txt`.

        Args:
            mesh_path:   Input mesh (`.glb` / `.obj` / `.ply` / `.stl`).
            output_dir:  Directory for intermediate + final rig artifacts.
            name:        Base name for outputs (defaults to the mesh stem).
            post_filter: Smooth skin weights by averaging across 1-ring neighbors.
            master_port: torchrun master port for the skinning stage.

        Returns:
            dict with keys: {"rig_txt", "skeleton_txt", "mesh_obj", "name"}.
            `rig_txt` is the final skeleton + skinning file consumed by retarget.
        """
        name = name or os.path.splitext(os.path.basename(mesh_path))[0]
        out = os.path.abspath(output_dir)
        mesh_dir = os.path.join(out, "mesh_folder")
        skel_dir = os.path.join(out, "skel_folder")
        skin_dir = os.path.join(out, "skinning")
        for d in (mesh_dir, skel_dir, skin_dir):
            os.makedirs(d, exist_ok=True)

        mesh_obj = self._to_obj(mesh_path, os.path.join(mesh_dir, f"{name}.obj"))

        # Stage 1: skeleton GPT -> "<name>_pred.txt"
        skel_root = os.path.join(self.model_path, "skeleton")
        save_name = f"openwl_{name}"
        self._run(
            [
                self.python_bin, "demo.py",
                "--input_path", mesh_obj,
                "--pretrained_weights", self.skeleton_ckpt,
                "--save_name", save_name,
                "--input_pc_num", str(self.input_pc_num),
                "--apply_marching_cubes", "--joint_token", "--seq_shuffle",
            ],
            cwd=skel_root,
        )
        skel_out_dir = os.path.join(skel_root, "outputs", save_name)
        pred_txts = glob.glob(os.path.join(skel_out_dir, f"{name}*_pred.txt"))
        if not pred_txts:
            pred_txts = glob.glob(os.path.join(skel_out_dir, "*_pred.txt"))
        if not pred_txts:
            raise RuntimeError(f"Skeleton stage produced no *_pred.txt in {skel_out_dir}")
        skeleton_txt = os.path.join(skel_dir, f"{name}.txt")
        shutil.copy(pred_txts[0], skeleton_txt)

        # Stage 2: skinning net (torchrun) -> "<save_folder>/generate/<name>_skin.txt"
        skin_root = os.path.join(self.model_path, "skinning")
        cmd = [
            self.python_bin, "-m", "torch.distributed.run",
            "--nproc_per_node=1", f"--master_port={master_port}", "main.py",
            "--num_workers", "1", "--batch_size", "1",
            "--generate", "--save_skin_npy",
            "--pretrained_weights", self.skinning_ckpt,
            "--input_skel_folder", skel_dir,
            "--mesh_folder", mesh_dir,
            "--depth", str(self.skinning_depth),
            "--save_folder", skin_dir,
        ]
        if post_filter:
            cmd.append("--post_filter")
        self._run(cmd, cwd=skin_root)

        skin_txts = glob.glob(os.path.join(skin_dir, "generate", f"{name}*_skin.txt"))
        if not skin_txts:
            skin_txts = glob.glob(os.path.join(skin_dir, "generate", "*_skin.txt"))
        if not skin_txts:
            raise RuntimeError(f"Skinning stage produced no *_skin.txt in {skin_dir}")
        rig_txt = skin_txts[0]

        print(f"[puppeteer] rig done: {rig_txt}")
        return {
            "rig_txt": rig_txt,
            "skeleton_txt": skeleton_txt,
            "mesh_obj": mesh_obj,
            "name": name,
        }

    # ------------------------------------------------------------------
    # Rigged FBX export (bpy)
    # ------------------------------------------------------------------

    def export_rigged_fbx(self, glb_path: str, rig_txt: str, output_fbx: str) -> str:
        """Export a bind-pose FBX (mesh + armature + skin) from GLB + rig.txt."""
        code = (
            "from models.gen_3d.puppeteer_retarget.world_delta import build_puppeteer_rig;"
            "import bpy;"
            f"build_puppeteer_rig(r'{glb_path}', r'{rig_txt}');"
            "bpy.ops.object.select_all(action='SELECT');"
            f"bpy.ops.export_scene.fbx(filepath=r'{output_fbx}', check_existing=False,"
            " add_leaf_bones=False, path_mode='COPY', embed_textures=True,"
            " object_types={'ARMATURE','MESH'})"
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_fbx)) or ".", exist_ok=True)
        self._run_bpy(["-c", code], expect_output=output_fbx)
        return output_fbx

    def _run_bpy(self, args: List[str], expect_output: Optional[str] = None) -> None:
        """Run a bpy subprocess.

        bpy is known to segfault on process *exit* in headless environments,
        after the FBX has already been written. We therefore don't fail on a
        nonzero return code as long as the expected output file was produced
        (and is newer than when we launched).
        """
        env = os.environ.copy()
        conda_prefix = env.get("CONDA_PREFIX")
        if conda_prefix:
            env["LD_LIBRARY_PATH"] = os.path.join(conda_prefix, "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["PYTHONPATH"] = self.repo_root + os.pathsep + env.get("PYTHONPATH", "")
        env.pop("PYOPENGL_PLATFORM", None)
        cmd = [self.bpy_python, *args]
        print(f"[puppeteer] $ (bpy, cwd={self.repo_root})\n  {' '.join(cmd)}")

        start = time.time()
        proc = subprocess.run(cmd, cwd=self.repo_root, env=env)
        if proc.returncode == 0:
            return
        # Tolerate the harmless exit-time segfault iff the output is present + fresh.
        if expect_output and os.path.exists(expect_output) and os.path.getmtime(expect_output) >= start - 1:
            print(
                f"[puppeteer] bpy exited with code {proc.returncode} but output "
                f"was written ({expect_output}); treating as success (known bpy "
                f"exit-time segfault)."
            )
            return
        raise subprocess.CalledProcessError(proc.returncode, cmd)

    # ------------------------------------------------------------------
    # Retarget (bpy): motion -> rigged character
    # ------------------------------------------------------------------

    def retarget(
        self,
        glb_path: str,
        rig_txt: str,
        motion_path: str,
        output_fbx: str,
        mapping: Optional[str] = None,
        source: str = "mixamo",
        action_name: str = "Take 001",
        fps: int = 30,
        anim_only: bool = False,
        global_scale: float = 1.0,
        root_scale: Optional[float] = None,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Retarget a motion clip onto the rigged character.

        The world-delta math transfers world-space rotations, so BVH motion
        (e.g. MoMask) can be retargeted *directly* — no Mixamo FBX detour.

        Args:
            glb_path:     Textured target character GLB.
            rig_txt:      Puppeteer rig `.txt` (from `rig()`).
            motion_path:  Source animation — a Mixamo FBX or a BVH.
            output_fbx:   Output animated FBX path.
            mapping:      Source->Puppeteer bone-map JSON. Defaults per `source`:
                          mixamo -> luffi_puppeteer_ue_mixamo_mapping.json,
                          bvh    -> momask_bvh_to_puppeteer_mapping.json.
            source:       "mixamo" or "bvh" (direct).
            action_name:  UE-friendly take name.
            fps:          Output FPS (use 20 for MoMask).
            anim_only:    Export armature animation only (UE Existing Skeleton).
            global_scale: BVH import scale (reconcile BVH units with the rig).
            root_scale:   Override root-translation scale (0 = in-place).
            extra_args:   Extra CLI flags forwarded to the retarget module.

        Returns:
            dict with keys {"output", "intermediate"} (intermediate is always None).
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_fbx)) or ".", exist_ok=True)

        if source == "bvh":
            # Direct BVH -> Puppeteer.
            mapping = mapping or os.path.join(MAPPINGS_DIR, "momask_bvh_to_puppeteer_mapping.json")
        elif source == "mixamo":
            mapping = mapping or os.path.join(MAPPINGS_DIR, "luffi_puppeteer_ue_mixamo_mapping.json")
        else:
            raise ValueError(f"Unsupported retarget source: {source!r}. Use 'mixamo' or 'bvh'.")

        cmd = [
            "-m", f"{RETARGET_PKG}.world_delta",
            "--glb", glb_path,
            "--rig", rig_txt,
            "--source-anim", motion_path,
            "--mapping", mapping,
            "--output", output_fbx,
            "--action-name", action_name,
            "--fps", str(fps),
            "--global-scale", str(global_scale),
        ]
        if root_scale is not None:
            cmd += ["--root-scale", str(root_scale)]
        if anim_only:
            cmd.append("--anim-only")
        if extra_args:
            cmd += extra_args
        self._run_bpy(cmd, expect_output=output_fbx)

        print(f"[puppeteer] retarget done: {output_fbx}")
        return {"output": output_fbx, "intermediate": None}

    # ------------------------------------------------------------------
    # Mapping generation (bpy): FBX pair -> source->Puppeteer bone-map JSON
    # ------------------------------------------------------------------

    def generate_mapping(
        self,
        puppeteer_fbx: str,
        mixamo_fbx: str,
        output_json: str,
    ) -> str:
        """
        Auto-generate a Mixamo->Puppeteer bone-map JSON from a bind-pose FBX pair.

        Infers the mapping purely from skeleton topology + world-pose geometry
        (no hardcoded joint indices), via the vendored
        `puppeteer_retarget/mappings/generate_mapping_auto.py`. The resulting
        JSON is directly consumable by `retarget(..., mapping=output_json)`.

        Args:
            puppeteer_fbx: Bind-pose FBX of the Puppeteer-rigged character
                           (e.g. the `*_rigged.fbx` from `export_rigged_fbx`).
            mixamo_fbx:    Any Mixamo-rigged FBX of the same/similar character
                           (T-pose or animated; first frame is the bind ref).
            output_json:   Path to write the mapping JSON.

        Returns:
            The `output_json` path.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
        self._run_bpy(
            [
                "-m", f"{MAPPINGS_PKG}.generate_mapping_auto",
                "--puppeteer-fbx", puppeteer_fbx,
                "--mixamo-fbx", mixamo_fbx,
                "--output", output_json,
            ],
            expect_output=output_json,
        )
        print(f"[puppeteer] mapping written: {output_json}")
        return output_json

    def inspect_skeletons(self, fbx_paths: List[str], output_txt: str) -> str:
        """Dump the hierarchy + world head/tail of FBX armatures to a text file.

        Debug helper for designing mapping rules; wraps
        `puppeteer_retarget/mappings/inspect_skeleton.py`.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_txt)) or ".", exist_ok=True)
        self._run_bpy(
            ["-m", f"{MAPPINGS_PKG}.inspect_skeleton", "--output", output_txt, *fbx_paths],
            expect_output=output_txt,
        )
        print(f"[puppeteer] skeleton dump written: {output_txt}")
        return output_txt

    # ------------------------------------------------------------------
    # High-level convenience: mesh -> rigged FBX
    # ------------------------------------------------------------------

    def mesh_to_rigged_fbx(
        self,
        mesh_path: str,
        output_dir: str,
        glb_path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, str]:
        """Rig a mesh and export a bind-pose FBX in one call."""
        result = self.rig(mesh_path, output_dir, name=name)
        glb = glb_path or (mesh_path if mesh_path.lower().endswith(".glb") else None)
        if glb:
            fbx = os.path.join(output_dir, f"{result['name']}_rigged.fbx")
            result["rigged_fbx"] = self.export_rigged_fbx(glb, result["rig_txt"], fbx)
        return result
