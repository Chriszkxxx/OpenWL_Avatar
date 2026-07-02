"""
gen_motion.py — Skeleton detection and motion generation for the avatar.

`detect_skeleton` is implemented via Puppeteer auto-rigging (skeleton + skin
weights); see `rig_avatar.py`. Applying an *existing* motion clip onto the
rigged character (Mixamo FBX / MoMask BVH) is handled by `retarget_motion.py`.
`gen_motion` generates a *novel* motion from text via MoMask (text-to-motion),
producing a HumanML3D BVH clip, and — when a rigged avatar is provided —
retargets it directly onto that avatar via the Puppeteer world-delta engine.

Input:
    motion_desc  : str           — text description of the desired motion / action
    momask_model : MoMaskModel   — loaded text-to-motion model
    (optional retarget) glb_path + rig_txt + puppeteer_model

Output:
    dict with the generated motion paths ({"bvh", ...}), augmented with the
    retarget outputs ({"output", "anim_only", ...}) when a rig is supplied.
"""

import os
from typing import Optional


DEFAULT_OUTPUT_DIR = "output/motion"


def detect_skeleton(
    mesh_path: str,
    model,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    export_fbx: bool = True,
) -> dict:
    """
    Detect and bind a skeleton (+ skin weights) to the 3D avatar mesh.

    Thin wrapper around `rig_avatar.rig_avatar` using the Puppeteer model.

    Args:
        mesh_path:  Path to the 3D avatar mesh file (`.glb` recommended).
        model:      Loaded `PuppeteerModel` skeleton/skinning model.
        output_dir: Directory for rig artifacts.
        name:       Base name for outputs.
        export_fbx: Also export a bind-pose FBX when the input is a `.glb`.

    Returns:
        dict with keys {"rig_txt", "skeleton_txt", "mesh_obj", "name",
        optionally "rigged_fbx"}.
    """
    from operators.gen_ue_avatar.funcs.rig_avatar import rig_avatar

    return rig_avatar(
        mesh_path, model, output_dir=output_dir, name=name, export_fbx=export_fbx
    )


def gen_motion(
    motion_desc: str,
    momask_model,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    motion_length: int = 0,
    use_ik: bool = True,
    seed: int = 10107,
    glb_path: Optional[str] = None,
    rig_txt: Optional[str] = None,
    puppeteer_model=None,
    retarget_output: Optional[str] = None,
    export_anim_only: bool = True,
    retarget_kwargs: Optional[dict] = None,
    **gen_kwargs,
) -> dict:
    """
    Generate a *novel* avatar motion from a text description via MoMask.

    Runs MoMask text-to-motion to produce a HumanML3D BVH clip. If a rigged
    avatar is supplied (`glb_path` + `rig_txt` + `puppeteer_model`), the clip is
    retargeted directly onto that avatar with the Puppeteer world-delta engine
    (`source="bvh"`, 20 fps to match MoMask), yielding an animated FBX.

    To apply an *existing* motion clip (Mixamo FBX / MoMask BVH) onto a rig
    without generating a new one, use `retarget_motion.retarget_motion`.

    Args:
        motion_desc:      Text description of the desired motion.
        momask_model:     Loaded `MoMaskModel` (text-to-motion).
        output_dir:       Directory for motion artifacts. Defaults to
                          `output/motion`.
        name:             Base name for outputs (defaults to a prompt slug).
        motion_length:    Number of poses (0 = MoMask estimates the length).
        use_ik:           Prefer the foot-IK-corrected BVH when available.
        seed:             RNG seed for reproducible generation.
        glb_path:         Textured target GLB (enables retarget when set).
        rig_txt:          Puppeteer rig `.txt` (enables retarget when set).
        puppeteer_model:  Loaded `PuppeteerModel` (enables retarget when set).
        retarget_output:  Output animated FBX path (default derived from names).
        export_anim_only: Also export an armature-only FBX for UE "Existing
                          Skeleton" import.
        retarget_kwargs:  Extra kwargs forwarded to `retarget_motion`.
        **gen_kwargs:     Extra kwargs forwarded to `MoMaskModel.generate`
                          (repeat_times, cond_scale, time_steps, temperature).

    Returns:
        The MoMask result dict ({"bvh", "bvh_raw", "bvh_ik"?, "npy"?, "mp4"?,
        "fps", ...}), augmented with retarget outputs ({"output", "anim_only"?})
        when a rig is provided.
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    result = momask_model.generate(
        motion_desc,
        output_dir=output_dir,
        name=name,
        motion_length=motion_length,
        use_ik=use_ik,
        seed=seed,
        **gen_kwargs,
    )
    print(f"[gen_motion] MoMask BVH: {result['bvh']}")

    if puppeteer_model is not None and glb_path and rig_txt:
        from operators.gen_ue_avatar.funcs.retarget_motion import retarget_motion

        retarget = retarget_motion(
            glb_path=glb_path,
            rig_txt=rig_txt,
            motion_path=result["bvh"],
            model=puppeteer_model,
            output_path=retarget_output,
            source="bvh",
            fps=result.get("fps", 20),
            export_anim_only=export_anim_only,
            **(retarget_kwargs or {}),
        )
        result.update(retarget)

    return result
