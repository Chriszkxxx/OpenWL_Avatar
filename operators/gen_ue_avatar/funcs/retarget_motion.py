"""
retarget_motion.py — Retarget a motion clip onto a Puppeteer-rigged avatar.

Transfers an existing animation onto the Puppeteer skeleton using the
world-conjugation-delta method (`models/gen_3d/puppeteer_retarget/`). The delta
is applied in world space, so it is independent of the auto-rig's local bone
roll — this fixes the arm-drift / backward-knee artifacts of local-frame
methods.

Source types:
  - "mixamo":         a Mixamo FBX animation, retargeted directly.
  - "bvh":            a BVH (e.g. MoMask), retargeted *directly* — the
                      world-delta math is independent of source bone roll, so
                      no Mixamo intermediate is needed.
  - "bvh_via_mixamo": legacy two-step (BVH -> Mixamo FBX -> Puppeteer), kept as
                      a fallback when a Mixamo bind rig is available.

Input:
    glb_path    : str            — textured target character GLB
    rig_txt     : str            — Puppeteer rig `.txt` (from rig_avatar)
    motion_path : str            — Mixamo FBX or MoMask BVH
    model       : PuppeteerModel — loaded Puppeteer model (provides bpy runner)

Output:
    dict with keys {"output", "intermediate"} (+ "anim_only" if requested).
"""

import os
from typing import List, Optional


DEFAULT_OUTPUT_DIR = "output/motion"


def retarget_motion(
    glb_path: str,
    rig_txt: str,
    motion_path: str,
    model,
    output_path: Optional[str] = None,
    source: str = "mixamo",
    mapping: Optional[str] = None,
    mixamo_ref: Optional[str] = None,
    bvh_mapping: Optional[str] = None,
    momask_keemap_json: Optional[str] = None,
    action_name: str = "Take 001",
    fps: int = 30,
    global_scale: float = 1.0,
    root_scale: Optional[float] = None,
    export_anim_only: bool = True,
    extra_args: Optional[List[str]] = None,
) -> dict:
    """
    Retarget a motion clip onto a Puppeteer-rigged avatar.

    Args:
        glb_path:        Textured target character GLB.
        rig_txt:         Puppeteer rig `.txt` (skeleton + skin weights).
        motion_path:     Source animation — Mixamo FBX or BVH (e.g. MoMask).
        model:           Loaded `PuppeteerModel` instance.
        output_path:     Output animated FBX (mesh + anim). Defaults to
                         `output/motion/<motion>_on_<char>.fbx`.
        source:          "mixamo", "bvh" (direct, recommended), or
                         "bvh_via_mixamo" (legacy two-step via Mixamo).
        mapping:         Source->Puppeteer bone-map JSON (bundled default per source).
        mixamo_ref:      Only for `source="bvh_via_mixamo"`: a Mixamo bind-rig FBX.
        bvh_mapping:     BVH->Mixamo bone-map JSON (legacy two-step).
        momask_keemap_json: Optional MoMask assets/mapping.json corrections (legacy).
        action_name:     UE-friendly take name.
        fps:             Output FPS (use 20 for MoMask BVH).
        global_scale:    BVH import scale (reconcile BVH units with the rig).
        root_scale:      Override root-translation scale (0 = in-place).
        export_anim_only: Also export an armature-only FBX for UE "Existing
                         Skeleton" import (recommended when the skeletal mesh
                         already exists in UE).
        extra_args:      Extra CLI flags forwarded to the retarget engine.

    Returns:
        dict: {"output", "intermediate", optionally "anim_only"}.
    """
    output_dir = DEFAULT_OUTPUT_DIR
    if output_path is None:
        char = os.path.splitext(os.path.basename(glb_path))[0]
        motion = os.path.splitext(os.path.basename(motion_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{motion}_on_{char}.fbx")

    result = model.retarget(
        glb_path=glb_path,
        rig_txt=rig_txt,
        motion_path=motion_path,
        output_fbx=output_path,
        mapping=mapping,
        source=source,
        mixamo_ref=mixamo_ref,
        bvh_mapping=bvh_mapping,
        momask_keemap_json=momask_keemap_json,
        action_name=action_name,
        fps=fps,
        global_scale=global_scale,
        root_scale=root_scale,
        anim_only=False,
        extra_args=extra_args,
    )
    print(f"[retarget_motion] mesh+anim FBX: {result['output']}")

    if export_anim_only:
        anim_only_path = os.path.splitext(output_path)[0] + "_anim_only.fbx"
        # Reuse the intermediate FBX from the legacy two-step path to avoid recompute.
        anim = model.retarget(
            glb_path=glb_path,
            rig_txt=rig_txt,
            motion_path=(result["intermediate"] or motion_path),
            output_fbx=anim_only_path,
            mapping=mapping,
            source=("mixamo" if result["intermediate"] else source),
            mixamo_ref=mixamo_ref,
            bvh_mapping=bvh_mapping,
            momask_keemap_json=momask_keemap_json,
            action_name=action_name,
            fps=fps,
            global_scale=global_scale,
            root_scale=root_scale,
            anim_only=True,
            extra_args=extra_args,
        )
        result["anim_only"] = anim["output"]
        print(f"[retarget_motion] anim-only FBX: {result['anim_only']}  "
              f"(import in UE with Existing Skeleton)")

    return result
