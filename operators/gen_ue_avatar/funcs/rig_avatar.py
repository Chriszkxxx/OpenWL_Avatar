"""
rig_avatar.py — Auto-rig a 3D avatar mesh with Puppeteer (skeleton + skinning).

Pipeline:
  1. Run the Puppeteer skeleton GPT to predict a skeleton for the mesh.
  2. Run the Puppeteer skinning network to predict per-vertex skin weights.
  3. (Optional) Export a bind-pose FBX (mesh + armature + weights) for UE / DCC.

Input:
    mesh_path  : str            — 3D avatar mesh (`.glb` recommended; `.obj` ok)
    model      : PuppeteerModel — loaded Puppeteer rigging model

Output:
    dict with keys {"rig_txt", "skeleton_txt", "mesh_obj", "name", "rigged_fbx"?}
    `rig_txt` is the skeleton + skin-weight file consumed by retarget_motion.
"""

import os
from typing import Optional


DEFAULT_OUTPUT_DIR = "output/rigging"


def rig_avatar(
    mesh_path: str,
    model,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    export_fbx: bool = True,
    post_filter: bool = True,
) -> dict:
    """
    Auto-rig an avatar mesh into a Puppeteer skeleton + skin-weighted rig.

    Args:
        mesh_path:   Path to the input mesh (`.glb` / `.obj` / `.ply` / `.stl`).
        model:       Loaded `PuppeteerModel` instance.
        output_dir:  Directory for rig artifacts. Defaults to `output/rigging`.
        name:        Base name for outputs (defaults to the mesh stem).
        export_fbx:  If True and the input is a `.glb`, also export a bind-pose
                     FBX (mesh + armature + skin weights).
        post_filter: Smooth skin weights across 1-ring neighbors.

    Returns:
        dict: {"rig_txt", "skeleton_txt", "mesh_obj", "name", optionally
               "rigged_fbx"}.
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    result = model.rig(mesh_path, output_dir, name=name, post_filter=post_filter)
    print(f"[rig_avatar] rig file: {result['rig_txt']}")

    if export_fbx and mesh_path.lower().endswith(".glb"):
        fbx_path = os.path.join(output_dir, f"{result['name']}_rigged.fbx")
        result["rigged_fbx"] = model.export_rigged_fbx(
            mesh_path, result["rig_txt"], fbx_path
        )
        print(f"[rig_avatar] bind-pose FBX: {result['rigged_fbx']}")

    return result
