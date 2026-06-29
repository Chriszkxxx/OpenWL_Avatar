"""
gen_motion.py — Skeleton detection and motion generation for the avatar.

`detect_skeleton` is implemented via Puppeteer auto-rigging (skeleton + skin
weights); see `rig_avatar.py`. Applying an *existing* motion clip onto the
rigged character (Mixamo FBX / MoMask BVH) is handled by `retarget_motion.py`.
`gen_motion` (text -> novel motion via a generative motion model, e.g. MoMask)
is left as a hook for the future generative-motion model.

Input:
    mesh_path   : str           — path to the 3D avatar mesh
    motion_desc : str           — text description of the desired motion / action
    model       : ...           — loaded motion generation model (TBD)

Output:
    motion_path : str           — path to the exported motion file (.bvh / .fbx)
"""

from typing import Optional


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


def gen_motion(rigged_mesh_path: str, motion_desc: str, model) -> str:
    """
    Generate *novel* avatar motion from a text description.

    Reserved for a generative motion model (e.g. MoMask text-to-motion). To
    apply an existing motion clip onto a rigged avatar, use
    `retarget_motion.retarget_motion` instead.

    Args:
        rigged_mesh_path: Path to the rigged 3D avatar mesh / rig file.
        motion_desc:      Text description of the desired motion.
        model:            Loaded motion generation model.

    Returns:
        Path to the exported motion file (.bvh / .fbx).
    """
    raise NotImplementedError(
        "Text-to-motion generation is not wired yet; use retarget_motion() to "
        "apply an existing Mixamo FBX or MoMask BVH clip onto the rigged avatar."
    )
