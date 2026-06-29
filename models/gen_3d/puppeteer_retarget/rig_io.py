"""
rig_io.py — Puppeteer rig (.txt) <-> Blender armature helpers.

Loads a Puppeteer rig file (``joints`` / ``root`` / ``hier`` / ``skin`` lines),
builds the corresponding Blender mesh + armature with skin weights, and
imports the matching textured GLB. Shared by the retarget scripts in this
package.

Adapted from the Puppeteer repo ``export.py`` + ``export_glb.py``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import bpy  # type: ignore
import numpy as np
import trimesh
from mathutils import Vector  # type: ignore

# Y-up (GLB / trimesh) -> Z-up (Blender) coordinate rotation. Applied to both
# mesh vertices and rig joints so geometry and skeleton stay consistent.
ROT = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


# ---------------------------------------------------------------------------
# Scene management
# ---------------------------------------------------------------------------

def clear_bpy_data() -> None:
    """Wipe the current bpy datablocks so each run starts from a clean scene."""
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)


# ---------------------------------------------------------------------------
# Rig file parsing
# ---------------------------------------------------------------------------

def load_rigged_mesh_data(
    mesh_path: str,
    rig_path: str,
    apply_coord_rot: bool = True,
) -> Dict[str, Any]:
    """Parse a Puppeteer rig `.txt` + its mesh into arrays for armature build.

    Returns a dict with vertices, faces, bones (head+tail), parents, names,
    per-vertex skin weights and a DFS bone order.
    """
    mesh = trimesh.load(mesh_path, force='mesh', process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    raw_verts = mesh.vertices
    vertices = (ROT @ raw_verts.T).T if apply_coord_rot else raw_verts
    faces = mesh.faces
    N = vertices.shape[0]

    def _map_coord(x: float, y: float, z: float) -> np.ndarray:
        p = np.array([x, y, z])
        return ROT @ p if apply_coord_rot else p

    joint_pos: Dict[str, np.ndarray] = {}
    joint_hier: Dict[str, List[str]] = {}
    joint_skin: List[List[str]] = []
    id_mapping: Dict[str, int] = {}
    name_mapping: Dict[int, str] = {}
    parent_mapping: Dict[str, str] = {}
    root_name: Optional[str] = None
    tot = 0
    with open(rig_path, 'r') as f_info:
        for line in f_info:
            word = line.split()
            if not word:
                continue
            if word[0] == 'joints':
                joint_pos[word[1]] = _map_coord(float(word[2]), float(word[3]), float(word[4]))
                id_mapping[word[1]] = tot
                name_mapping[tot] = word[1]
                tot += 1
            elif word[0] == 'root':
                root_name = word[1]
            elif word[0] == 'hier':
                joint_hier.setdefault(word[1], []).append(word[2])
            elif word[0] == 'skin':
                joint_skin.append(word[1:])

    J = len(joint_pos)
    bones = np.zeros((J, 6))
    parents: List[Optional[int]] = []
    names: List[str] = []

    for name in joint_hier:
        for son in joint_hier[name]:
            parent_mapping[son] = name

    son = defaultdict(list)
    for i in range(J):
        name = name_mapping[i]
        names.append(name)
        parents.append(None if name == root_name else id_mapping[parent_mapping[name]])
        if name != root_name:
            son[id_mapping[parent_mapping[name]]].append(i)

    for i in range(J):
        name = name_mapping[i]
        head = joint_pos[name]
        tail = head + np.array([0., 0., 0.1])
        if len(joint_hier.get(name, [])) == 1:
            tail = joint_pos[joint_hier[name][0]]
        elif name != root_name:
            pname = name_mapping[parents[i]]
            direction = joint_pos[name] - joint_pos[pname]
            tail = head + direction * 0.5
        bones[i, :3] = head
        bones[i, 3:] = tail

    skin = np.zeros((N, J))
    for skin_item in joint_skin:
        u = int(skin_item[0])
        for j in range(1, len(skin_item), 2):
            jid = id_mapping[skin_item[j]]
            skin[u, jid] = float(skin_item[j + 1])

    dfs_order: List[int] = []
    Q = [id_mapping[root_name]]
    while Q:
        u = Q.pop()
        dfs_order.append(u)
        Q.extend(son[u])

    return {
        "vertices": vertices,
        "faces": faces,
        "bones": bones,
        "parents": parents,
        "names": names,
        "vertex_group": skin,
        "dfs_order": dfs_order,
    }


# ---------------------------------------------------------------------------
# Armature / weight construction
# ---------------------------------------------------------------------------

def assign_vertex_weights(
    ob: bpy.types.Object,
    names: List[str],
    vertex_group: np.ndarray,
    group_per_vertex: int = 4,
) -> None:
    """Assign the top-`group_per_vertex` skin weights per vertex (normalized)."""
    vis = [x.name for x in ob.vertex_groups]
    argsorted = np.argsort(-vertex_group, axis=1)
    vgw = vertex_group[np.arange(vertex_group.shape[0])[..., None], argsorted]
    vgw = vgw / vgw[..., :group_per_vertex].sum(axis=1)[..., None]

    for v, _w in enumerate(vertex_group):
        for ii in range(group_per_vertex):
            i = argsorted[v, ii]
            if i >= len(names):
                continue
            n = names[i]
            if n not in vis:
                continue
            ob.vertex_groups[n].add([v], float(vgw[v, ii]), 'REPLACE')


def build_rigged_mesh_objects(
    data: Dict[str, Any],
    mesh_name: str = "character",
    collection_name: str = "CA_collection",
) -> Tuple[bpy.types.Object, bpy.types.Object]:
    """Build a Blender mesh + armature from parsed rig data and bind them."""
    vertices = data["vertices"]
    faces = data["faces"]
    bones = data["bones"]
    parents = data["parents"]
    names = data["names"]
    vertex_group = data["vertex_group"]
    dfs_order = data["dfs_order"]

    mesh = bpy.data.meshes.new('mesh')
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    ob = bpy.data.objects.new(mesh_name, mesh)

    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    collection.objects.link(ob)

    bpy.ops.object.armature_add(enter_editmode=True)
    armature = bpy.data.armatures.get('Armature')
    edit_bones = armature.edit_bones
    bone_root = edit_bones.get('Bone')

    J = len(names)

    def extrude_bone(
        edit_bones_,
        name: str,
        parent_name: Optional[str],
        head: Tuple[float, float, float],
        tail: Tuple[float, float, float],
        is_root: bool = False,
    ):
        bone = bone_root if is_root else edit_bones_.new(name)
        bone.head = Vector((head[0], head[1], head[2]))
        bone.tail = Vector((tail[0], tail[1], tail[2]))
        bone.name = name
        bone.parent = edit_bones_.get(parent_name) if parent_name is not None else None
        bone.use_connect = True

    for k in range(J):
        i = dfs_order[k]
        edit_bones = armature.edit_bones
        if parents[i] is None:
            extrude_bone(edit_bones, names[i], None, bones[i, :3], bones[i, 3:], is_root=True)
        else:
            pname = names[parents[i]]
            extrude_bone(edit_bones, names[i], pname, bones[i, :3], bones[i, 3:])

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    arm = bpy.data.objects['Armature']
    arm.select_set(True)
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    assign_vertex_weights(ob, names, vertex_group)
    return ob, arm


# ---------------------------------------------------------------------------
# GLB import / parenting
# ---------------------------------------------------------------------------

def import_glb(glb_path: str) -> bpy.types.Object:
    """Import GLB as-is (preserve original orientation and materials)."""
    bpy.ops.import_scene.gltf(filepath=glb_path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh found in GLB: {glb_path}")

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
        print(f"  Joined {len(meshes)} GLB mesh objects.")

    obj = bpy.context.view_layer.objects.active
    obj.name = "TexturedMesh"
    return obj


def parent_to_armature(mesh_obj: bpy.types.Object, arm_obj: bpy.types.Object) -> None:
    """Parent ``mesh_obj`` to ``arm_obj`` using existing vertex-group names."""
    bpy.ops.object.select_all(action="DESELECT")
    mesh_obj.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_NAME")


# ---------------------------------------------------------------------------
# Standalone rig -> FBX export (no animation)
# ---------------------------------------------------------------------------

def export_fbx(
    path: str,
    vertices: np.ndarray,
    faces: Union[np.ndarray, None],
    bones: np.ndarray,
    parents: List[Union[int, None]],
    names: List[str],
    vertex_group: np.ndarray,
    dfs_order: List[int],
    group_per_vertex: int = 4,
) -> str:
    """Build the rigged proxy mesh and export a bind-pose FBX."""
    clear_bpy_data()
    data = {
        "vertices": vertices,
        "faces": faces,
        "bones": bones,
        "parents": parents,
        "names": names,
        "vertex_group": vertex_group,
        "dfs_order": dfs_order,
    }
    build_rigged_mesh_objects(data)
    bpy.ops.export_scene.fbx(filepath=path, check_existing=False, add_leaf_bones=False)
    return path
