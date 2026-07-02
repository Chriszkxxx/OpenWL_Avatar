"""
puppeteer_retarget — Blender/bpy retarget engine for Puppeteer-rigged avatars.

This sub-package is the *vendored* (committed) half of the Puppeteer
integration. The heavy ML rigging models (skeleton GPT + skinning network)
live under `models/gen_3d/Puppeteer_main/` and are installed via
`scripts/installing/install_puppeteer.sh`; the retarget code here is pure
bpy geometry and ships with the repo so the motion pipeline is runnable on
its own.

Modules:
  - rig_io.py       : load Puppeteer rig `.txt`, build armature, skin weights,
                      import textured GLB, export FBX (from Puppeteer export.py).
  - world_delta.py  : world-conjugation-delta retarget (FBX/BVH -> Puppeteer).
  - mappings/       : source->Puppeteer bone-map JSONs *and* their bpy builders
                      (generate_mapping_auto.py auto-generates a bone-map JSON
                      from an FBX pair; inspect_skeleton.py dumps armature
                      hierarchies).

Run the scripts as modules so relative imports resolve, e.g.

    python -m models.gen_3d.puppeteer_retarget.world_delta --help
    python -m models.gen_3d.puppeteer_retarget.mappings.generate_mapping_auto --help
"""
