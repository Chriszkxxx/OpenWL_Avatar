"""
puppeteer_retarget.mappings — source->Puppeteer bone-map JSONs + their builders.

Bundled mapping JSONs (consumed at retarget time by `world_delta.py`):
  - luffi_puppeteer_ue_mixamo_mapping.json : Mixamo  -> Puppeteer
  - momask_bvh_to_puppeteer_mapping.json   : MoMask BVH -> Puppeteer

Design-time builders (require a `bpy`-capable interpreter; not on the runtime
retarget path):
  - generate_mapping_auto.py : auto-generate a Mixamo->Puppeteer bone-map JSON
                               from a bind-pose FBX pair, using skeleton
                               topology + world-pose geometry (no hardcoded
                               joint indices).
  - inspect_skeleton.py      : dump FBX armature hierarchies (head/tail) to a
                               text file, for designing / debugging mappings.

Run the builders as package modules so imports resolve, e.g.

    python -m models.gen_3d.puppeteer_retarget.mappings.generate_mapping_auto --help

or via `PuppeteerModel.generate_mapping(...)`.
"""
