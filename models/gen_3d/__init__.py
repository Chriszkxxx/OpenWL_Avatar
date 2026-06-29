"""
This sub-package contains 3D content generation models.
- 添加 Trellis 生成3D 人物 (trellis.py)
- 添加 Puppeteer 监测骨骼 + 蒙皮 (auto-rigging) 及 FBX retarget
  - puppeteer.py            : PuppeteerModel (skeleton GPT + skinning net + retarget)
  - puppeteer_retarget/     : bpy 世界增量 (world-delta) 动作重定向引擎
- 添加 FlashWorld 或者 Hunyuan-WorldPlay2 生成3D场景
- 搜索什么模型来 1. 监测骨骼 (Puppeteer 已接入) ; 2. 生成motion ; 3. 生成UE特效
"""
