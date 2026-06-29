#!/bin/bash
set -e

# Puppeteer auto-rigging (skeleton + skinning) + bpy retarget setup.
#
# Layout (mirrors install_trellis.sh / TRELLIS2_main):
#   models/gen_3d/Puppeteer_main/          <- cloned Puppeteer source (rigging)
#   models/gen_3d/puppeteer.py             <- PuppeteerModel wrapper (committed)
#   models/gen_3d/puppeteer_retarget/      <- bpy retarget engine (committed)
#
# 初始化环境（首次使用时取消注释）
# conda create -n puppeteer python=3.10
# conda activate puppeteer

# torch (match your CUDA; example: cu124)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 下载 Puppeteer 到 models/gen_3d/Puppeteer_main
git clone https://github.com/bytedance/Puppeteer.git models/gen_3d/Puppeteer_main

# rigging deps (skeleton GPT + skinning net)
pip install -r models/gen_3d/Puppeteer_main/requirements.txt
pip install accelerate transformers trimesh

# 下载 checkpoints（skeleton + skinning）
# 见各子模块的 download.py / README：
#   models/gen_3d/Puppeteer_main/skeleton/skeleton_ckpts/puppeteer_skeleton_w_diverse_pose.pth
#   models/gen_3d/Puppeteer_main/skinning/skinning_ckpts/puppeteer_skin_w_diverse_pose_depth1.pth
python models/gen_3d/Puppeteer_main/skeleton/download.py  || true
python models/gen_3d/Puppeteer_main/skinning/download.py  || true

# ---------------------------------------------------------------------------
# bpy retarget runtime (headless Blender as a Python module).
# In a headless / server box, bpy needs X11 + OpenGL shared libs at import time
# (libXrender, libXi, libGL, ...). Install them into the conda env and make sure
# LD_LIBRARY_PATH points at $CONDA_PREFIX/lib.
# ---------------------------------------------------------------------------
pip install bpy==4.2.0
if command -v conda &>/dev/null; then
  conda install -y -c conda-forge \
    xorg-libxrender xorg-libxi xorg-libxfixes xorg-libxxf86vm \
    xorg-libxkbcommon mesalib libgl
fi

cat <<'EOF'

Done. Before running rigging / retarget, ensure bpy can find its libs:

  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
  export PYTHONPATH="$(pwd):$PYTHONPATH"

Quick test:
  python test/test_rigging.py     # GLB -> Puppeteer rig (skeleton + skinning)
  python test/test_retarget.py    # rig + Mixamo FBX -> animated FBX
EOF
