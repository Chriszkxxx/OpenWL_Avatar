#!/bin/bash
set -e

# MoMask text-to-motion generation setup.
#
# Layout (mirrors install_puppeteer.sh / Puppeteer_main):
#   models/gen_3d/momask_main/             <- cloned MoMask source (gen_t2m.py)
#   models/gen_3d/momask.py                <- MoMaskModel wrapper (committed)
#
# 初始化环境（首次使用时取消注释）
# conda create -n momask python=3.10
# conda activate momask

# 下载 MoMask 到 models/gen_3d/momask_main
git clone https://github.com/EricGuo5513/momask-codes.git models/gen_3d/momask_main

# torch (match your CUDA; example: cu121)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# text-to-motion deps (masked/residual transformers + RVQ + CLIP text encoder)
pip install --no-build-isolation chumpy
pip install -r models/gen_3d/momask_main/requirements.txt
pip install imageio imageio-ffmpeg
pip install git+https://github.com/openai/CLIP.git

# 下载 checkpoints（RVQ + masked/residual transformer + length estimator）
# gen_t2m.py 从 momask_main/checkpoints/<dataset>/<name> 读取权重。
( cd models/gen_3d/momask_main && bash prepare/download_models.sh )

cat <<'EOF'

Done. MoMask generates motion in the HumanML3D 22-joint skeleton at 20 fps and
writes .bvh clips, which the Puppeteer world-delta retarget engine consumes
directly (source="bvh", fps=20).

Before running, make sure the repo root is importable:

  export PYTHONPATH="$(pwd):$PYTHONPATH"

Quick test (text -> BVH):
  python - <<'PY'
  from models.gen_3d.momask import MoMaskModel
  m = MoMaskModel(gpu=0)
  print(m.generate("a person walks forward and waves",
                   output_dir="output/motion")["bvh"])
  PY

Then retarget onto a rigged avatar (see test/test_retarget.py, source="bvh").
EOF
