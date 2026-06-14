#!/bin/bash
set -e

# 初始化环境（首次使用时取消注释）
# conda create -n trellis2 python=3.10
# conda activate trellis2

# torch 安装
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
conda install -y -c nvidia cuda-toolkit=12.4

# 下载 TRELLIS.2 到 models/gen_3d/TRELLIS2_main
git clone https://github.com/microsoft/TRELLIS.2.git models/gen_3d/TRELLIS2_main

# eigen 下载到 o-voxel/third_party/eigen
git clone https://gitlab.com/libeigen/eigen.git models/gen_3d/TRELLIS2_main/extensions/o-voxel/third_party/eigen

# flash-attn 安装
pip install psutil
pip install flash-attn==2.7.3 --no-build-isolation

# transformers 版本
pip install transformers==4.57.1

cd models/gen_3d/TRELLIS2_main/
. ./setup_cu124.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
