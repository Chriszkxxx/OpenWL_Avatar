pip install bpy

conda install -y -c conda-forge \
  xorg-libxi xorg-libxrender xorg-libxext xorg-libxrandr \
  xorg-libxxf86vm xorg-libxfixes libxkbcommon \
  xorg-libx11 xorg-libsm xorg-libice mesalib libgl

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
