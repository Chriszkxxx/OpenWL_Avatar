#!/bin/bash
# Download & install Blender 4.2.0 for headless Linux cluster use.
#
# Two components:
#   1. Blender binary  — official portable tarball (CLI / background mode)
#   2. bpy + X11/OpenGL — Python module in a conda env (export.py, bake scripts)
#
# Works with any conda env via ENV_PREFIX (default: puppeteer env).
#
# Usage:
#   bash scripts/install_blender.sh              # binary + bpy (default)
#   bash scripts/install_blender.sh --bpy-only   # conda bpy only (no tarball)
#   bash scripts/install_blender.sh --binary-only
#   bash scripts/install_blender.sh --verify     # check existing install
#
# Environment overrides:
#   BLENDER_VERSION=4.2.0
#   BLENDER_INSTALL_ROOT=/workspace/zhukaixin/tools
#   ENV_PREFIX=/workspace/zhukaixin/anaconda_envs/puppeteer
#   CONDA=/home/zhukaixin/miniconda3/bin/conda
#   http_proxy / https_proxy  (auto-detected if local proxy is up)
#
set -euo pipefail

BLENDER_VERSION="${BLENDER_VERSION:-4.2.0}"
BLENDER_MAJOR="${BLENDER_MAJOR:-${BLENDER_VERSION%.*}}"
BLENDER_INSTALL_ROOT="${BLENDER_INSTALL_ROOT:-/workspace/zhukaixin/tools}"
BLENDER_CACHE_DIR="${BLENDER_CACHE_DIR:-${BLENDER_INSTALL_ROOT}/cache}"
ENV_PREFIX="${ENV_PREFIX:-/workspace/zhukaixin/anaconda_envs/puppeteer}"
CONDA="${CONDA:-/home/zhukaixin/miniconda3/bin/conda}"

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) BLENDER_ARCH_SUFFIX="linux-x64" ;;
  aarch64) BLENDER_ARCH_SUFFIX="linux-arm64" ;;
  *)
    echo "Unsupported architecture: $ARCH (need x86_64 or aarch64)"
    exit 1
    ;;
esac

BLENDER_TARBALL="blender-${BLENDER_VERSION}-${BLENDER_ARCH_SUFFIX}.tar.xz"
BLENDER_EXTRACT_DIR="${BLENDER_INSTALL_ROOT}/blender-${BLENDER_VERSION}-${BLENDER_ARCH_SUFFIX}"
BLENDER_BIN="${BLENDER_EXTRACT_DIR}/blender"
BLENDER_URL="${BLENDER_URL:-https://download.blender.org/release/Blender${BLENDER_MAJOR}/${BLENDER_TARBALL}}"

MODE="${MODE:-all}"  # all | binary | bpy | verify

usage() {
  cat <<EOF
Usage: bash $(basename "$0") [OPTIONS]

Options:
  --all            Install Blender binary + bpy in conda (default)
  --binary-only    Download/extract official Blender tarball only
  --bpy-only       Install bpy==${BLENDER_VERSION} + X11/OpenGL libs in conda only
  --verify         Verify binary and bpy import; exit non-zero if missing
  -h, --help       Show this help

Paths:
  Binary:  ${BLENDER_BIN}
  Conda:   ${ENV_PREFIX}
EOF
}

setup_proxy() {
  if curl -s --max-time 2 -x http://127.0.0.1:7890 https://pypi.org >/dev/null 2>&1; then
    export http_proxy="${http_proxy:-http://127.0.0.1:7890}"
    export https_proxy="${https_proxy:-http://127.0.0.1:7890}"
    echo "  Using proxy: $http_proxy"
  else
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  fi
}

download_blender_binary() {
  mkdir -p "$BLENDER_INSTALL_ROOT" "$BLENDER_CACHE_DIR"

  if [[ -x "$BLENDER_BIN" ]]; then
    echo "==> Blender binary already installed: $BLENDER_BIN"
    "$BLENDER_BIN" --version | head -1
    return 0
  fi

  local archive="${BLENDER_CACHE_DIR}/${BLENDER_TARBALL}"
  echo "==> Downloading Blender ${BLENDER_VERSION} (${BLENDER_ARCH_SUFFIX})..."
  echo "    URL: $BLENDER_URL"

  if [[ -f "$archive" ]]; then
    echo "    Cache hit: $archive"
  else
    curl -L --retry 3 --retry-delay 5 --connect-timeout 30 --max-time 3600 \
      -o "$archive.part" "$BLENDER_URL"
    mv "$archive.part" "$archive"
    echo "    Saved: $archive ($(du -h "$archive" | cut -f1))"
  fi

  echo "==> Extracting to ${BLENDER_INSTALL_ROOT}..."
  tar -xJf "$archive" -C "$BLENDER_INSTALL_ROOT"
  chmod +x "$BLENDER_BIN"

  if [[ ! -x "$BLENDER_BIN" ]]; then
    echo "ERROR: blender executable not found after extract: $BLENDER_BIN"
    exit 1
  fi

  ln -sfn "$BLENDER_BIN" "${BLENDER_INSTALL_ROOT}/blender"

  echo "==> Blender binary OK: $("$BLENDER_BIN" --version | head -1)"
}

write_blender_env_hook() {
  local hook_dir="${BLENDER_INSTALL_ROOT}/env"
  mkdir -p "$hook_dir"
  cat > "${hook_dir}/blender.sh" <<EOF
# Source after conda activate (optional, for CLI blender):
#   source ${hook_dir}/blender.sh
export BLENDER_ROOT="${BLENDER_EXTRACT_DIR}"
export BLENDER_BIN="${BLENDER_BIN}"
export PATH="\$(dirname "\${BLENDER_BIN}"):\${PATH}"
EOF
  echo "==> Env hook: ${hook_dir}/blender.sh"
}

write_bpy_activate_hook() {
  local activate_dir="${ENV_PREFIX}/etc/conda/activate.d"
  mkdir -p "$activate_dir"
  cat > "${activate_dir}/bpy.sh" <<'HOOKEOF'
# bpy / headless OpenGL (install_blender.sh)
export _BLENDER_CONDA_LIB="${CONDA_PREFIX}/lib"
export LD_LIBRARY_PATH="${_BLENDER_CONDA_LIB}:${LD_LIBRARY_PATH:-}"
unset PYOPENGL_PLATFORM

[ ! -e "${_BLENDER_CONDA_LIB}/libGL.so" ] && ln -sf libGL.so.1 "${_BLENDER_CONDA_LIB}/libGL.so" 2>/dev/null || true
[ ! -e "${_BLENDER_CONDA_LIB}/libOpenGL.so.0" ] && ln -sf libGL.so.1 "${_BLENDER_CONDA_LIB}/libOpenGL.so.0" 2>/dev/null || true
HOOKEOF
  echo "==> Created ${activate_dir}/bpy.sh"
}

install_bpy_runtime() {
  if [[ ! -x "$CONDA" ]]; then
    echo "ERROR: conda not found at $CONDA"
    echo "Set CONDA= path or install miniconda first."
    exit 1
  fi

  eval "$("$CONDA" shell.bash hook)"

  if [[ ! -d "$ENV_PREFIX" ]]; then
    echo "ERROR: conda env not found: $ENV_PREFIX"
    echo "Create it first, e.g.: conda create -p $ENV_PREFIX python=3.11.9 -y"
    exit 1
  fi

  conda activate "$ENV_PREFIX"

  local pyver
  pyver="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$pyver" != "3.11" ]]; then
    echo "WARNING: bpy ${BLENDER_VERSION} requires Python 3.11; current env is Python $pyver"
  fi

  echo "==> Installing bpy==${BLENDER_VERSION} in ${ENV_PREFIX}..."
  pip install --default-timeout=600 "bpy==${BLENDER_VERSION}" pillow

  echo "==> Installing X11 / OpenGL runtime (conda-forge, headless cluster)..."
  "$CONDA" install -y -p "$ENV_PREFIX" -c conda-forge \
    mesalib libgl libglib \
    xorg-libsm xorg-libice xorg-libxxf86vm xorg-libxfixes \
    xorg-libxi xorg-libxrender xorg-libxext xorg-libxrandr \
    libxkbcommon xorg-libx11 -q

  local lib="${ENV_PREFIX}/lib"
  ln -sf libGL.so.1 "${lib}/libGL.so" 2>/dev/null || true
  ln -sf libGL.so.1 "${lib}/libOpenGL.so.0" 2>/dev/null || true

  write_bpy_activate_hook

  export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  unset PYOPENGL_PLATFORM

  echo "==> Verifying bpy import..."
  python -c "
import bpy
print('bpy', bpy.app.version_string)
print('python', __import__('sys').version.split()[0])
"
}

verify_install() {
  local ok=0

  echo "=== Verify Blender install ==="
  if [[ -x "$BLENDER_BIN" ]]; then
    echo "[OK] Binary: $("$BLENDER_BIN" --version | head -1)"
  else
    echo "[MISSING] Binary: $BLENDER_BIN"
    ok=1
  fi

  if [[ -x "$CONDA" && -d "$ENV_PREFIX" ]]; then
    eval "$("$CONDA" shell.bash hook)"
    conda activate "$ENV_PREFIX"
    export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    unset PYOPENGL_PLATFORM
    if python -c "import bpy; print('[OK] bpy', bpy.app.version_string)" 2>/dev/null; then
      :
    else
      echo "[MISSING] bpy in conda env: $ENV_PREFIX"
      ok=1
    fi
  else
    echo "[SKIP] conda env: $ENV_PREFIX"
  fi

  return "$ok"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) MODE=all ;;
    --binary-only) MODE=binary ;;
    --bpy-only) MODE=bpy ;;
    --verify) MODE=verify ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
  shift
done

setup_proxy

echo "=============================================="
echo " Blender ${BLENDER_VERSION} installer"
echo "=============================================="
echo " Mode:    $MODE"
echo " Binary:  $BLENDER_BIN"
echo " Conda:   $ENV_PREFIX"
echo "=============================================="

case "$MODE" in
  all)
    download_blender_binary
    write_blender_env_hook
    install_bpy_runtime
    verify_install
    ;;
  binary)
    download_blender_binary
    write_blender_env_hook
    verify_install || true
    ;;
  bpy)
    install_bpy_runtime
    verify_install || true
    ;;
  verify)
    verify_install
    ;;
esac

echo ""
echo "Done."
echo ""
echo "Quick start:"
echo "  conda activate ${ENV_PREFIX}"
echo "  source ${BLENDER_INSTALL_ROOT}/env/blender.sh   # optional: blender CLI on PATH"
echo "  blender --background --python your_script.py    # headless Blender binary"
echo "  python -c \"import bpy; print(bpy.app.version_string)\"  # bpy in conda"
