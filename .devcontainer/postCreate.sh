#!/usr/bin/env bash
set -euo pipefail

cd /workspace/gsplat

python -m pip install --upgrade pip setuptools wheel

# VS Code bind mounts can show up with a different owner inside the container.
# Mark the workspace as safe so git/submodule operations remain reliable.
git config --global --add safe.directory /workspace/gsplat

# The CUDA extension depends on the vendored glm headers provided via git submodule.
git submodule update --init
git config --global --add safe.directory /workspace/gsplat/gsplat/cuda/csrc/third_party/glm

# Match CUDA 12.8 image and RTX 50-series hardware.
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128

# Core dependencies used by gsplat source build/runtime.
python -m pip install \
  ninja \
  numpy \
  jaxtyping \
  rich \
  tyro \
  Pillow

# Install gsplat from local source in editable mode using the already-installed
# toolchain instead of creating an isolated build env with another torch.
python -m pip install -e . --no-build-isolation

# Keep container creation stable. The full examples stack pulls several moving
# git dependencies; install it manually only when needed.
#   python -m pip install -r examples/requirements.txt

echo "gsplat dev container setup complete."
