#!/bin/bash
# Build flashinfer from source with the system's CUDA version
# This ensures flashinfer is compiled against the correct CUDA version
# No sudo required - all files saved to $LFS_HOME/.local
#
# Prerequisites:
#   - CUDA_HOME must be set to the CUDA installation directory
#   - LFS_HOME must be set to the Local File System HOME directory (e.g., could be $HOME)
#   - nvcc must be in PATH
#
# Usage:
#   source script/cluster/build_flashinfer.sh

set -e  # Exit on error

echo "========================================"
echo "Building flashinfer from source"
echo "========================================"

# Verify required environment variables
if [ -z "$LFS_HOME" ]; then
    echo "ERROR: LFS_HOME environment variable not set"
    echo "Please set it to the LFS home directory"
    exit 1
fi

if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: CUDA_HOME environment variable not set"
    echo "Please set it to the CUDA installation directory"
    echo "Example: export CUDA_HOME=/usr/local/cuda-12.4"
    exit 1
fi

# Verify CUDA installation exists
if [ ! -d "$CUDA_HOME" ]; then
    echo "ERROR: CUDA_HOME directory does not exist: $CUDA_HOME"
    exit 1
fi

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "ERROR: nvcc not found at $CUDA_HOME/bin/nvcc"
    exit 1
fi

# Set CUDACXX for the build
export CUDACXX="$CUDA_HOME/bin/nvcc"

# Extract CUDA version for build directory naming
CUDA_VERSION=$("$CUDA_HOME/bin/nvcc" --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
if [ -z "$CUDA_VERSION" ]; then
    echo "ERROR: Could not detect CUDA version"
    CUDA_VERSION="unknown"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  LFS_HOME: $LFS_HOME"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA_VERSION: $CUDA_VERSION"
echo "  CUDACXX: $CUDACXX"
echo "  nvcc version:"
"$CUDA_HOME/bin/nvcc" --version | head -4

# Store current directory to return later
ORIGINAL_DIR=$(pwd)

echo ""
echo "Step 1: Activate virtual environment"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv/bin/activate"
    echo "Please run from the project root directory"
    exit 1
fi

echo ""
echo "Step 2: Uninstall existing flashinfer wheel"
uv pip uninstall flashinfer-python 2>/dev/null || true

echo ""
echo "Step 3: Install build dependencies"
uv pip install ninja setuptools wheel

echo ""
echo "Step 4: Create build directory in $LFS_HOME/.local"
BUILD_DIR="$LFS_HOME/.local/flashinfer-cuda-$CUDA_VERSION"
mkdir -p "$BUILD_DIR"
echo "  Build directory: $BUILD_DIR"

echo ""
echo "Step 5: Clone flashinfer repository"
cd "$BUILD_DIR"
if [ -d "flashinfer" ]; then
    echo "  Removing existing clone..."
    rm -rf flashinfer
fi
git clone --recursive https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer

echo ""
echo "Step 6: Checkout latest stable version"
# -- Get latest tag
# LATEST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.5.3")
# -- Set latest tag
LATEST_TAG="v0.5.3"
echo "  Using version: $LATEST_TAG"
git checkout "$LATEST_TAG"
git submodule update --init --recursive

echo ""
echo "Step 7: Build flashinfer with CUDA $CUDA_VERSION"
echo "  This may take 5-10 minutes..."
uv pip install -v -e . --no-build-isolation

echo ""
echo "Step 8: Verify installation"
python -c "import flashinfer; print('flashinfer version:', flashinfer.__version__)"

echo ""
echo "Step 9: Return to project directory"
cd "$ORIGINAL_DIR"

echo ""
echo "Note: Build files kept at $BUILD_DIR for debugging"
echo "      To remove: rm -rf $BUILD_DIR"

echo ""
echo "========================================"
echo "✓ flashinfer built successfully with CUDA $CUDA_VERSION!"
echo "========================================"
echo ""
