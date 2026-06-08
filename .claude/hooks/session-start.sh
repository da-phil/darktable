#!/bin/bash
# SessionStart hook: install clspv so Claude Code on the web can build
# darktable's Vulkan kernels via the production clspv path (rather than
# the glslang fallback, which can't emit multi-entry-point .spv).
#
# Behaviour:
#   - Only runs in the remote (web) environment ($CLAUDE_CODE_REMOTE=true).
#     Skips on local Claude Code installs so it doesn't trigger
#     30-60 min builds on your laptop.
#   - Idempotent: skips the build if `clspv` is already on PATH.
#   - Asynchronous: the session starts immediately and clspv builds
#     in the background. Builds that finish before the agent needs
#     clspv "just work"; if the agent is faster than the build, it
#     falls back to the bundled glslang path.
#   - Persists CLSPV_BIN to $CLAUDE_ENV_FILE so subsequent darktable
#     configure runs find the binary without sudo install.
#
# First-session cost: ~30 min on 8 cores, ~60-75 min on 4 cores.
# Subsequent sessions in the same container reuse the built binary.
#
# See dev-doc/gpu_acceleration_clspv_vulkan.md "Installing clspv" for
# the manual build steps this script automates.

set -euo pipefail

# Announce async mode BEFORE any other stdout so the session can start
# while the build runs. asyncTimeout matches the upper bound of the
# clspv build on a 4-core container.
echo '{"async": true, "asyncTimeout": 5400000}'

# Web sessions only — don't trigger a multi-hour build on someone's
# laptop just because they opened the repo locally.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

CLSPV_CACHE_DIR="${HOME}/.cache/clspv-build"
CLSPV_BIN_PATH="${CLSPV_CACHE_DIR}/clspv/build/bin/clspv"

# Already installed system-wide?
if command -v clspv >/dev/null 2>&1; then
  echo "clspv already on PATH at $(command -v clspv)"
  exit 0
fi

# Already built in the cache from a prior session in this container?
if [ -x "${CLSPV_BIN_PATH}" ]; then
  echo "clspv cached at ${CLSPV_BIN_PATH}"
  echo "export CLSPV_BIN=\"${CLSPV_BIN_PATH}\"" >> "${CLAUDE_ENV_FILE}"
  echo "export PATH=\"${CLSPV_CACHE_DIR}/clspv/build/bin:\${PATH}\"" >> "${CLAUDE_ENV_FILE}"
  exit 0
fi

echo "Building clspv from source (this takes 30-60 min on the container)..."

# Build deps
sudo apt-get update -qq
sudo apt-get install -y -qq cmake ninja-build python3 git build-essential

mkdir -p "${CLSPV_CACHE_DIR}"
cd "${CLSPV_CACHE_DIR}"

# Clone + fetch
if [ ! -d clspv ]; then
  git clone --depth 1 https://github.com/google/clspv.git
fi
cd clspv

if [ ! -d third_party/llvm ]; then
  python3 utils/fetch_sources.py --shallow
fi

# Configure (skip if already configured)
if [ ! -f build/build.ninja ]; then
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
fi

# Build (incremental if partially done)
cmake --build build -j"$(nproc)"

if [ ! -x "${CLSPV_BIN_PATH}" ]; then
  echo "ERROR: clspv build completed but binary not found at ${CLSPV_BIN_PATH}" >&2
  exit 1
fi

# Persist for subsequent commands in this session.
echo "export CLSPV_BIN=\"${CLSPV_BIN_PATH}\"" >> "${CLAUDE_ENV_FILE}"
echo "export PATH=\"${CLSPV_CACHE_DIR}/clspv/build/bin:\${PATH}\"" >> "${CLAUDE_ENV_FILE}"

echo "clspv build complete: ${CLSPV_BIN_PATH}"
"${CLSPV_BIN_PATH}" --version
