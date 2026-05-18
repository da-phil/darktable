#!/usr/bin/env bash
#
# Quick verification harness for the Vulkan kernel set.
#
# Runs the three cheap layers from §7 "Testing a Vulkan kernel port"
# in dev-doc/gpu_acceleration_clspv_vulkan.md:
#
#   1. spirv-val on every produced .spv module
#   2. spirv-dis push-constant offset dump (informational — visual
#      cross-check against the C-side vk_<module>_pc_t structs)
#   3. dt_vk_compute_poc dispatch against lavapipe (or any other
#      Vulkan ICD that's installed)
#
# Returns 0 if all three layers pass, 1 otherwise.
#
# Usage from a darktable build tree:
#
#   ./tools/vulkan_compute_poc/smoke_test.sh [path-to-darktable-build]
#
# Defaults to ./build relative to the repo root.

set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="${1:-${REPO_ROOT}/build}"
KERNEL_DIR="${BUILD_DIR}/share/darktable/kernels/vulkan"
POC_BIN="${REPO_ROOT}/tools/vulkan_compute_poc/build/dt_vk_compute_poc"
POC_SPV="${REPO_ROOT}/tools/vulkan_compute_poc/build/basicadj_min.spv"

failed=0

if [ ! -d "${KERNEL_DIR}" ]; then
    echo "smoke_test: ${KERNEL_DIR} not found — build with USE_VULKAN=ON first" >&2
    exit 1
fi

# ---- 1. spirv-val every module ----
echo "== spirv-val =="
if ! command -v spirv-val >/dev/null 2>&1; then
    echo "  spirv-val not in PATH — skipping (install spirv-tools)"
else
    for spv in "${KERNEL_DIR}"/*.spv; do
        name=$(basename "${spv}")
        if spirv-val "${spv}" >/dev/null 2>&1; then
            printf "  OK   %s\n" "${name}"
        else
            printf "  FAIL %s\n" "${name}"
            spirv-val "${spv}"
            failed=$((failed + 1))
        fi
    done
fi

# ---- 2. spirv-dis push-constant offset dump ----
echo
echo "== push-constant offsets (cross-check against vk_<module>_pc_t) =="
if ! command -v spirv-dis >/dev/null 2>&1; then
    echo "  spirv-dis not in PATH — skipping"
else
    for spv in "${KERNEL_DIR}"/*.spv; do
        name=$(basename "${spv}" .spv)
        offsets=$(spirv-dis "${spv}" 2>/dev/null \
                  | grep "OpMemberDecorate %PC" \
                  | awk '{print $NF}' | tr '\n' ' ')
        if [ -n "${offsets}" ]; then
            printf "  %-30s  %s\n" "${name}" "${offsets}"
        fi
    done
fi

# ---- 3. PoC dispatch on whatever Vulkan ICD is available ----
echo
echo "== PoC dispatch =="
if [ -x "${POC_BIN}" ] && [ -f "${POC_SPV}" ]; then
    if "${POC_BIN}" --spv "${POC_SPV}" 2>&1 | tee /dev/stderr | grep -q "PoC OK"; then
        echo "  OK"
    else
        rc=$?
        if [ "${rc}" = "77" ]; then
            echo "  no Vulkan device — skipping"
        else
            echo "  FAIL"
            failed=$((failed + 1))
        fi
    fi
else
    echo "  PoC binary not built — skipping (cmake -DBUILD_VULKAN_COMPUTE_POC=ON)"
fi

echo
if [ "${failed}" -eq 0 ]; then
    echo "smoke_test: all checks passed"
    exit 0
else
    echo "smoke_test: ${failed} check(s) failed"
    exit 1
fi
