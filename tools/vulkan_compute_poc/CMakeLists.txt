#
# darktable Vulkan compute proof-of-concept
#
# This is a standalone build target intentionally kept out of the main
# darktable graph. It demonstrates the proposed clspv -> SPIR-V ->
# Vulkan compute pipeline that will eventually replace
# src/common/opencl.{c,h} on Linux/Windows and bypass macOS Apple OpenCL
# in favour of MoltenVK. See dev-doc/gpu_acceleration_clspv_vulkan.md
# for the design rationale.
#
# Usage from a darktable build tree:
#
#   cmake -S . -B build -DBUILD_VULKAN_COMPUTE_POC=ON
#   cmake --build build --target dt_vk_compute_poc
#   ./build/tools/vulkan_compute_poc/dt_vk_compute_poc --spv <path-to.spv>
#

cmake_minimum_required(VERSION 3.18)

project(dt_vk_compute_poc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

# Tooling discovery. We prefer clspv (matches the production path: same
# .cl source as src/data/kernels), but fall back to glslangValidator on
# the GLSL twin so the PoC stays exercisable on hosts that haven't
# installed clspv yet.
find_program(CLSPV_BIN clspv)
find_program(GLSLANG_BIN glslangValidator)
find_program(SPIRV_DIS_BIN spirv-dis)
find_program(SPIRV_VAL_BIN spirv-val)

set(POC_KERNEL_CL   "${CMAKE_CURRENT_SOURCE_DIR}/kernels/basicadj_min.cl")
set(POC_KERNEL_COMP "${CMAKE_CURRENT_SOURCE_DIR}/kernels/basicadj_min.comp")
set(POC_SPV         "${CMAKE_CURRENT_BINARY_DIR}/basicadj_min.spv")
set(POC_SPV_TXT     "${CMAKE_CURRENT_BINARY_DIR}/basicadj_min.spv.txt")

if(CLSPV_BIN)
    message(STATUS "vk-poc: using clspv (${CLSPV_BIN}) -- production path")
    add_custom_command(
        OUTPUT "${POC_SPV}"
        COMMAND "${CLSPV_BIN}"
                "-cl-std=CL1.2"
                "--inline-entry-points"
                "--cl-kernel-arg-info"
                -o "${POC_SPV}"
                "${POC_KERNEL_CL}"
        DEPENDS "${POC_KERNEL_CL}"
        COMMENT "clspv: ${POC_KERNEL_CL} -> ${POC_SPV}"
    )
    set(POC_SPV_SOURCE_LABEL "clspv(${POC_KERNEL_CL})")
elseif(GLSLANG_BIN)
    message(STATUS "vk-poc: clspv not found, falling back to glslang on the GLSL twin")
    add_custom_command(
        OUTPUT "${POC_SPV}"
        COMMAND "${GLSLANG_BIN}" -V "${POC_KERNEL_COMP}" -o "${POC_SPV}"
        DEPENDS "${POC_KERNEL_COMP}"
        COMMENT "glslang: ${POC_KERNEL_COMP} -> ${POC_SPV}"
    )
    set(POC_SPV_SOURCE_LABEL "glslang(${POC_KERNEL_COMP})")
else()
    message(FATAL_ERROR
            "vk-poc: neither clspv nor glslangValidator found in PATH. "
            "Install one to build the proof-of-concept.")
endif()

if(SPIRV_VAL_BIN)
    add_custom_command(
        OUTPUT "${POC_SPV}.validated"
        COMMAND "${SPIRV_VAL_BIN}" "${POC_SPV}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${POC_SPV}.validated"
        DEPENDS "${POC_SPV}"
        COMMENT "spirv-val: ${POC_SPV}"
    )
    list(APPEND POC_EXTRA_OUTPUTS "${POC_SPV}.validated")
endif()

if(SPIRV_DIS_BIN)
    add_custom_command(
        OUTPUT "${POC_SPV_TXT}"
        COMMAND "${SPIRV_DIS_BIN}" "${POC_SPV}" -o "${POC_SPV_TXT}"
        DEPENDS "${POC_SPV}"
        COMMENT "spirv-dis: ${POC_SPV} -> ${POC_SPV_TXT}"
    )
    list(APPEND POC_EXTRA_OUTPUTS "${POC_SPV_TXT}")
endif()

add_custom_target(dt_vk_compute_poc_kernel ALL DEPENDS "${POC_SPV}" ${POC_EXTRA_OUTPUTS})

add_executable(dt_vk_compute_poc
    src/main.cc
    src/vk_compute.cc
)
target_link_libraries(dt_vk_compute_poc PRIVATE Vulkan::Vulkan)
target_include_directories(dt_vk_compute_poc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
add_dependencies(dt_vk_compute_poc dt_vk_compute_poc_kernel)

# A check target that runs the PoC against its compiled kernel.
add_custom_target(dt_vk_compute_poc_check
    COMMAND $<TARGET_FILE:dt_vk_compute_poc> --spv "${POC_SPV}"
    DEPENDS dt_vk_compute_poc
    COMMENT "Running dt_vk_compute_poc against ${POC_SPV_SOURCE_LABEL}"
)
