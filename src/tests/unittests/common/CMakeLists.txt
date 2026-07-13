# Tests for the Vulkan capture executor
# (dev-doc/gpu_resident_pixelpipe_dag.md, milestone M1).
#
# The test kernels are GLSL twins compiled with glslangValidator: the
# tests exercise the HAL's capture/flush scheduler, not clspv codegen,
# so the fallback toolchain is sufficient and keeps the test buildable
# on hosts without clspv. At runtime the tests skip cleanly when no
# Vulkan device (e.g. lavapipe) is available.

find_program(GLSLANG_BIN glslangValidator)

if(NOT GLSLANG_BIN)
  message(WARNING
          "unittests/common: glslangValidator not found — "
          "skipping test_vulkan_capture")
  return()
endif()

set(_capt_spv)
foreach(_k capt_add capt_mul capt_lut)
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${_k}.spv"
    COMMAND "${GLSLANG_BIN}" -V "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${_k}.comp"
            -o "${CMAKE_CURRENT_BINARY_DIR}/${_k}.spv"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${_k}.comp"
    COMMENT "glslang: ${_k}.comp -> ${_k}.spv")
  list(APPEND _capt_spv "${CMAKE_CURRENT_BINARY_DIR}/${_k}.spv")
endforeach()

add_custom_target(test_vulkan_capture_kernels DEPENDS ${_capt_spv})

add_cmocka_test(test_vulkan_capture
                SOURCES test_vulkan_capture.c
                LINK_LIBRARIES lib_darktable cmocka)
add_dependencies(test_vulkan_capture test_vulkan_capture_kernels)
target_compile_definitions(test_vulkan_capture
                           PRIVATE TEST_VK_KERNEL_DIR="${CMAKE_CURRENT_BINARY_DIR}")

# Blendop kernel + host-path tests (DAG milestone M0). These load the
# production .spv modules from the build datadir, so they depend on
# the kernel build target.
add_cmocka_test(test_vulkan_blendop
                SOURCES test_vulkan_blendop.c
                LINK_LIBRARIES lib_darktable cmocka)
add_dependencies(test_vulkan_blendop build_vulkan_kernels)
target_compile_definitions(test_vulkan_blendop
                           PRIVATE TEST_VK_DATADIR="${DARKTABLE_DATADIR}")

# Colorspace glue-node tests (DAG milestone M2). Load the production
# .spv modules from the build datadir, so they depend on the kernel
# build target.
add_cmocka_test(test_vulkan_colorspace
                SOURCES test_vulkan_colorspace.c
                LINK_LIBRARIES lib_darktable cmocka)
add_dependencies(test_vulkan_colorspace build_vulkan_kernels)
target_compile_definitions(test_vulkan_colorspace
                           PRIVATE TEST_VK_DATADIR="${DARKTABLE_DATADIR}")

# Histogram reduction-tap tests (DAG milestone M5). Load the production
# .spv modules from the build datadir, so they depend on the kernel
# build target.
add_cmocka_test(test_vulkan_histogram
                SOURCES test_vulkan_histogram.c
                LINK_LIBRARIES lib_darktable cmocka)
add_dependencies(test_vulkan_histogram build_vulkan_kernels)
target_compile_definitions(test_vulkan_histogram
                           PRIVATE TEST_VK_DATADIR="${DARKTABLE_DATADIR}")

# Color-picker reduction-tap tests (DAG milestone M5).
add_cmocka_test(test_vulkan_picker
                SOURCES test_vulkan_picker.c
                LINK_LIBRARIES lib_darktable cmocka)
add_dependencies(test_vulkan_picker build_vulkan_kernels)
target_compile_definitions(test_vulkan_picker
                           PRIVATE TEST_VK_DATADIR="${DARKTABLE_DATADIR}")

# Windows: libs have to be copied next to the executable
if(WIN32)
    _copy_required_library(test_vulkan_capture lib_darktable)
    _copy_required_library(test_vulkan_blendop lib_darktable)
    _copy_required_library(test_vulkan_colorspace lib_darktable)
    _copy_required_library(test_vulkan_histogram lib_darktable)
    _copy_required_library(test_vulkan_picker lib_darktable)
endif(WIN32)
