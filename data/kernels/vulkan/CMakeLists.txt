# Build-time compilation of the Vulkan-targeted kernels to SPIR-V.
#
# Production path: clspv compiles each .cl source to a .spv module
# with an entry point named after the OpenCL kernel function (e.g.
# "velvia"). When clspv isn't installed, we fall back to glslang on
# the matching .comp twin, using --source-entrypoint + -e to give
# the output module the same entry-point name as the clspv path.
# This keeps the runtime kernel-lookup uniform.
#
# Only built when USE_VULKAN=ON.

find_program(CLSPV_BIN clspv)
find_program(GLSLANG_BIN glslangValidator)
find_program(SPIRV_VAL_BIN spirv-val)

if(NOT CLSPV_BIN AND NOT GLSLANG_BIN)
  message(FATAL_ERROR
          "USE_VULKAN=ON but neither clspv nor glslangValidator was found in PATH. "
          "Install one (or set CLSPV_BIN / GLSLANG_BIN explicitly) to build the "
          "Vulkan kernels under data/kernels/vulkan/.")
endif()

# Each entry is "kernel_filename:entry_point". For most kernels the
# .cl filename matches the kernel function name and the .spv has a
# single entry point. A few .cl sources have multiple `kernel void`
# entries (currently: channelmixerrgb's 5 adaptation kernels); clspv
# emits all entry points into one .spv automatically, while the
# glslang fallback can only expose the single entry named in the
# spec. Host code probes each entry by name and falls back when it
# isn't present — modules that need extra entry points just won't
# have them available in glslang-only builds.
set(VK_KERNELS
    exposure:exposure
    channelmixerrgb:channelmixerrgb_bradford_linear
    velvia:velvia
    invert:invert_4f
    vibrance:vibrance
    colorcorrection:colorcorrection
    colorcontrast:colorcontrast
    colorize:colorize
    flip:flip
    negadoctor:negadoctor
    primaries:primaries
    temperature:whitebalance_4f
    profile_gamma_log:profilegamma_log
    profile_gamma:profilegamma
    graduatednd:graduatednd
    relight:relight
    vignette:vignette
    splittoning:splittoning
    zonesystem:zonesystem
    levels:levels
    whitebalance_1f:whitebalance_1f
    whitebalance_1f_xtrans:whitebalance_1f_xtrans)

set(VK_SPV_FILES "")
foreach(SPEC IN LISTS VK_KERNELS)
  string(REPLACE ":" ";" SPEC_LIST "${SPEC}")
  list(GET SPEC_LIST 0 K)
  list(GET SPEC_LIST 1 ENTRY)

  set(CL_SRC   "${CMAKE_CURRENT_SOURCE_DIR}/${K}.cl")
  set(COMP_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${K}.comp")
  set(SPV_OUT  "${CMAKE_CURRENT_BINARY_DIR}/${K}.spv")

  if(CLSPV_BIN)
    add_custom_command(
      OUTPUT  "${SPV_OUT}"
      COMMAND "${CLSPV_BIN}"
              "-cl-std=CL1.2"
              "--inline-entry-points"
              "--cl-kernel-arg-info"
              -I "${CMAKE_CURRENT_SOURCE_DIR}"
              -o "${SPV_OUT}"
              "${CL_SRC}"
      DEPENDS "${CL_SRC}" "${CMAKE_CURRENT_SOURCE_DIR}/dt_vulkan_common.h"
      COMMENT "clspv: ${K}.cl -> ${K}.spv  (entry: ${ENTRY})"
    )
  else()
    # glslang renames its GLSL "main" entry to the kernel's actual
    # entry-point name so the runtime can address it identically to
    # the clspv output.
    add_custom_command(
      OUTPUT  "${SPV_OUT}"
      COMMAND "${GLSLANG_BIN}"
              -V --source-entrypoint main -e "${ENTRY}"
              "${COMP_SRC}" -o "${SPV_OUT}"
      DEPENDS "${COMP_SRC}"
      COMMENT "glslang: ${K}.comp -> ${K}.spv  (entry: ${ENTRY})"
    )
  endif()

  if(SPIRV_VAL_BIN)
    add_custom_command(OUTPUT "${SPV_OUT}.validated"
                       COMMAND "${SPIRV_VAL_BIN}" "${SPV_OUT}"
                       COMMAND "${CMAKE_COMMAND}" -E touch "${SPV_OUT}.validated"
                       DEPENDS "${SPV_OUT}"
                       COMMENT "spirv-val: ${K}.spv")
    list(APPEND VK_SPV_FILES "${SPV_OUT}.validated")
  endif()
  list(APPEND VK_SPV_FILES "${SPV_OUT}")
endforeach()

add_custom_target(build_vulkan_kernels ALL DEPENDS ${VK_SPV_FILES})

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/darktable/kernels/vulkan"
        COMPONENT DTApplication
        FILES_MATCHING PATTERN "*.spv")

foreach(SPEC IN LISTS VK_KERNELS)
  string(REPLACE ":" ";" SPEC_LIST "${SPEC}")
  list(GET SPEC_LIST 0 K)
  add_custom_command(TARGET build_vulkan_kernels POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${DARKTABLE_DATADIR}/kernels/vulkan"
    COMMAND "${CMAKE_COMMAND}" -E copy
            "${CMAKE_CURRENT_BINARY_DIR}/${K}.spv"
            "${DARKTABLE_DATADIR}/kernels/vulkan/${K}.spv"
  )
endforeach()
