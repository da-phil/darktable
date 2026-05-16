# Build-time compilation of the Vulkan-targeted kernels to SPIR-V.
#
# Production path: clspv compiles the .cl sources to .spv (single
# source of truth, same as the OpenCL kernels under data/kernels/).
#
# Fallback path: if clspv isn't installed, glslangValidator compiles
# the .comp GLSL twins instead. Both produce equivalent .spv modules;
# the C++ host loads them by the same .spv filename.
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

set(VK_KERNELS exposure channelmixerrgb diffuse)

set(VK_SPV_FILES "")
foreach(K IN LISTS VK_KERNELS)
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
              -o "${SPV_OUT}"
              "${CL_SRC}"
      DEPENDS "${CL_SRC}"
      COMMENT "clspv: ${K}.cl -> ${K}.spv"
    )
  else()
    add_custom_command(
      OUTPUT  "${SPV_OUT}"
      COMMAND "${GLSLANG_BIN}" -V "${COMP_SRC}" -o "${SPV_OUT}"
      DEPENDS "${COMP_SRC}"
      COMMENT "glslang: ${K}.comp -> ${K}.spv"
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

# Install alongside the OpenCL kernels under a `vulkan/` subdirectory.
# Runtime kernel loading uses the same DARKTABLE_DATADIR/kernels root.
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/darktable/kernels/vulkan"
        COMPONENT DTApplication
        FILES_MATCHING PATTERN "*.spv")

# Also copy to the in-tree datadir so an uninstalled run finds them.
foreach(K IN LISTS VK_KERNELS)
  add_custom_command(TARGET build_vulkan_kernels POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
            "${DARKTABLE_DATADIR}/kernels/vulkan"
    COMMAND "${CMAKE_COMMAND}" -E copy
            "${CMAKE_CURRENT_BINARY_DIR}/${K}.spv"
            "${DARKTABLE_DATADIR}/kernels/vulkan/${K}.spv"
  )
endforeach()
