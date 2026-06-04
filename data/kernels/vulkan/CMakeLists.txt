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
    borders:borders_fill
    colisa:colisa
    graduatednd:graduatednd
    relight:relight
    vignette:vignette
    gaussian:gaussian_column_4c
    gaussian_row_1c:gaussian_row_1c
    gaussian_column_1c:gaussian_column_1c
    gaussian_row_2c:gaussian_row_2c
    gaussian_column_2c:gaussian_column_2c
    splittoning:splittoning
    zonesystem:zonesystem
    levels:levels
    whitebalance_1f:whitebalance_1f
    whitebalance_1f_xtrans:whitebalance_1f_xtrans
    overexposed:overexposed
    basicadj:basicadj
    colorspaces:colorspaces_transform_rgb_matrix_to_rgb
    lowlight:lowlight
    bilateral_zero:bilateral_zero
    bilateral_splat:bilateral_splat
    bilateral_blur_line:bilateral_blur_line
    bilateral_blur_line_z:bilateral_blur_line_z
    bilateral_slice:bilateral_slice
    bilateral_slice_to_output:bilateral_slice_to_output
    monochrome_filter:monochrome_filter
    monochrome:monochrome
    rgbcurve:rgbcurve
    rgblevels:rgblevels
    tonecurve:tonecurve
    lowpass_mix:lowpass_mix
    shadows_highlights_mix:shadows_highlights_mix
    colorbalance:colorbalance
    colorbalance_lgg:colorbalance_lgg
    colorbalance_cdl:colorbalance_cdl
    colorout:colorout
    filmic:filmic
    filmicrgb:filmicrgb
    overlay:overlay_blend
    sigmoid_loglogistic_rgb_ratio:sigmoid_loglogistic_rgb_ratio
    sigmoid_loglogistic_per_channel:sigmoid_loglogistic_per_channel
    agx:kernel_agx
    channelmixer:channelmixer
    colorharmonizer_map:colorharmonizer_map
    colorharmonizer_apply:colorharmonizer_apply
    colorbalancergb:colorbalancergb
    basecurve_legacy_lut:basecurve_legacy_lut
    basecurve_lut:basecurve_lut
    colorchecker:colorchecker
    colorzones:colorzones
    colorzones_v3:colorzones_v3
    soften_overexposed:soften_overexposed
    soften_hblur:soften_hblur
    soften_vblur:soften_vblur
    soften_mix:soften_mix
    sharpen_hblur:sharpen_hblur
    sharpen_vblur:sharpen_vblur
    sharpen_mix:sharpen_mix
    highpass_invert:highpass_invert
    highpass_hblur:highpass_hblur
    highpass_vblur:highpass_vblur
    highpass_mix:highpass_mix
    blurs_convolve:convolve
    blurs_convolve_sparse:convolve_sparse
    blurs_restore_alpha:restore_alpha
    bloom_threshold:bloom_threshold
    bloom_hblur:bloom_hblur
    bloom_vblur:bloom_vblur
    bloom_mix:bloom_mix
    eaw_decompose:eaw_decompose
    eaw_synthesize:eaw_synthesize
    guided_filter_split_rgb:guided_filter_split_rgb_image
    guided_filter_box_mean_x:guided_filter_box_mean_x
    guided_filter_box_mean_y:guided_filter_box_mean_y
    guided_filter_covariances:guided_filter_covariances
    guided_filter_variances:guided_filter_variances
    guided_filter_update_covariance:guided_filter_update_covariance
    guided_filter_solve:guided_filter_solve
    guided_filter_generate_result:guided_filter_generate_result
    hazeremoval_box_min_x:hazeremoval_box_min_x
    hazeremoval_box_min_y:hazeremoval_box_min_y
    hazeremoval_box_max_x:hazeremoval_box_max_x
    hazeremoval_box_max_y:hazeremoval_box_max_y
    hazeremoval_transision_map:hazeremoval_transision_map
    hazeremoval_dehaze:hazeremoval_dehaze
    colorreconstruction_zero:colorreconstruction_zero
    colorreconstruction_splat:colorreconstruction_splat
    colorreconstruction_blur_line:colorreconstruction_blur_line
    colorreconstruction_slice:colorreconstruction_slice
    nlmeans_init:nlmeans_init
    nlmeans_dist:nlmeans_dist
    nlmeans_horiz:nlmeans_horiz
    nlmeans_vert:nlmeans_vert
    nlmeans_accu:nlmeans_accu
    nlmeans_finish:nlmeans_finish
    dwt_init_buffer:dwt_init_buffer
    dwt_add_img_to_layer:dwt_add_img_to_layer
    dwt_subtract_layer:dwt_subtract_layer
    dwt_hat_transform_row:dwt_hat_transform_row
    dwt_hat_transform_col:dwt_hat_transform_col
    global_tonemap_reinhard:global_tonemap_reinhard
    global_tonemap_drago:global_tonemap_drago
    global_tonemap_filmic:global_tonemap_filmic
    colormapping_histogram:colormapping_histogram
    colormapping_mapping:colormapping_mapping
    ll_pad_input:ll_pad_input
    ll_gauss_reduce:ll_gauss_reduce
    ll_process_curve:ll_process_curve
    ll_laplacian_assemble:ll_laplacian_assemble
    ll_write_back:ll_write_back
    colorin_unbound:colorin_unbound
    colorin_clipping:colorin_clipping
    lut3d_none:lut3d_none
    lut3d_trilinear:lut3d_trilinear
    lut3d_tetrahedral:lut3d_tetrahedral
    lut3d_pyramid:lut3d_pyramid
    liquify_warp:liquify_warp
    interpolation_resample:interpolation_resample
    interpolation_copy:interpolation_copy
    ashift_bilinear:ashift_bilinear
    ashift_bicubic:ashift_bicubic
    ashift_lanczos2:ashift_lanczos2
    ashift_lanczos3:ashift_lanczos3
    clip_rotate_bilinear:clip_rotate_bilinear
    clip_rotate_bicubic:clip_rotate_bicubic
    clip_rotate_lanczos2:clip_rotate_lanczos2
    clip_rotate_lanczos3:clip_rotate_lanczos3
    rawprepare_4f:rawprepare_4f
    highlights_4f_clip:highlights_4f_clip
    censorize_pixelate:censorize_pixelate
    censorize_noise:censorize_noise)

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
