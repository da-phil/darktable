// Vulkan port of colorspaces.cl :: colorspaces_transform_rgb_matrix_to_rgb.
//
// Applies a working-space → output-space RGB transform: optional
// input TRC LUT (if profile_info_from->nonlinearlut), 3×3 matrix
// product, optional output TRC LUT (if profile_info_to->nonlinearlut).
//
// Used by the host-side helper dt_ioppr_transform_image_colorspace_rgb_vk
// to remove CPU round-trips in modules that consume two different
// profiles (e.g. overexposed: working profile → histogram profile).
//
// Binding layout (6 storage buffers):
//   0: in     (float4)  — input pixels in the FROM working space
//   1: out    (float4)  — output pixels in the TO working space
//   2: profile_info_from  — vk_dt_colorspaces_iccprofile_info_t
//   3: lut_from           — 6·lutsize tone-curve LUT (FROM)
//   4: profile_info_to    — vk_dt_colorspaces_iccprofile_info_t
//   5: lut_to             — 6·lutsize tone-curve LUT (TO)
// Push constants: 2 ints + 9 floats (combined matrix) = 44 bytes.
//
// The host pre-combines profile_info_to->matrix_out * profile_info_from->matrix_in
// into the 9-float matrix passed via push constants so the kernel
// only has to do one matmul.

#include "dt_vulkan_common.h"

kernel void colorspaces_transform_rgb_matrix_to_rgb
  (global const float4 *in,
   global       float4 *out,
   global const vk_dt_colorspaces_iccprofile_info_t *profile_info_from,
   global const float  *lut_from,
   global const vk_dt_colorspaces_iccprofile_info_t *profile_info_to,
   global const float  *lut_to,
   const int   width,
   const int   height,
   const float m0, const float m1, const float m2,
   const float m3, const float m4, const float m5,
   const float m6, const float m7, const float m8)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  if(profile_info_from->nonlinearlut)
    pixel = vk_apply_trc_in(pixel, profile_info_from, lut_from);

  // matrix product: row-major 3x3 stored in 9 floats via push constants.
  const float4 r = (float4)(m0 * pixel.x + m1 * pixel.y + m2 * pixel.z,
                            m3 * pixel.x + m4 * pixel.y + m5 * pixel.z,
                            m6 * pixel.x + m7 * pixel.y + m8 * pixel.z,
                            pixel.w);

  out[idx] = profile_info_to->nonlinearlut
              ? vk_apply_trc_out(r, profile_info_to, lut_to)
              : r;
}
