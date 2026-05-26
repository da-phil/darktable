// Vulkan port of basecurve.cl :: basecurve_lut.
//
// Norm-preserving base curve: compute the work-profile luminance norm,
// apply the 65536-entry curve LUT to it, then scale RGB by the ratio.
// Used when "preserve colors" != DT_RGB_NORM_NONE. Reuses the §5.11 ICC
// profile plumbing (vk_dt_rgb_norm). The exposure-fusion path stays on
// OpenCL/CPU and is gated off via process_vk_ready.
//
// Bindings (5 storage buffers):
//   0 in, 1 out, 2 table, 3 profile_info, 4 profile_lut.
// Push constants: 4 ints + 4 floats = 32 bytes.

#include "dt_vulkan_common.h"

kernel void basecurve_lut(global const float4 *in,
                          global       float4 *out,
                          global const float  *table,
                          global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                          global const float  *profile_lut,
                          const int   width,
                          const int   height,
                          const int   preserve_colors,
                          const int   use_work_profile,
                          const float mul,
                          const float c0, const float c1, const float c2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  float ratio = 1.0f;
  const float lum = mul * vk_dt_rgb_norm(pixel, preserve_colors, use_work_profile,
                                          profile_info, profile_lut);
  if(lum > 0.0f)
  {
    const float curve_lum = vk_lookup_unbounded(table, lum, c0, c1, c2);
    ratio = mul * curve_lum / lum;
  }
  pixel.x *= ratio;
  pixel.y *= ratio;
  pixel.z *= ratio;
  pixel = fmax(pixel, 0.0f);
  out[idx] = pixel;
}
