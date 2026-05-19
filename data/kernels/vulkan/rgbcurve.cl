// Vulkan port of rgbcurve.cl :: rgbcurve.
//
// Per-channel or norm-preserving RGB tone curve via three 65536-
// entry LUTs with linear-extrapolation tails (the lookup_unbounded
// pattern). The norm-preserving branch uses the ICC profile
// plumbing in §5.11 to compute luminance via the work profile.
//
// Bindings (7 storage buffers):
//   0: in  (float4)
//   1: out (float4)
//   2: table_r (float, 65536 entries)
//   3: table_g (float, 65536 entries)
//   4: table_b (float, 65536 entries)
//   5: profile_info  (vk_dt_colorspaces_iccprofile_info_t)
//   6: profile_lut   (6 * lutsize floats)
//
// Push constants: 5 ints + 9 floats = 56 bytes.

#include "dt_vulkan_common.h"

// Curve-autoscale modes — mirror DT_S_SCALE_AUTOMATIC_RGB /
// DT_S_SCALE_MANUAL_RGB from src/iop/rgbcurve.c.
#define VK_RGBCURVE_AUTOSCALE_AUTOMATIC 0
#define VK_RGBCURVE_AUTOSCALE_MANUAL    1

kernel void rgbcurve(global const float4 *in,
                     global       float4 *out,
                     global const float  *table_r,
                     global const float  *table_g,
                     global const float  *table_b,
                     global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                     global const float  *profile_lut,
                     const int   width,
                     const int   height,
                     const int   autoscale,
                     const int   preserve_colors,
                     const int   use_work_profile,
                     const float cr0, const float cr1, const float cr2,
                     const float cg0, const float cg1, const float cg2,
                     const float cb0, const float cb1, const float cb2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];

  if(autoscale == VK_RGBCURVE_AUTOSCALE_MANUAL)
  {
    pixel.x = vk_lookup_unbounded(table_r, pixel.x, cr0, cr1, cr2);
    pixel.y = vk_lookup_unbounded(table_g, pixel.y, cg0, cg1, cg2);
    pixel.z = vk_lookup_unbounded(table_b, pixel.z, cb0, cb1, cb2);
  }
  else // VK_RGBCURVE_AUTOSCALE_AUTOMATIC
  {
    if(preserve_colors == VK_RGB_NORM_NONE)
    {
      // All three channels run through the R curve in automatic mode.
      pixel.x = vk_lookup_unbounded(table_r, pixel.x, cr0, cr1, cr2);
      pixel.y = vk_lookup_unbounded(table_r, pixel.y, cr0, cr1, cr2);
      pixel.z = vk_lookup_unbounded(table_r, pixel.z, cr0, cr1, cr2);
    }
    else
    {
      float ratio = 1.0f;
      const float lum = vk_dt_rgb_norm(pixel, preserve_colors, use_work_profile,
                                        profile_info, profile_lut);
      if(lum > 0.0f)
      {
        const float curve_lum = vk_lookup_unbounded(table_r, lum, cr0, cr1, cr2);
        ratio = curve_lum / lum;
      }
      pixel.x *= ratio;
      pixel.y *= ratio;
      pixel.z *= ratio;
    }
  }

  out[idx] = pixel;
}
