// Vulkan port of rgblevels.cl :: rgblevels.
//
// Per-channel or norm-preserving levels remap with three 65536-
// entry LUTs (one per channel) covering the [low, high] interior
// range and a power-of-inv_gamma extrapolation for x ≥ high.
//
// Bindings (7 storage buffers):
//   0: in  (float4)
//   1: out (float4)
//   2: lutr (float, 65536)
//   3: lutg (float, 65536)
//   4: lutb (float, 65536)
//   5: profile_info  (vk_dt_colorspaces_iccprofile_info_t)
//   6: profile_lut   (6 * lutsize floats)
//
// Push constants: 5 ints + 12 floats = 68 bytes.

#include "dt_vulkan_common.h"

// Autoscale modes — mirror DT_IOP_RGBLEVELS_LINKED_CHANNELS (0)
// / DT_IOP_RGBLEVELS_INDEPENDENT_CHANNELS (1) in src/iop/rgblevels.c.
#define VK_RGBLEVELS_LINKED      0
#define VK_RGBLEVELS_INDEPENDENT 1

// Single-channel levels remap helper. Below low → 0, above high →
// gamma-power extrapolation, between → LUT lookup of the
// normalised position in [low, high].
static inline float vk_rgblevels_1c(const float v,
                                    const float lo, const float mid, const float hi,
                                    const float inv_gamma,
                                    global const float *lut)
{
  (void)mid;
  if(v <= lo) return 0.0f;
  if(v >= hi)
  {
    const float pct = (v - lo) / (hi - lo);
    return pow(pct, inv_gamma);
  }
  const float pct = (v - lo) / (hi - lo);
  return vk_lookup(lut, pct);
}

kernel void rgblevels(global const float4 *in,
                      global       float4 *out,
                      global const float  *lutr,
                      global const float  *lutg,
                      global const float  *lutb,
                      global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                      global const float  *profile_lut,
                      const int   width,
                      const int   height,
                      const int   autoscale,
                      const int   preserve_colors,
                      const int   use_work_profile,
                      const float r_lo, const float r_mid, const float r_hi,
                      const float g_lo, const float g_mid, const float g_hi,
                      const float b_lo, const float b_mid, const float b_hi,
                      const float inv_gamma_r,
                      const float inv_gamma_g,
                      const float inv_gamma_b)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];

  if(autoscale == VK_RGBLEVELS_INDEPENDENT || preserve_colors == VK_RGB_NORM_NONE)
  {
    pixel.x = vk_rgblevels_1c(pixel.x, r_lo, r_mid, r_hi, inv_gamma_r, lutr);
    pixel.y = vk_rgblevels_1c(pixel.y, g_lo, g_mid, g_hi, inv_gamma_g, lutg);
    pixel.z = vk_rgblevels_1c(pixel.z, b_lo, b_mid, b_hi, inv_gamma_b, lutb);
  }
  else
  {
    float ratio = 0.0f;
    const float lum = vk_dt_rgb_norm(pixel, preserve_colors, use_work_profile,
                                      profile_info, profile_lut);
    if(lum > r_lo)
    {
      const float curve_lum = vk_rgblevels_1c(lum, r_lo, r_mid, r_hi, inv_gamma_r, lutr);
      ratio = curve_lum / lum;
    }
    pixel.x *= ratio;
    pixel.y *= ratio;
    pixel.z *= ratio;
  }

  out[idx] = pixel;
}
