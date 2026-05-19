// Vulkan port of basic.cl :: tonecurve.
//
// L-channel tone curve plus optional independent a/b curves, with
// four chroma-handling modes (autoscale_ab):
//   0 = independent curves on a, b (with optional two-sided
//       unbounded extrapolation)
//   1 = automatic chroma — scale a, b by L_new / L_old
//   2 = automatic in XYZ — apply L curve to all three XYZ channels
//   3 = automatic in ProPhoto RGB — same, optionally norm-
//       preserving via vk_dt_rgb_norm (§5.11)
//
// Bindings (7 storage buffers):
//   0: in (float4), 1: out (float4),
//   2-4: table_L / table_a / table_b (float, 65536 each)
//   5: profile_info, 6: profile_lut.
//
// Push constants: 5 ints + 16 floats = 84 bytes.

#include "dt_vulkan_common.h"

// Two-sided unbounded LUT lookup — mirrors basic.cl's
// lookup_unbounded_twosided. `aa` is laid out as 6 floats: aa[0..2]
// is the right-side (x >= ar) extrapolation, aa[3..5] is the
// left-side (x < al). When a[0] < 0 the curve is linear, return x.
static inline float vk_lookup_twosided(global const float *lut, const float x,
                                       const float a0, const float a1, const float a2,
                                       const float a3, const float a4, const float a5)
{
  if(a0 < 0.0f) return x;
  const float ar = 1.0f / a0;
  const float al = 1.0f - 1.0f / a3;
  if(x < ar && x >= al) return vk_lookup(lut, x);
  if(x >= ar) return a1 * pow(x * a0, a2);
  // left side: xx = 1 - x
  return a4 * pow((1.0f - x) * a3, a5);
}

kernel void tonecurve(global const float4 *in,
                     global       float4 *out,
                     global const float  *table_L,
                     global const float  *table_a,
                     global const float  *table_b,
                     global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                     global const float  *profile_lut,
                     const int   width,
                     const int   height,
                     const int   autoscale_ab,
                     const int   unbound_ab,
                     const int   preserve_colors,
                     const float low_approximation,
                     // coeffs_L: 3 unbound floats for the L curve
                     const float cL0, const float cL1, const float cL2,
                     // coeffs_ab: 12 floats (6 per channel a, b)
                     const float ca0, const float ca1, const float ca2,
                     const float ca3, const float ca4, const float ca5,
                     const float cb0, const float cb1, const float cb2,
                     const float cb3, const float cb4, const float cb5)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  const float L_in = pixel.x / 100.0f;
  const float L    = vk_lookup_unbounded(table_L, L_in, cL0, cL1, cL2);

  if(autoscale_ab == 0)
  {
    const float a_in = (pixel.y + 128.0f) / 256.0f;
    const float b_in = (pixel.z + 128.0f) / 256.0f;
    if(unbound_ab == 0)
    {
      pixel.y = vk_lookup(table_a, a_in);
      pixel.z = vk_lookup(table_b, b_in);
    }
    else
    {
      pixel.y = vk_lookup_twosided(table_a, a_in,
                                    ca0, ca1, ca2, ca3, ca4, ca5);
      pixel.z = vk_lookup_twosided(table_b, b_in,
                                    cb0, cb1, cb2, cb3, cb4, cb5);
    }
    pixel.x = L;
  }
  else if(autoscale_ab == 1)
  {
    if(L_in > 0.01f)
    {
      const float scale = L / pixel.x;
      pixel.y *= scale;
      pixel.z *= scale;
    }
    else
    {
      pixel.y *= low_approximation;
      pixel.z *= low_approximation;
    }
    pixel.x = L;
  }
  else if(autoscale_ab == 2)
  {
    float4 xyz = vk_Lab_to_XYZ(pixel);
    xyz.x = vk_lookup_unbounded(table_L, xyz.x, cL0, cL1, cL2);
    xyz.y = vk_lookup_unbounded(table_L, xyz.y, cL0, cL1, cL2);
    xyz.z = vk_lookup_unbounded(table_L, xyz.z, cL0, cL1, cL2);
    const float4 lab = vk_XYZ_to_Lab(xyz);
    pixel.x = lab.x;
    pixel.y = lab.y;
    pixel.z = lab.z;
  }
  else // autoscale_ab == 3 — ProPhoto RGB path
  {
    float4 rgb = vk_Lab_to_prophotorgb(pixel);
    if(preserve_colors == VK_RGB_NORM_NONE)
    {
      rgb.x = vk_lookup_unbounded(table_L, rgb.x, cL0, cL1, cL2);
      rgb.y = vk_lookup_unbounded(table_L, rgb.y, cL0, cL1, cL2);
      rgb.z = vk_lookup_unbounded(table_L, rgb.z, cL0, cL1, cL2);
    }
    else
    {
      float ratio = 1.0f;
      const float lum = vk_dt_rgb_norm(rgb, preserve_colors, 1,
                                        profile_info, profile_lut);
      if(lum > 0.0f)
      {
        const float curve_lum = vk_lookup_unbounded(table_L, lum, cL0, cL1, cL2);
        ratio = curve_lum / lum;
      }
      rgb.x *= ratio;
      rgb.y *= ratio;
      rgb.z *= ratio;
    }
    const float4 lab = vk_prophotorgb_to_Lab(rgb);
    pixel.x = lab.x;
    pixel.y = lab.y;
    pixel.z = lab.z;
  }

  out[idx] = pixel;
}
