// Vulkan port of extended.cl :: colorbalance (LEGACY variant).
//
// Lift / gamma / gain in sRGB space — round-trips through Lab →
// XYZ → linear sRGB and back. Push-constant-only, no LUTs.
//
// Bindings: 0 = in (float4), 1 = out (float4).
// PC: 2 ints + 15 floats = 68 bytes (3 colour triples × 4 + 3 scalars).
// `_w` slots on the colour triples are padding for std430 packing
// uniformity vs the .comp twin.

#include "dt_vulkan_common.h"

static inline float4 vk_cb_pow4(const float4 v, const float4 e)
{
  return (float4)(pow(v.x, e.x), pow(v.y, e.y), pow(v.z, e.z), v.w);
}

kernel void colorbalance(global const float4 *in,
                         global       float4 *out,
                         const int   width,
                         const int   height,
                         const float lift_r,  const float lift_g,  const float lift_b,  const float lift_w,
                         const float gain_r,  const float gain_g,  const float gain_b,  const float gain_w,
                         const float gam_r,   const float gam_g,   const float gam_b,   const float gam_w,
                         const float saturation,
                         const float contrast,
                         const float grey)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  (void)saturation; (void)contrast; (void)grey;

  const float4 lift      = (float4)(lift_r, lift_g, lift_b, lift_w);
  const float4 gain      = (float4)(gain_r, gain_g, gain_b, gain_w);
  const float4 gamma_inv = (float4)(gam_r,  gam_g,  gam_b,  gam_w);

  float4 Lab = in[idx];
  float4 sRGB = vk_XYZ_to_sRGB(vk_Lab_to_XYZ(Lab));

  // linear sRGB -> sRGB
  sRGB = (sRGB <= (float4)0.0031308f)
            ? 12.92f * sRGB
            : (1.0f + 0.055f) * vk_cb_pow4(sRGB, (float4)(1.0f / 2.4f, 1.0f / 2.4f, 1.0f / 2.4f, 1.0f))
              - (float4)0.055f;

  // lift, gain, gamma
  sRGB = vk_cb_pow4(fmax(((sRGB - (float4)1.0f) * lift + (float4)1.0f) * gain, (float4)0.0f), gamma_inv);

  // back to linear sRGB
  sRGB = (sRGB <= (float4)0.04045f)
            ? sRGB / 12.92f
            : vk_cb_pow4((sRGB + (float4)0.055f) / (1.0f + 0.055f),
                          (float4)(2.4f, 2.4f, 2.4f, 1.0f));

  const float4 lab = vk_XYZ_to_Lab(vk_sRGB_to_XYZ(sRGB));
  Lab.x = lab.x; Lab.y = lab.y; Lab.z = lab.z;
  out[idx] = Lab;
}
