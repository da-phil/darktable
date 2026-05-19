// Vulkan port of extended.cl :: colorbalance_cdl.
//
// ASC CDL-style slope/offset/power in ProPhoto-RGB space. Same
// 2-binding push-constant shape as LEGACY/LGG; the lift bias is
// applied as an additive offset (vs LEGACY's multiplicative form),
// and gamma_inv is used directly as the power exponent (no 2.2
// round-trip). PC: 2 ints + 16 floats = 72 bytes.

#include "dt_vulkan_common.h"

static inline float4 vk_cb_pow4(const float4 v, const float4 e)
{
  return (float4)(pow(v.x, e.x), pow(v.y, e.y), pow(v.z, e.z), v.w);
}

kernel void colorbalance_cdl(global const float4 *in,
                             global       float4 *out,
                             const int   width,
                             const int   height,
                             const float lift_r,  const float lift_g,  const float lift_b,  const float lift_w,
                             const float gain_r,  const float gain_g,  const float gain_b,  const float gain_w,
                             const float gam_r,   const float gam_g,   const float gam_b,   const float gam_w,
                             const float saturation,
                             const float contrast,
                             const float grey,
                             const float saturation_out)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 lift      = (float4)(lift_r, lift_g, lift_b, lift_w);
  const float4 gain      = (float4)(gain_r, gain_g, gain_b, gain_w);
  const float4 gamma_inv = (float4)(gam_r,  gam_g,  gam_b,  gam_w);

  float4 Lab = in[idx];
  const float4 XYZ = vk_Lab_to_XYZ(Lab);
  float4 RGB = vk_XYZ_to_prophotorgb(XYZ);

  if(saturation != 1.0f)
  {
    const float luma = XYZ.y;
    RGB.x = luma + saturation * (RGB.x - luma);
    RGB.y = luma + saturation * (RGB.y - luma);
    RGB.z = luma + saturation * (RGB.z - luma);
  }

  // Slope (gain) + offset (lift) + power (gamma_inv).
  RGB = RGB * gain + lift;
  RGB = (RGB <= (float4)0.0f) ? (float4)0.0f : vk_cb_pow4(RGB, gamma_inv);

  if(saturation_out != 1.0f)
  {
    const float luma = vk_prophotorgb_to_XYZ(RGB).y;
    RGB.x = luma + saturation_out * (RGB.x - luma);
    RGB.y = luma + saturation_out * (RGB.y - luma);
    RGB.z = luma + saturation_out * (RGB.z - luma);
  }

  if(contrast != 1.0f)
  {
    const float4 contrast4 = (float4)contrast;
    const float4 grey4     = (float4)grey;
    RGB = (RGB <= (float4)0.0f) ? (float4)0.0f : vk_cb_pow4(RGB / grey4, contrast4) * grey4;
  }

  const float4 lab = vk_prophotorgb_to_Lab(RGB);
  Lab.x = lab.x; Lab.y = lab.y; Lab.z = lab.z;
  out[idx] = Lab;
}
