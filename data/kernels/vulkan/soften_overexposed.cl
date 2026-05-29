// Vulkan port of soften.cl :: soften_overexposed.
//
// Soften's "overexposed" pre-pass: a global saturation + brightness
// boost in HSL, the soft-focus reference image that is then blurred
// and blended back over the original. Bit-equal to soften.cl: the
// vk_RGB_to_HSL / vk_HSL_to_RGB twins in dt_vulkan_common.h are the
// exact byte-for-byte equivalents of colorspace.h.
//
// Bindings (2 storage buffers):
//   0: in   (float4 RGBA)
//   1: out  (float4 RGBA)
// Push constants: 16 B (width, height, saturation, brightness).

#include "dt_vulkan_common.h"

kernel void soften_overexposed(global const float4 *in,
                               global       float4 *out,
                               const int   width,
                               const int   height,
                               const float saturation,
                               const float brightness)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  float4 hsl   = vk_RGB_to_HSL(pixel);
  hsl.y = clamp(hsl.y * saturation, 0.0f, 1.0f);
  hsl.z = clamp(hsl.z * brightness, 0.0f, 1.0f);
  out[idx] = vk_HSL_to_RGB(hsl);
}
