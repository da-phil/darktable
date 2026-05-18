// Vulkan port of basic.cl :: monochrome — second half of the
// monochrome module. Reads the original pixel + the bilateral-
// filtered weight produced by monochrome_filter + dt_bilateral_*_vk
// (§5.13), and blends into a desaturated monochrome output with
// optional highlight protection.
//
// Bindings:  0 = in (float4)        — original Lab pixels
//            1 = base (float4)      — bilateral-filtered weight
//            2 = out (float4)
// PC: 2 ints + 4 floats = 24 bytes.

#include "dt_vulkan_common.h"

static inline float vk_mono_envelope(const float L)
{
  const float x = clipf(L / 100.0f);
  const float beta = 0.6f;
  if(x < beta)
  {
    const float tmp = fabs(x / beta - 1.0f);
    return 1.0f - tmp * tmp;
  }
  const float tmp1 = (1.0f - x) / (1.0f - beta);
  const float tmp2 = tmp1 * tmp1;
  const float tmp3 = tmp2 * tmp1;
  return 3.0f * tmp2 - 2.0f * tmp3;
}

kernel void monochrome(global const float4 *in,
                       global const float4 *base,
                       global       float4 *out,
                       const int   width,
                       const int   height,
                       const float a,
                       const float b,
                       const float size,
                       const float highlights)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  const float4 basep = base[idx];
  const float tt = vk_mono_envelope(pixel.x);
  const float t  = tt + (1.0f - tt) * (1.0f - highlights);
  pixel.x = mix(pixel.x, pixel.x * basep.x / 100.0f, t);
  pixel.y = 0.0f;
  pixel.z = 0.0f;
  out[idx] = pixel;
}
