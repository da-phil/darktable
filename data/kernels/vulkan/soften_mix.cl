// Vulkan port of soften.cl :: soften_mix.
//
// Final blend of the original with the soft-focus (blurred-overexposed)
// reference: pixel = orig * (1 - amount) + clip(processed) * amount.
// Alpha is taken from the original.
//
// Bindings (3 storage buffers):
//   0: in_a  (float4 RGBA, the original input)
//   1: in_b  (float4 RGBA, the blurred soft-focus reference)
//   2: out   (float4 RGBA)
// Push constants: 12 B (width, height, amount).

#include "dt_vulkan_common.h"

kernel void soften_mix(global const float4 *in_a,
                       global const float4 *in_b,
                       global       float4 *out,
                       const int   width,
                       const int   height,
                       const float amount)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 orig = in_a[idx];
  const float4 proc = in_b[idx];
  float4 pix = orig * (1.0f - amount) + clamp(proc, (float4)0.0f, (float4)1.0f) * amount;
  pix.w = orig.w;
  out[idx] = pix;
}
