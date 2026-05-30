// Vulkan port of atrous.cl :: eaw_synthesize.
//
// Soft-thresholded detail accumulation: amount = copysign(max(0,
// |detail| - threshold), detail); out = coarse + boost * amount.
// Alpha taken from coarse.
//
// Bindings (3 storage buffers):
//   0: out     (float4)
//   1: coarse  (float4, low-pass at this scale)
//   2: detail  (float4, high-frequency at this scale)
// Push constants: 40 B (width, height, threshold[4], boost[4]).
// (threshold/boost stored as float[4] so the std430 layout is the
// same 4-byte-aligned packing as the host C struct.)

#include "dt_vulkan_common.h"

kernel void eaw_synthesize(global       float4 *out,
                           global const float4 *coarse,
                           global const float4 *detail,
                           const int width,
                           const int height,
                           const float threshold0, const float threshold1,
                           const float threshold2, const float threshold3,
                           const float boost0,     const float boost1,
                           const float boost2,     const float boost3)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 c = coarse[idx];
  const float4 d = detail[idx];
  const float4 threshold = (float4)(threshold0, threshold1, threshold2, threshold3);
  const float4 boost     = (float4)(boost0,     boost1,     boost2,     boost3);

  const float4 amount = copysign(fmax((float4)0.0f, fabs(d) - threshold), d);
  float4 sum = c + boost * amount;
  sum.w = c.w;
  out[idx] = sum;
}
