// Vulkan port of nlmeans.cl :: nlmeans_finish.
//
// Normalises the accumulator by its weight channel and blends the
// denoised result with the input per the L/C weight:
//   o = in * (1 - weight) + (U2 / U2.w) * weight   (alpha kept).
//
// Bindings (3 storage buffers):
//   0: in  (float4, input image)
//   1: U2  (float4, accumulator)
//   2: out (float4, output image)
// Push constants: 24 B (width, height, weight[4]).

#include "dt_vulkan_common.h"

kernel void nlmeans_finish(global const float4 *in,
                           global const float4 *U2,
                           global       float4 *out,
                           const int   width,
                           const int   height,
                           const float weight_x,
                           const float weight_y,
                           const float weight_z,
                           const float weight_w)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int gidx = idx2d(x, y, width);

  const float4 weight = (float4)(weight_x, weight_y, weight_z, weight_w);
  const float4 i  = in[gidx];
  const float4 u2 = U2[gidx];
  const float  u3 = u2.w;

  float4 o = i * ((float4)1.0f - weight) + u2 / (float4)u3 * weight;
  o.w = i.w;
  out[gidx] = o;
}
