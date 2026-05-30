// Vulkan port of bloom.cl :: bloom_mix.
//
// Screen-blend the bloomed-and-blurred bright light back over the
// original L channel: out.L = 100 - ((100 - in_a.L)*(100 - in_b))/100.
// Chroma and alpha pass through unchanged.
//
// Bindings (3 storage buffers):
//   0: in_a (float4 Lab, original)
//   1: in_b (float, blurred light buffer)
//   2: out  (float4 Lab)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void bloom_mix(global const float4 *in_a,
                     global const float  *in_b,
                     global       float4 *out,
                     const int width,
                     const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in_a[idx];
  const float processed = in_b[idx];
  pixel.x = 100.0f - ((100.0f - pixel.x) * (100.0f - processed)) / 100.0f;
  out[idx] = pixel;
}
