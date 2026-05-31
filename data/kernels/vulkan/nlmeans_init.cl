// Vulkan port of nlmeans.cl :: nlmeans_init.
//
// Zeros the float4 accumulator U2 (the (Σ w·neighbour, Σ w) buffer).
//
// Bindings (1 storage buffer):
//   0: out (float4, the U2 accumulator)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void nlmeans_init(global float4 *out,
                         const int width,
                         const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  out[idx2d(x, y, width)] = (float4)0.0f;
}
