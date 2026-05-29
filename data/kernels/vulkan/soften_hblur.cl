// Vulkan port of soften.cl :: soften_hblur.
//
// Horizontal Gaussian convolution, length 2*rad+1. The OpenCL kernel
// tiles via workgroup-local memory; the Vulkan twin reads directly
// from the global storage buffer — the math is identical (same
// CLAMP_TO_EDGE boundary via int clamp on the source index, same
// Gaussian weights), only the L1-cache pattern differs.
//
// Bindings (3 storage buffers):
//   0: in   (float4 RGBA)
//   1: out  (float4 RGBA)
//   2: m    (Gaussian kernel, 2*rad+1 floats, already normalised host-side)
// Push constants: 12 B (width, height, rad).

#include "dt_vulkan_common.h"

kernel void soften_hblur(global const float4 *in,
                         global       float4 *out,
                         global const float  *m,
                         const int width,
                         const int height,
                         const int rad)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float4 sum = (float4)0.0f;
  for(int i = -rad; i <= rad; i++)
  {
    const int xx = clamp(x + i, 0, width - 1);
    sum += in[idx2d(xx, y, width)] * m[i + rad];
  }
  out[idx2d(x, y, width)] = sum;
}
