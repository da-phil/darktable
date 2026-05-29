// Vulkan port of soften.cl :: soften_vblur.
//
// Vertical Gaussian convolution, length 2*rad+1. See soften_hblur.cl
// for the design rationale (no shared-memory tiling, identical math).
//
// Bindings (3 storage buffers):
//   0: in   (float4 RGBA)
//   1: out  (float4 RGBA)
//   2: m    (Gaussian kernel, 2*rad+1 floats)
// Push constants: 12 B (width, height, rad).

#include "dt_vulkan_common.h"

kernel void soften_vblur(global const float4 *in,
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
    const int yy = clamp(y + i, 0, height - 1);
    sum += in[idx2d(x, yy, width)] * m[i + rad];
  }
  out[idx2d(x, y, width)] = sum;
}
