// Vulkan port of sharpen.cl :: sharpen_vblur.
//
// Vertical companion to sharpen_hblur. Same design (L-channel only,
// no shared-memory tiling, skip the outermost rad rows).
//
// Bindings (3 storage buffers):
//   0: in   (float4 Lab)
//   1: out  (float4 Lab, L modified)
//   2: m    (Gaussian kernel, 2*rad+1 floats)
// Push constants: 12 B (width, height, rad).

#include "dt_vulkan_common.h"

kernel void sharpen_vblur(global const float4 *in,
                          global       float4 *out,
                          global const float  *m,
                          const int width,
                          const int height,
                          const int rad)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  if(y >= rad && y < height - rad)
  {
    float sum = 0.0f;
    for(int i = -rad; i <= rad; i++)
      sum += in[idx2d(x, y + i, width)].x * m[i + rad];
    pixel.x = sum;
  }
  out[idx] = pixel;
}
