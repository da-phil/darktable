// Vulkan port of bloom.cl :: bloom_vblur.
//
// Vertical companion to bloom_hblur. Same uniform box average.
//
// Bindings (2 storage buffers):
//   0: in   (float, single-channel)
//   1: out  (float, single-channel)
// Push constants: 12 B (width, height, rad).

#include "dt_vulkan_common.h"

kernel void bloom_vblur(global const float *in,
                       global       float *out,
                       const int width,
                       const int height,
                       const int rad)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float sum = 0.0f;
  for(int i = -rad; i <= rad; i++)
  {
    const int yy = clamp(y + i, 0, height - 1);
    sum += in[idx2d(x, yy, width)];
  }
  out[idx2d(x, y, width)] = sum / (float)(2 * rad + 1);
}
