// Vulkan port of bloom.cl :: bloom_hblur.
//
// Horizontal box blur of length 2*rad+1: uniform average, no
// Gaussian weights. Both buffers are single-channel float (the
// thresholded light buffer). CLAMP_TO_EDGE via clamp(x+i, 0, w-1)
// on the source index, matching the OpenCL sampler clamp on the
// local-mem wing reads.
//
// Bindings (2 storage buffers):
//   0: in   (float, single-channel)
//   1: out  (float, single-channel)
// Push constants: 12 B (width, height, rad).

#include "dt_vulkan_common.h"

kernel void bloom_hblur(global const float *in,
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
    const int xx = clamp(x + i, 0, width - 1);
    sum += in[idx2d(xx, y, width)];
  }
  out[idx2d(x, y, width)] = sum / (float)(2 * rad + 1);
}
