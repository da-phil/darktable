// Vulkan port of bloom.cl :: bloom_threshold.
//
// First step: extract the bright lights by scaling L and zeroing
// anything below the threshold. Output is a single-channel float
// buffer (the OpenCL kernel writes to an image2d_t allocated with
// sizeof(float), i.e. CL_R format; the broadcast write_imagef(out,
// coord, L) just puts L into the single channel).
//
// Bindings (2 storage buffers):
//   0: in   (float4 Lab)
//   1: out  (float, single-channel light buffer)
// Push constants: 16 B (width, height, scale, threshold).

#include "dt_vulkan_common.h"

kernel void bloom_threshold(global const float4 *in,
                            global       float  *out,
                            const int   width,
                            const int   height,
                            const float scale,
                            const float threshold)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float L = in[idx].x * scale;
  L = L > threshold ? L : 0.0f;
  out[idx] = L;
}
