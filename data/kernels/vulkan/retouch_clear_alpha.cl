// Vulkan port of retouch.cl :: retouch_clear_alpha.
//
// Zeroes the alpha channel of every pixel in a float4 buffer.
// Used when retouch wants to expose the mask via the blend alpha.
//
// Binding layout (1 storage buffer):
//   0: in  (float4, in-place)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void retouch_clear_alpha(global float4 *in_buf,
                                const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float4 p = in_buf[y * width + x];
  p.w = 0.f;
  in_buf[y * width + x] = p;
}
