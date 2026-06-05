// Vulkan port of retouch.cl :: retouch_copy_alpha.
//
// Copies the alpha channel from an input image (float4 buffer in VK)
// into the alpha of an output float4 buffer. Same dims for both.
//
// Binding layout (2 storage buffers):
//   0: in   (float4) — source image
//   1: out  (float4) — destination, alpha is overwritten

#include "dt_vulkan_common.h"

kernel void retouch_copy_alpha(global const float4 *in_buf,
                               global       float4 *out_buf,
                               const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int idx = y * width + x;
  float4 p = out_buf[idx];
  p.w = in_buf[idx].w;
  out_buf[idx] = p;
}
