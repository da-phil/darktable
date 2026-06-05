// Vulkan port of retouch.cl :: retouch_copy_buffer_to_buffer
// AND retouch_copy_buffer_to_image. Both have identical semantics
// in our flat-buffer world (image2d_t -> storage buffer in VK), so
// one kernel covers both OpenCL entry points and the host code
// dispatches it under whichever name is required.
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 4 ints (in_width, in_height, out_width, out_height)
//                 + 2 ints (xoffs, yoffs) = 24 B.

#include "dt_vulkan_common.h"

kernel void retouch_copy_buffer_to_buffer(global const float4 *in_buf,
                                          global       float4 *out_buf,
                                          const int in_width, const int in_height,
                                          const int out_width, const int out_height,
                                          const int xoffs, const int yoffs)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= out_width || y >= out_height) return;
  if(x + xoffs >= in_width || y + yoffs >= in_height) return;

  out_buf[y * out_width + x] = in_buf[(y + yoffs) * in_width + (x + xoffs)];
}
