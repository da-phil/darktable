// Vulkan port of dwt.cl :: dwt_init_buffer.
//
// Zeroes a float4 image buffer. The host helper uses it to clear the
// accumulator (`layers`) and the optional merged-layer accumulator
// before walking the wavelet scales.
//
// Binding layout (1 storage buffer):
//   0: buffer (float4)
// Push constants: 2 ints = 8 bytes.

#include "dt_vulkan_common.h"

kernel void dwt_init_buffer(global float4 *buffer,
                            const int width,
                            const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  buffer[y * width + x] = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
}
