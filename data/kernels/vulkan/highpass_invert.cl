// Vulkan port of highpass.cl :: highpass_invert.
//
// L-channel invert: pixel.x = clamp(100 - pixel.x, 0, 100). a, b, w
// pass through. Acts as the soft-focus reference's negative — what
// follows is a blur + mix with the original to yield the high-pass.
//
// Bindings (2 storage buffers):
//   0: in   (float4 Lab)
//   1: out  (float4 Lab, L inverted)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void highpass_invert(global const float4 *in,
                            global       float4 *out,
                            const int width,
                            const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  pixel.x = clamp(100.0f - pixel.x, 0.0f, 100.0f);
  out[idx] = pixel;
}
