// Vulkan port of highpass.cl :: highpass_vblur.
//
// Vertical companion to highpass_hblur. Same shape: convolve L
// with CLAMP_TO_EDGE on the source index, chroma passes through.
//
// Bindings (3 storage buffers):
//   0: in   (float4 Lab)
//   1: out  (float4 Lab, L blurred)
//   2: m    (Gaussian kernel, 2*rad+1 floats)
// Push constants: 12 B (width, height, rad).

#include "dt_vulkan_common.h"

kernel void highpass_vblur(global const float4 *in,
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
  float sum = 0.0f;
  for(int i = -rad; i <= rad; i++)
  {
    const int yy = clamp(y + i, 0, height - 1);
    sum += in[idx2d(x, yy, width)].x * m[i + rad];
  }
  pixel.x = sum;
  out[idx] = pixel;
}
