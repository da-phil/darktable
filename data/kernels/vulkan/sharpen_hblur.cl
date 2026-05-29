// Vulkan port of sharpen.cl :: sharpen_hblur.
//
// Horizontal Gaussian convolution applied to the L channel only
// (sharpen runs in Lab; only the L component of the pixel is blurred,
// chroma is passed through). The OpenCL kernel tiles via workgroup-
// local memory; the Vulkan twin reads straight from the global storage
// buffer. Math is identical: same Gaussian weights, same skip-the-
// outermost-rad-pixels semantic — for x in [rad, width-rad), the
// convolution writes a blurred L; otherwise the pixel passes through
// unchanged. Inside the convolution zone the i-shifts stay strictly
// in-bounds, so no clamp is needed.
//
// Bindings (3 storage buffers):
//   0: in   (float4 Lab)
//   1: out  (float4 Lab, L modified)
//   2: m    (Gaussian kernel, 2*rad+1 floats, host-normalised)
// Push constants: 12 B (width, height, rad).

#include "dt_vulkan_common.h"

kernel void sharpen_hblur(global const float4 *in,
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
  if(x >= rad && x < width - rad)
  {
    float sum = 0.0f;
    for(int i = -rad; i <= rad; i++)
      sum += in[idx2d(x + i, y, width)].x * m[i + rad];
    pixel.x = sum;
  }
  out[idx] = pixel;
}
