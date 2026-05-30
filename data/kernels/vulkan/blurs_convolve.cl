// Vulkan port of blurs.cl :: convolve.
//
// Dense 2D convolution with a (2*radius+1)² kernel. Used as the
// fallback when the sparse kernel buffers can't be allocated; matches
// the OpenCL kernel byte-for-byte (CLAMP_TO_EDGE on the source index,
// alpha taken from the centre pixel).
//
// Bindings (3 storage buffers):
//   0: in    (float4 RGBA)
//   1: kern  (kernel_width * kernel_width floats, row-major)
//   2: out   (float4 RGBA)
// Push constants: 16 B (width, height, radius, kernel_width).

#include "dt_vulkan_common.h"

kernel void convolve(global const float4 *in,
                     global const float  *kern,
                     global       float4 *out,
                     const int width,
                     const int height,
                     const int radius,
                     const int kernel_width)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 pix_in = in[idx];
  float4 acc = (float4)0.0f;
  for(int l = -radius; l <= radius; l++)
    for(int m = -radius; m <= radius; m++)
    {
      const int ii = clamp(y + l, 0, height - 1);
      const int jj = clamp(x + m, 0, width - 1);
      const int ik = l + radius;
      const int jk = m + radius;
      acc += kern[ik * kernel_width + jk] * in[idx2d(jj, ii, width)];
    }
  acc.w = pix_in.w;
  out[idx] = acc;
}
