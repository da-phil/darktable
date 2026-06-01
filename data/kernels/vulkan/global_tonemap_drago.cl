// Vulkan port of extended.cl :: global_tonemap_drago.
//
// Drago adaptive logarithmic tone mapping:
//   L = 100 * (ldc * log(max(eps, lw+1)) /
//                  log(max(eps, 2 + pow(lw/lwmax, bl) * 8)))
// where lw = L_in * 0.01 and (eps, ldc, bl, lwmax) are the four
// parameters. Chroma + alpha pass through.
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 2 ints + 4 floats = 24 bytes
//   (width, height, eps, ldc, bl, lwmax).

#include "dt_vulkan_common.h"

kernel void global_tonemap_drago(global const float4 *in,
                                 global       float4 *out,
                                 const int width,
                                 const int height,
                                 const float eps,
                                 const float ldc,
                                 const float bl,
                                 const float lwmax)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;

  float4 pixel = in[idx];
  const float lw = pixel.x * 0.01f;
  pixel.x = 100.0f * (ldc * log(fmax(eps, lw + 1.0f))
                          / log(fmax(eps, 2.0f + (pow(lw / lwmax, bl)) * 8.0f)));
  out[idx] = pixel;
}
