// Vulkan port of extended.cl :: global_tonemap_filmic.
//
// Hejl/Burgess-Dawson filmic tone mapping: with m = max(0, L*0.01 −
// 0.004), L_out = 100 * (m*(6.2*m+0.5)) / (m*(6.2*m+1.7)+0.06). The
// `parameters` float4 is part of the OpenCL signature but unused.
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 2 ints + 4 floats = 24 bytes (parameters unused).

#include "dt_vulkan_common.h"

kernel void global_tonemap_filmic(global const float4 *in,
                                  global       float4 *out,
                                  const int width,
                                  const int height,
                                  const float p0,
                                  const float p1,
                                  const float p2,
                                  const float p3)
{
  (void)p0; (void)p1; (void)p2; (void)p3;
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;

  float4 pixel = in[idx];
  const float l = pixel.x * 0.01f;
  const float m = fmax(0.0f, l - 0.004f);
  pixel.x = 100.0f * ((m * (6.2f * m + 0.5f))
                      / (m * (6.2f * m + 1.7f) + 0.06f));
  out[idx] = pixel;
}
