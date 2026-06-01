// Vulkan port of extended.cl :: global_tonemap_reinhard.
//
// Simple Reinhard tone mapping: L = 100 * (l/(1+l)) on L only,
// chroma + alpha pass through. The `parameters` float4 is part of
// the OpenCL signature but unused for this operator (kept here so
// the three global-tonemap kernels share the same PC layout).
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 2 ints + 4 floats = 24 bytes (parameters unused).

#include "dt_vulkan_common.h"

kernel void global_tonemap_reinhard(global const float4 *in,
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
  pixel.x = 100.0f * (l / (1.0f + l));
  out[idx] = pixel;
}
