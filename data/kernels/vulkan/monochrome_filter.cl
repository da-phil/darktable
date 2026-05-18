// Vulkan port of basic.cl :: monochrome_filter — first half of
// the monochrome module. Replaces .x with a per-pixel filter weight
// derived from the user's (a, b) target chromaticity and size; the
// host then runs the bilateral helper over this output to produce
// the smooth blend mask.
//
// Bindings:  0 = in (float4), 1 = out (float4)
// PC: 2 ints + 3 floats = 20 bytes.

#include "dt_vulkan_common.h"

static inline float vk_mono_fast_expf(const float x)
{
  // meant for x in [-100, 0]. Same approximation as the OpenCL
  // dt_fast_expf in common.h — kept here so the kernel is self-
  // contained.
  const int i1 = 0x3f800000;
  const int i2 = 0x402DF854;
  const int k0 = i1 + (int)(x * (float)(i2 - i1));
  union { float f; int k; } u;
  u.k = k0 > 0 ? k0 : 0;
  return u.f;
}

kernel void monochrome_filter(global const float4 *in,
                              global       float4 *out,
                              const int   width,
                              const int   height,
                              const float a,
                              const float b,
                              const float size)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  const float da = pixel.y - a;
  const float db = pixel.z - b;
  pixel.x = 100.0f * vk_mono_fast_expf(-clipf((da * da + db * db) / (2.0f * size)));
  out[idx] = pixel;
}
