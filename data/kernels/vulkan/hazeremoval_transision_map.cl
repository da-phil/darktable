// Vulkan port of hazeremoval.cl :: hazeremoval_transision_map.
//
// (The OpenCL spelling — "transision" — is kept for entry-point
// fidelity with the existing module-side kernel-lookup table.)
//
// Per-pixel dark-channel-prior transition map:
//   m = min(pixel.r/A0_r, pixel.g/A0_g, pixel.b/A0_b)
//   out = 1 − m·strength
//
// Bindings (2 storage buffers):
//   0: in   (float4 RGB)
//   1: out  (float, single-channel transition map)
// Push constants: 24 B (width, height, strength, A0_r, A0_g, A0_b).

#include "dt_vulkan_common.h"

kernel void hazeremoval_transision_map(global const float4 *in,
                                       global       float  *out,
                                       const int   width,
                                       const int   height,
                                       const float strength,
                                       const float A0_r,
                                       const float A0_g,
                                       const float A0_b)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 pixel = in[idx];
  float m = pixel.x / A0_r;
  m = fmin(pixel.y / A0_g, m);
  m = fmin(pixel.z / A0_b, m);
  out[idx] = 1.f - m * strength;
}
