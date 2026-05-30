// Vulkan port of highpass.cl :: highpass_mix.
//
// High-pass blend: out.L = 50 + ((0.5*a.L + 0.5*b.L) - 50) *
// contrast_scale, out.a = out.b = 0 (desaturated), out.w = a.w.
// Final clamp to [(0,-128,-128,-inf), (100,128,128,+inf)] matches
// the OpenCL min/max.
//
// Bindings (3 storage buffers):
//   0: in_a  (float4 Lab, original)
//   1: in_b  (float4 Lab, blurred inverted reference; only .x is read)
//   2: out   (float4 Lab)
// Push constants: 12 B (width, height, contrast_scale).

#include "dt_vulkan_common.h"

kernel void highpass_mix(global const float4 *in_a,
                         global const float4 *in_b,
                         global       float4 *out,
                         const int   width,
                         const int   height,
                         const float contrast_scale)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 a = in_a[idx];
  const float4 b = in_b[idx];
  float4 o;
  o.x = 50.0f + ((0.5f * a.x + 0.5f * b.x) - 50.0f) * contrast_scale;
  o.y = 0.0f;
  o.z = 0.0f;
  o.w = a.w;
  out[idx] = clamp(o,
                   (float4)(0.0f,   -128.0f, -128.0f, -INFINITY),
                   (float4)(100.0f,  128.0f,  128.0f,  INFINITY));
}
