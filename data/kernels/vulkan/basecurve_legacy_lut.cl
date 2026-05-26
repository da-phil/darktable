// Vulkan port of basecurve.cl :: basecurve_legacy_lut.
//
// Legacy per-channel base curve: apply one 65536-entry LUT (with
// linear-extrapolation tail) to each RGB channel independently. Used
// when "preserve colors" is DT_RGB_NORM_NONE. The exposure-fusion path
// (Laplacian pyramids over image2d) stays on OpenCL/CPU and is gated
// off via process_vk_ready (§10.2 predictive pattern).
//
// Bindings (3 storage buffers): 0 in, 1 out, 2 table.
// Push constants: 2 ints + 4 floats = 24 bytes.

#include "dt_vulkan_common.h"

kernel void basecurve_legacy_lut(global const float4 *in,
                                 global       float4 *out,
                                 global const float  *table,
                                 const int   width,
                                 const int   height,
                                 const float mul,
                                 const float c0, const float c1, const float c2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in[idx];
  pixel.x = vk_lookup_unbounded(table, mul * pixel.x, c0, c1, c2);
  pixel.y = vk_lookup_unbounded(table, mul * pixel.y, c0, c1, c2);
  pixel.z = vk_lookup_unbounded(table, mul * pixel.z, c0, c1, c2);
  pixel = fmax(pixel, 0.0f);
  out[idx] = pixel;
}
