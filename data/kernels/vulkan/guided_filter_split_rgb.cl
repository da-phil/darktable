// Vulkan port of guided_filter.cl :: guided_filter_split_rgb_image.
//
// Splits the (clipped, weighted) RGB guide image into three
// single-channel float planes. Part of the dt_guided_filter_vk
// helper (shared by hazeremoval, colorequal, …).
//
// In the VK buffer model the guide is a flat float4 storage buffer and
// the three outputs are flat single-channel float buffers; the OpenCL
// `first` y-offset is kept for fidelity but the non-tiled VK helper
// always passes 0.
//
// Bindings (4 storage buffers):
//   0: guide  (float4 RGBA)
//   1: out_r  (float)
//   2: out_g  (float)
//   3: out_b  (float)
// Push constants: 16 B (width, height, first, guide_weight).

#include "dt_vulkan_common.h"

kernel void guided_filter_split_rgb_image(global const float4 *guide,
                                          global       float  *out_r,
                                          global       float  *out_g,
                                          global       float  *out_b,
                                          const int   width,
                                          const int   height,
                                          const int   first,
                                          const float guide_weight)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 g = fmin(100.0f, guide[idx2d(x, y + first, width)]);
  out_r[idx] = guide_weight * g.x;
  out_g[idx] = guide_weight * g.y;
  out_b[idx] = guide_weight * g.z;
}
