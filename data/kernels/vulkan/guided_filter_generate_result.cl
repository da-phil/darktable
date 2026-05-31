// Vulkan port of guided_filter.cl :: guided_filter_generate_result.
//
// Applies the box-meaned coefficients to the guide:
//   res = clamp(guide_weight * (a.guide) + b, min, max).
//
// Bindings (6 storage buffers):
//   0: guide (float4 RGBA)
//   1: a_r   (float)
//   2: a_g   (float)
//   3: a_b   (float)
//   4: b     (float)
//   5: res   (float)
// Push constants: 24 B (width, height, first, guide_weight, minval, maxval).

#include "dt_vulkan_common.h"

kernel void guided_filter_generate_result(global const float4 *guide,
                                          global const float  *a_r,
                                          global const float  *a_g,
                                          global const float  *a_b,
                                          global const float  *b,
                                          global       float  *res,
                                          const int   width,
                                          const int   height,
                                          const int   first,
                                          const float guide_weight,
                                          const float minval,
                                          const float maxval)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 g = fmin(100.0f, guide[idx2d(x, y + first, width)]);
  const float v = guide_weight * (g.x * a_r[idx] + g.y * a_g[idx] + g.z * a_b[idx]) + b[idx];
  res[idx] = fmin(maxval, fmax(minval, v));
}
