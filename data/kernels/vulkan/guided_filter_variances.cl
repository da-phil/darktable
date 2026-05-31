// Vulkan port of guided_filter.cl :: guided_filter_variances.
//
// Per-pixel products of the (weighted, clipped) guide channels with
// each other — the 6 unique entries of the symmetric 3x3 guide
// covariance matrix, before box-meaning.
//
// Bindings (7 storage buffers):
//   0: guide   (float4 RGBA)
//   1: var_rr  (float)
//   2: var_rg  (float)
//   3: var_rb  (float)
//   4: var_gg  (float)
//   5: var_gb  (float)
//   6: var_bb  (float)
// Push constants: 16 B (width, height, first, guide_weight).

#include "dt_vulkan_common.h"

kernel void guided_filter_variances(global const float4 *guide,
                                    global       float  *var_rr,
                                    global       float  *var_rg,
                                    global       float  *var_rb,
                                    global       float  *var_gg,
                                    global       float  *var_gb,
                                    global       float  *var_bb,
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
  const float r = guide_weight * g.x;
  const float gg = guide_weight * g.y;
  const float b = guide_weight * g.z;
  var_rr[idx] = r  * r;
  var_rg[idx] = r  * gg;
  var_rb[idx] = r  * b;
  var_gg[idx] = gg * gg;
  var_gb[idx] = gg * b;
  var_bb[idx] = b  * b;
}
