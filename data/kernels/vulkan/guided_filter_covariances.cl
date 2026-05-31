// Vulkan port of guided_filter.cl :: guided_filter_covariances.
//
// Per-pixel products img * (weighted, clipped) guide channel, the raw
// covariance terms cov(imgg, img) before box-meaning.
//
// Bindings (5 storage buffers):
//   0: guide  (float4 RGBA)
//   1: img    (float)
//   2: cov_r  (float)
//   3: cov_g  (float)
//   4: cov_b  (float)
// Push constants: 16 B (width, height, first, guide_weight).

#include "dt_vulkan_common.h"

kernel void guided_filter_covariances(global const float4 *guide,
                                      global const float  *img,
                                      global       float  *cov_r,
                                      global       float  *cov_g,
                                      global       float  *cov_b,
                                      const int   width,
                                      const int   height,
                                      const int   first,
                                      const float guide_weight)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float img_ = img[idx];
  const float4 g = fmin(100.0f, guide[idx2d(x, y + first, width)]);
  cov_r[idx] = img_ * guide_weight * g.x;
  cov_g[idx] = img_ * guide_weight * g.y;
  cov_b[idx] = img_ * guide_weight * g.z;
}
