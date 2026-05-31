// Vulkan port of guided_filter.cl :: guided_filter_update_covariance.
//
// out = in - a*b + eps  (turns a box-meaned product into a centred
// covariance/variance; eps regularises the diagonal entries).
//
// Bindings (4 storage buffers):
//   0: in   (float, box-meaned product)
//   1: out  (float)
//   2: a    (float, box-meaned channel mean)
//   3: b    (float, box-meaned channel mean)
// Push constants: 12 B (width, height, eps).

#include "dt_vulkan_common.h"

kernel void guided_filter_update_covariance(global const float *in,
                                            global       float *out,
                                            global const float *a,
                                            global const float *b,
                                            const int   width,
                                            const int   height,
                                            const float eps)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  out[idx] = in[idx] - a[idx] * b[idx] + eps;
}
