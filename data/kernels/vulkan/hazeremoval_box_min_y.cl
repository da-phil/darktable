// Vulkan port of hazeremoval.cl :: hazeremoval_box_min_y.
//
// Vertical companion to hazeremoval_box_min_x. 1D dispatch — one
// work item per column.
//
// Bindings (2 storage buffers):
//   0: in   (float)
//   1: out  (float)
// Push constants: 12 B (width, height, w).

#include "dt_vulkan_common.h"

kernel void hazeremoval_box_min_y(global const float *in,
                                  global       float *out,
                                  const int width,
                                  const int height,
                                  const int w)
{
  const int x = get_global_id(0);
  if(x >= width) return;

  float m = INFINITY;
  for(int i = 0, i_end = min(w + 1, height); i < i_end; ++i)
    m = fmin(in[idx2d(x, i, width)], m);
  for(int i = 0; i < height; i++)
  {
    out[idx2d(x, i, width)] = m;
    if(i - w >= 0 && in[idx2d(x, i - w, width)] == m)
    {
      m = INFINITY;
      for(int j = max(i - w + 1, 0), j_end = min(i + w + 2, height); j < j_end; ++j)
        m = fmin(in[idx2d(x, j, width)], m);
    }
    if(i + w + 1 < height) m = fmin(in[idx2d(x, i + w + 1, width)], m);
  }
}
