// Vulkan port of hazeremoval.cl :: hazeremoval_box_max_y.
// Vertical companion to hazeremoval_box_max_x.

#include "dt_vulkan_common.h"

kernel void hazeremoval_box_max_y(global const float *in,
                                  global       float *out,
                                  const int width,
                                  const int height,
                                  const int w)
{
  const int x = get_global_id(0);
  if(x >= width) return;

  float m = -INFINITY;
  for(int i = 0, i_end = min(w + 1, height); i < i_end; ++i)
    m = fmax(in[idx2d(x, i, width)], m);
  for(int i = 0; i < height; i++)
  {
    out[idx2d(x, i, width)] = m;
    if(i - w >= 0 && in[idx2d(x, i - w, width)] == m)
    {
      m = -INFINITY;
      for(int j = max(i - w + 1, 0), j_end = min(i + w + 2, height); j < j_end; ++j)
        m = fmax(in[idx2d(x, j, width)], m);
    }
    if(i + w + 1 < height) m = fmax(in[idx2d(x, i + w + 1, width)], m);
  }
}
