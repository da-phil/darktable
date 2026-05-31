// Vulkan port of hazeremoval.cl :: hazeremoval_box_max_x.
// See hazeremoval_box_min_x for the design rationale; this is the
// max twin (seeded with -INFINITY, fmax instead of fmin).

#include "dt_vulkan_common.h"

kernel void hazeremoval_box_max_x(global const float *in,
                                  global       float *out,
                                  const int width,
                                  const int height,
                                  const int w)
{
  const int y = get_global_id(0);
  if(y >= height) return;
  const int row = y * width;

  float m = -INFINITY;
  for(int i = 0, i_end = min(w + 1, width); i < i_end; ++i)
    m = fmax(in[row + i], m);
  for(int i = 0; i < width; i++)
  {
    out[row + i] = m;
    if(i - w >= 0 && in[row + i - w] == m)
    {
      m = -INFINITY;
      for(int j = max(i - w + 1, 0), j_end = min(i + w + 2, width); j < j_end; ++j)
        m = fmax(in[row + j], m);
    }
    if(i + w + 1 < width) m = fmax(in[row + i + w + 1], m);
  }
}
