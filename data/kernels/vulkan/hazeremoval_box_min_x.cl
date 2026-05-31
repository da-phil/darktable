// Vulkan port of hazeremoval.cl :: hazeremoval_box_min_x.
//
// Horizontal running-box minimum over a single-channel float buffer
// with window radius w. 1D dispatch — one work item per row. Math
// is byte-for-byte the OpenCL kernel (re-seed the sliding min when
// the value that left the window equals the current min).
//
// Bindings (2 storage buffers):
//   0: in   (float)
//   1: out  (float)
// Push constants: 12 B (width, height, w).

#include "dt_vulkan_common.h"

kernel void hazeremoval_box_min_x(global const float *in,
                                  global       float *out,
                                  const int width,
                                  const int height,
                                  const int w)
{
  const int y = get_global_id(0);
  if(y >= height) return;
  const int row = y * width;

  float m = INFINITY;
  for(int i = 0, i_end = min(w + 1, width); i < i_end; ++i)
    m = fmin(in[row + i], m);
  for(int i = 0; i < width; i++)
  {
    out[row + i] = m;
    if(i - w >= 0 && in[row + i - w] == m)
    {
      m = INFINITY;
      for(int j = max(i - w + 1, 0), j_end = min(i + w + 2, width); j < j_end; ++j)
        m = fmin(in[row + j], m);
    }
    if(i + w + 1 < width) m = fmin(in[row + i + w + 1], m);
  }
}
