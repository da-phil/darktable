// Vulkan port of locallaplacian.cl :: gauss_reduce.
//
// Box-of-binomials blur + decimate by 2: writes the coarse-resolution
// blurred output, one output pixel per workitem. Input is sampled at
// (2*cx+ii, 2*cy+jj) with ii,jj ∈ [-2,2] — all in-bounds by
// construction since cx,cy are clamped to [1, wd-2] and the fine
// buffer is sized 2*wd-1 or 2*wd (downsample-by-2 with the ceiling
// `dl()` formula).
//
// Binding layout (2 storage buffers):
//   0: in     (float)   — fine input buffer
//   1: coarse (float)   — coarse output buffer
// Push constants: 4 ints = 16 bytes (wd, ht, fine_w, _pad)
//   `fine_w` is the fine buffer's row stride needed to index into it
//   from this kernel which only knows the coarse dimensions.

#include "dt_vulkan_common.h"

kernel void ll_gauss_reduce(global const float *in,
                            global       float *coarse,
                            const int wd,
                            const int ht,
                            const int fine_w,
                            const int fine_h)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= wd || y >= ht) return;

  const int cx = clamp(x, 1, wd - 2);
  const int cy = clamp(y, 1, ht - 2);
  const float w[5] = { 1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f };

  float pixel = 0.0f;
  for(int jj = -2; jj <= 2; jj++)
    for(int ii = -2; ii <= 2; ii++)
    {
      const int fx = clamp(2 * cx + ii, 0, fine_w - 1);
      const int fy = clamp(2 * cy + jj, 0, fine_h - 1);
      pixel += in[fy * fine_w + fx] * w[ii + 2] * w[jj + 2];
    }
  coarse[y * wd + x] = pixel;
}
