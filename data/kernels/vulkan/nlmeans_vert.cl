// Vulkan port of nlmeans.cl :: nlmeans_vert.
//
// Vertical box sum (window radius P) of the distance buffer, then
// convert the accumulated patch distance to a weight via the
// fast 2^-x approximation gh(d) = vk_fast_mexp2f(d * sharpness).
// Same clamp-to-edge as nlmeans_horiz.
//
// Bindings (2 storage buffers):
//   0: U4_in  (float)
//   1: U4_out (float, the patch weight)
// Push constants: 16 B (width, height, P, sharpness).

#include "dt_vulkan_common.h"

kernel void nlmeans_vert(global const float *U4_in,
                         global       float *U4_out,
                         const int   width,
                         const int   height,
                         const int   P,
                         const float sharpness)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float distacc = 0.0f;
  for(int pj = -P; pj <= P; pj++)
  {
    const int yy = clamp(y + pj, 0, height - 1);
    distacc += U4_in[idx2d(x, yy, width)];
  }
  distacc = vk_fast_mexp2f(distacc * sharpness);
  U4_out[idx2d(x, y, width)] = distacc;
}
