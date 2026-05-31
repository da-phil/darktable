// Vulkan port of nlmeans.cl :: nlmeans_horiz.
//
// Horizontal box sum (window radius P) of the distance buffer. The
// OpenCL kernel tiles via workgroup-local memory and clamps the wing
// reads to [0, width-1]; the VK twin reads straight from the global
// buffer with the same clamp-to-edge — bit-equal. (The `q` arg in the
// OpenCL signature is unused; dropped here.)
//
// Bindings (2 storage buffers):
//   0: U4_in  (float)
//   1: U4_out (float)
// Push constants: 12 B (width, height, P).

#include "dt_vulkan_common.h"

kernel void nlmeans_horiz(global const float *U4_in,
                          global       float *U4_out,
                          const int width,
                          const int height,
                          const int P)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  float distacc = 0.0f;
  for(int pi = -P; pi <= P; pi++)
  {
    const int xx = clamp(x + pi, 0, width - 1);
    distacc += U4_in[idx2d(xx, y, width)];
  }
  U4_out[idx2d(x, y, width)] = distacc;
}
