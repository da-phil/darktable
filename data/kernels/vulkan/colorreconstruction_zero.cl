// Vulkan port of colorreconstruction.cl :: colorreconstruction_zero.
//
// Zeros the bilateral grid. The OpenCL host treats the 3D grid as a
// flat 2D float buffer (width=4*size_x, height=size_y*size_z) and
// dispatches one item per cell; we do the same.
//
// Bindings (1 storage buffer):
//   0: grid (float; size = 4 * size_x * size_y * size_z)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void colorreconstruction_zero(global float *grid,
                                     const int width,
                                     const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  grid[y * width + x] = 0.0f;
}
