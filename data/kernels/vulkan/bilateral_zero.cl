// Vulkan port of bilateral.cl :: kernel zero — zeroes the
// scratch grid before splatting. Trivial buffer fill; we keep it
// as a kernel rather than vkCmdFillBuffer so the bilateral host
// helper stays toolchain-uniform.
//
// Binding: 0 = grid (float buffer of size_x * size_y * size_z)
// Push constants: width (size_x), height (size_y * size_z) = 8 bytes.

#include "dt_vulkan_common.h"

kernel void bilateral_zero(global float *grid,
                           const int width,
                           const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  grid[x + width * y] = 0.0f;
}
