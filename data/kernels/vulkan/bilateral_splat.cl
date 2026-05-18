// Vulkan port of bilateral.cl :: kernel splat.
//
// One work-item per image pixel. Computes the pixel's position in
// the 3-D bilateral grid (x_grid, y_grid, L_grid via sigma_s and
// sigma_r) and trilinearly distributes its contribution across 8
// surrounding grid cells via vk_atomic_add_f.
//
// The OpenCL version uses workgroup-local-memory to reduce atomic
// contention before the global atomic_add. We start with a simpler
// direct-atomic-add form; the local-memory reduction is a follow-
// up perf pass (§5.13 in the dev-doc — "scope-controlled scaffold
// before optimisation").
//
// Binding layout (2 storage buffers):
//   0: in   (float4)  — input Lab pixels
//   1: grid (float)   — bilateral 3-D grid as flat array
// Push constants: 5 ints + 2 floats = 28 bytes.

#include "dt_vulkan_common.h"

kernel void bilateral_splat(global const float4 *in,
                            global       float  *grid,
                            const int   width,
                            const int   height,
                            const int   size_x,
                            const int   size_y,
                            const int   size_z,
                            const float sigma_s,
                            const float sigma_r)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const float4 pixel = in[y * width + x];
  const float L = pixel.x;

  // Project (x, y, L) into grid space and clip to [0, size-1].
  const float gx = clamp((float)x / sigma_s, 0.0f, (float)(size_x - 1));
  const float gy = clamp((float)y / sigma_s, 0.0f, (float)(size_y - 1));
  const float gz = clamp(L      / sigma_r,    0.0f, (float)(size_z - 1));

  // Integer corner of the 2x2x2 cube; clamp to size-2 so the +1
  // neighbours always exist.
  const int xi = min(size_x - 2, (int)gx);
  const int yi = min(size_y - 2, (int)gy);
  const int zi = min(size_z - 2, (int)gz);
  const float fx = gx - (float)xi;
  const float fy = gy - (float)yi;
  const float fz = gz - (float)zi;

  const int ox = 1;
  const int oy = size_x;
  const int oz = size_y * size_x;
  const int gi0 = xi + oy * yi + oz * zi;

  const float contrib = 100.0f / (sigma_s * sigma_s);
  vk_atomic_add_f(grid + gi0,             contrib * (1.0f - fx) * (1.0f - fy) * (1.0f - fz));
  vk_atomic_add_f(grid + gi0 + ox,        contrib * (       fx) * (1.0f - fy) * (1.0f - fz));
  vk_atomic_add_f(grid + gi0 + oy,        contrib * (1.0f - fx) * (       fy) * (1.0f - fz));
  vk_atomic_add_f(grid + gi0 + oy + ox,   contrib * (       fx) * (       fy) * (1.0f - fz));
  vk_atomic_add_f(grid + gi0 + oz,        contrib * (1.0f - fx) * (1.0f - fy) * (       fz));
  vk_atomic_add_f(grid + gi0 + oz + ox,   contrib * (       fx) * (1.0f - fy) * (       fz));
  vk_atomic_add_f(grid + gi0 + oz + oy,   contrib * (1.0f - fx) * (       fy) * (       fz));
  vk_atomic_add_f(grid + gi0 + oz + oy + ox, contrib * (    fx) * (       fy) * (       fz));
}
