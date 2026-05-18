// Vulkan port of bilateral.cl :: kernel slice.
//
// Trilinearly samples the bilateral grid at each pixel's
// (x, y, L_in) position, scales the sampled delta by
// `detail * sigma_r * -0.04`, and stores `L_in + norm * Ldiff`
// back into the output buffer's L channel.
//
// Binding layout (3 storage buffers):
//   0: in   (float4)  — input Lab pixels (used for the L lookup)
//   1: out  (float4)  — output Lab pixels (.x replaced; .yzw passthrough)
//   2: grid (float)
// Push constants: 5 ints + 3 floats = 32 bytes.

#include "dt_vulkan_common.h"

kernel void bilateral_slice(global const float4 *in,
                            global       float4 *out,
                            global const float  *grid,
                            const int   width,
                            const int   height,
                            const int   size_x,
                            const int   size_y,
                            const int   size_z,
                            const float sigma_s,
                            const float sigma_r,
                            const float detail)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float norm = -detail * sigma_r * 0.04f;
  const int ox = 1;
  const int oy = size_x;
  const int oz = size_y * size_x;

  float4 pixel = in[idx];
  const float L = pixel.x;

  const float gx = clamp((float)x / sigma_s, 0.0f, (float)(size_x - 1));
  const float gy = clamp((float)y / sigma_s, 0.0f, (float)(size_y - 1));
  const float gz = clamp(L         / sigma_r, 0.0f, (float)(size_z - 1));

  const int xi = min(size_x - 2, (int)gx);
  const int yi = min(size_y - 2, (int)gy);
  const int zi = min(size_z - 2, (int)gz);
  const float fx = gx - (float)xi;
  const float fy = gy - (float)yi;
  const float fz = gz - (float)zi;
  const int gi = xi + oy * yi + oz * zi;

  const float Ldiff =
      grid[gi]                * (1.0f - fx) * (1.0f - fy) * (1.0f - fz)
    + grid[gi + ox]           * (       fx) * (1.0f - fy) * (1.0f - fz)
    + grid[gi + oy]           * (1.0f - fx) * (       fy) * (1.0f - fz)
    + grid[gi + ox + oy]      * (       fx) * (       fy) * (1.0f - fz)
    + grid[gi + oz]           * (1.0f - fx) * (1.0f - fy) * (       fz)
    + grid[gi + ox + oz]      * (       fx) * (1.0f - fy) * (       fz)
    + grid[gi + oy + oz]      * (1.0f - fx) * (       fy) * (       fz)
    + grid[gi + ox + oy + oz] * (       fx) * (       fy) * (       fz);

  pixel.x = fmax(0.0f, L + norm * Ldiff);
  out[idx] = pixel;
}
