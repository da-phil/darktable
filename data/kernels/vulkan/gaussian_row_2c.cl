// Vulkan port of gaussian.cl :: gaussian_row_4c, 2-channel variant.
//
// Deriche IIR forward+backward sweep along rows of a float2 buffer.
// Used by colorequal's UV chromaticity passes (and any future
// 2-channel Gaussian consumer).
//
// Bindings: 0=in float2, 1=out float2.
// Push constants: 56 B
//   (width, height, a0..a3, b1, b2, coefp, coefn,
//    Labmax_x, Labmax_y, Labmin_x, Labmin_y).

#include "dt_vulkan_common.h"

kernel void gaussian_row_2c(global const float2 *in,
                            global       float2 *out,
                            const int width,
                            const int height,
                            const float a0,
                            const float a1,
                            const float a2,
                            const float a3,
                            const float b1,
                            const float b2,
                            const float coefp,
                            const float coefn,
                            const float Labmax_x,
                            const float Labmax_y,
                            const float Labmin_x,
                            const float Labmin_y)
{
  const int y = get_global_id(0);
  if(y >= height) return;
  const int row = y * width;

  const float2 Labmax = (float2)(Labmax_x, Labmax_y);
  const float2 Labmin = (float2)(Labmin_x, Labmin_y);

  float2 xp = clamp(in[row + 0], Labmin, Labmax);
  float2 yb = xp * coefp;
  float2 yp = yb;

  for(int x = 0; x < width; x++)
  {
    const int idx = row + x;
    const float2 xc = clamp(in[idx], Labmin, Labmax);
    const float2 yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc; yb = yp; yp = yc;
    out[idx] = yc;
  }

  float2 xn = clamp(in[row + (width - 1)], Labmin, Labmax);
  float2 xa = xn;
  float2 yn = xn * coefn;
  float2 ya = yn;

  for(int x = width - 1; x >= 0; x--)
  {
    const int idx = row + x;
    const float2 xc = clamp(in[idx], Labmin, Labmax);
    const float2 yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn; xn = xc;
    ya = yn; yn = yc;
    out[idx] += yc;
  }
}
