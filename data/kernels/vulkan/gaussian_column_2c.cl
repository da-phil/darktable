// Vulkan port of gaussian.cl :: gaussian_column_4c, 2-channel variant.
//
// Deriche IIR forward+backward sweep along columns of a float2 buffer.
// Pairs with gaussian_row_2c for full 2-axis blur.

#include "dt_vulkan_common.h"

kernel void gaussian_column_2c(global const float2 *in,
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
  const int x = get_global_id(0);
  if(x >= width) return;

  const float2 Labmax = (float2)(Labmax_x, Labmax_y);
  const float2 Labmin = (float2)(Labmin_x, Labmin_y);

  float2 xp = clamp(in[x], Labmin, Labmax);
  float2 yb = xp * coefp;
  float2 yp = yb;

  for(int y = 0; y < height; y++)
  {
    const int idx = y * width + x;
    const float2 xc = clamp(in[idx], Labmin, Labmax);
    const float2 yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc; yb = yp; yp = yc;
    out[idx] = yc;
  }

  float2 xn = clamp(in[(height - 1) * width + x], Labmin, Labmax);
  float2 xa = xn;
  float2 yn = xn * coefn;
  float2 ya = yn;

  for(int y = height - 1; y >= 0; y--)
  {
    const int idx = y * width + x;
    const float2 xc = clamp(in[idx], Labmin, Labmax);
    const float2 yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn; xn = xc;
    ya = yn; yn = yc;
    out[idx] += yc;
  }
}
