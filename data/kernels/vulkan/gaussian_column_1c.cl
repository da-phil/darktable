// Vulkan port of gaussian.cl :: gaussian_column_4c, single-channel variant.
//
// Deriche IIR forward+backward sweep along columns of a single-channel
// float buffer. Pairs with gaussian_row_1c for full 2-axis blur.

#include "dt_vulkan_common.h"

kernel void gaussian_column_1c(global const float *in,
                               global       float *out,
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
                               const float Labmax,
                               const float Labmin)
{
  const int x = get_global_id(0);
  if(x >= width) return;

  float xp = clamp(in[x], Labmin, Labmax);
  float yb = xp * coefp;
  float yp = yb;

  for(int y = 0; y < height; y++)
  {
    const int idx = y * width + x;
    const float xc = clamp(in[idx], Labmin, Labmax);
    const float yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc; yb = yp; yp = yc;
    out[idx] = yc;
  }

  float xn = clamp(in[(height - 1) * width + x], Labmin, Labmax);
  float xa = xn;
  float yn = xn * coefn;
  float ya = yn;

  for(int y = height - 1; y >= 0; y--)
  {
    const int idx = y * width + x;
    const float xc = clamp(in[idx], Labmin, Labmax);
    const float yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn; xn = xc;
    ya = yn; yn = yc;
    out[idx] += yc;
  }
}
