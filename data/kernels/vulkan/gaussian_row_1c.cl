// Vulkan port of gaussian.cl :: gaussian_row_4c, single-channel variant.
//
// Deriche IIR forward+backward sweep along rows of a single-channel
// float buffer. Used by colorequal's L-scharr / saturation passes
// (and any future 1-channel Gaussian consumer).
//
// Bindings: 0=in float, 1=out float.
// Push constants: 48 B
//   (width, height, a0..a3, b1, b2, coefp, coefn, Labmax, Labmin).

#include "dt_vulkan_common.h"

kernel void gaussian_row_1c(global const float *in,
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
  const int y = get_global_id(0);
  if(y >= height) return;
  const int row = y * width;

  float xp = clamp(in[row + 0], Labmin, Labmax);
  float yb = xp * coefp;
  float yp = yb;

  for(int x = 0; x < width; x++)
  {
    const int idx = row + x;
    const float xc = clamp(in[idx], Labmin, Labmax);
    const float yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc; yb = yp; yp = yc;
    out[idx] = yc;
  }

  float xn = clamp(in[row + (width - 1)], Labmin, Labmax);
  float xa = xn;
  float yn = xn * coefn;
  float ya = yn;

  for(int x = width - 1; x >= 0; x--)
  {
    const int idx = row + x;
    const float xc = clamp(in[idx], Labmin, Labmax);
    const float yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn; xn = xc;
    ya = yn; yn = yc;
    out[idx] += yc;
  }
}
