/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of gaussian.cl's recursive Deriche IIR blur.

    The OpenCL build runs column blur + transpose + column blur +
    transpose to keep both passes cache-friendly. The transpose
    kernel needs dynamically-sized workgroup-local memory, which
    clspv/glslang both handle awkwardly — and the speed-up versus
    a simpler row+column scheme isn't compelling at the buffer
    sizes the Vulkan path runs at today.

    This port therefore drops the transpose and uses one kernel per
    axis: gaussian_row_4c does the forward+backward sweep along
    rows, gaussian_column_4c does the same along columns. The host
    side dispatches them in sequence with a temp buffer between.

    Each work-item owns one row (gaussian_row_4c) or one column
    (gaussian_column_4c) and walks it sequentially — the IIR
    recurrence is inherently serial along the sweep axis, so this
    is the natural parallelism granularity.

    Bindings:
      0 = input  float4 buffer
      1 = output float4 buffer
    Push constants: width, height, a0..a3, b1, b2, coefp, coefn,
                    Labmax[4], Labmin[4] = 2 ints + 16 floats = 72 bytes.
*/

#include "dt_vulkan_common.h"

kernel void gaussian_row_4c(global const float4 *in,
                            global float4 *out,
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
                            const float Labmax_r,
                            const float Labmax_g,
                            const float Labmax_b,
                            const float Labmax_a,
                            const float Labmin_r,
                            const float Labmin_g,
                            const float Labmin_b,
                            const float Labmin_a)
{
  const int y = get_global_id(0);
  if(y >= height) return;

  const float4 Labmax = (float4)(Labmax_r, Labmax_g, Labmax_b, Labmax_a);
  const float4 Labmin = (float4)(Labmin_r, Labmin_g, Labmin_b, Labmin_a);
  const int row = y * width;

  // forward filter
  float4 xp = clamp(in[row + 0], Labmin, Labmax);
  float4 yb = xp * coefp;
  float4 yp = yb;

  for(int x = 0; x < width; x++)
  {
    const int idx = row + x;
    const float4 xc = clamp(in[idx], Labmin, Labmax);
    const float4 yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc; yb = yp; yp = yc;
    out[idx] = yc;
  }

  // backward filter
  float4 xn = clamp(in[row + (width - 1)], Labmin, Labmax);
  float4 xa = xn;
  float4 yn = xn * coefn;
  float4 ya = yn;

  for(int x = width - 1; x >= 0; x--)
  {
    const int idx = row + x;
    const float4 xc = clamp(in[idx], Labmin, Labmax);
    const float4 yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn; xn = xc;
    ya = yn; yn = yc;
    out[idx] += yc;
  }
}

kernel void gaussian_column_4c(global const float4 *in,
                               global float4 *out,
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
                               const float Labmax_r,
                               const float Labmax_g,
                               const float Labmax_b,
                               const float Labmax_a,
                               const float Labmin_r,
                               const float Labmin_g,
                               const float Labmin_b,
                               const float Labmin_a)
{
  const int x = get_global_id(0);
  if(x >= width) return;

  const float4 Labmax = (float4)(Labmax_r, Labmax_g, Labmax_b, Labmax_a);
  const float4 Labmin = (float4)(Labmin_r, Labmin_g, Labmin_b, Labmin_a);

  // forward filter
  float4 xp = clamp(in[x], Labmin, Labmax);
  float4 yb = xp * coefp;
  float4 yp = yb;

  for(int y = 0; y < height; y++)
  {
    const int idx = y * width + x;
    const float4 xc = clamp(in[idx], Labmin, Labmax);
    const float4 yc = (a0 * xc) + (a1 * xp) - (b1 * yp) - (b2 * yb);
    xp = xc; yb = yp; yp = yc;
    out[idx] = yc;
  }

  // backward filter
  float4 xn = clamp(in[(height - 1) * width + x], Labmin, Labmax);
  float4 xa = xn;
  float4 yn = xn * coefn;
  float4 ya = yn;

  for(int y = height - 1; y >= 0; y--)
  {
    const int idx = y * width + x;
    const float4 xc = clamp(in[idx], Labmin, Labmax);
    const float4 yc = (a2 * xn) + (a3 * xa) - (b1 * yn) - (b2 * ya);
    xa = xn; xn = xc;
    ya = yn; yn = yc;
    out[idx] += yc;
  }
}
