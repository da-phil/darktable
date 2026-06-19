/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Standalone gaussian_row_4c kernel — a GLSL-fallback companion
    program to gaussian.cl. clspv builds emit both gaussian_row_4c and
    gaussian_column_4c into gaussian.spv; glslang's one-entry-per-module
    limit drops everything but the primary entry there. Without this
    fallback program the row blur is unreachable on glslang builds,
    which breaks the 4-channel Gaussian helper and every consumer of it
    (colorequal, blurs, censorize…).

    Body is byte-identical to the gaussian_row_4c entry in gaussian.cl
    so clspv compilation stays consistent — see the host fallback
    chain in src/common/gaussian.c.

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
