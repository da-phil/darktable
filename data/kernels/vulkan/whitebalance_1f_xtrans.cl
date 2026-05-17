/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::whitebalance_1f_xtrans — pre-demosaic
    X-Trans white balance. Same shape as the Bayer 1f path but the
    filter pattern is a 6x6 byte matrix rather than the packed
    `filters` bitmask; we pass it as a flat 36-element uint buffer.

    Bindings:
      0 = input float buffer
      1 = output float buffer
      2 = xtrans 6x6 pattern as 36 uints (one byte per slot)
    Push constants: width, height, coeffs[4].
*/

#include "dt_vulkan_common.h"

kernel void whitebalance_1f_xtrans(global const float *in,
                                   global float *out,
                                   global const uint *xtrans_flat,
                                   const int width,
                                   const int height,
                                   const float c0,
                                   const float c1,
                                   const float c2,
                                   const float c3)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;
  const float pixel = in[idx];
  const int fc = vk_FCxtrans(y, x, xtrans_flat);
  float coeff = c0;
  if(fc == 1) coeff = c1;
  else if(fc == 2) coeff = c2;
  else if(fc == 3) coeff = c3;
  out[idx] = pixel * coeff;
}
