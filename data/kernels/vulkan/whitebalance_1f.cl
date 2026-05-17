/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::whitebalance_1f — pre-demosaic Bayer
    white balance. Single-channel float input/output; the kernel
    multiplies each pixel by one of four coefficients selected by the
    Bayer pattern at that (row, col).

    Bindings:
      0 = input float buffer  (1 float per pixel, sized roi_in)
      1 = output float buffer (1 float per pixel, sized roi_out)
    Push constants: width, height, filters, coeffs[4].
*/

#include "dt_vulkan_common.h"

kernel void whitebalance_1f(global const float *in,
                            global float *out,
                            const int width,
                            const int height,
                            const uint filters,
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
  const int fc = vk_FC(y, x, filters);
  // 4 coefficients indexed 0..3; emit a single multiply per pixel
  // without a branchy switch — clspv and glslang both vectorise this
  // into a select.
  float coeff = c0;
  if(fc == 1) coeff = c1;
  else if(fc == 2) coeff = c2;
  else if(fc == 3) coeff = c3;
  out[idx] = pixel * coeff;
}
