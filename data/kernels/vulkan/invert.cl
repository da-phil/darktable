/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::invert_4f (the demosaiced/RGB variant).
    The 1f Bayer variant stays on OpenCL.
*/

#include "dt_vulkan_common.h"

kernel void invert_4f(global const float4 *in,
                      global float4 *out,
                      const int width,
                      const int height,
                      const float color_r,
                      const float color_g,
                      const float color_b)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);
  float4 pixel = in[i];
  pixel.x = clipf(color_r - pixel.x);
  pixel.y = clipf(color_g - pixel.y);
  pixel.z = clipf(color_b - pixel.z);
  out[i] = pixel;
}
