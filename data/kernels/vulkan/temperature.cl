/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::whitebalance_4f — the demosaiced
    (filters==0) path of the temperature/white-balance module. The 1f
    Bayer / xtrans paths operate on RAW pre-demosaic data and stay
    on OpenCL/CPU; they don't fit the float4-buffer convention.
*/

#include "dt_vulkan_common.h"

kernel void whitebalance_4f(global const float4 *in,
                            global float4 *out,
                            const int width,
                            const int height,
                            const float c0,
                            const float c1,
                            const float c2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);
  const float4 p = in[i];
  out[i] = (float4)(p.x * c0, p.y * c1, p.z * c2, p.w);
}
