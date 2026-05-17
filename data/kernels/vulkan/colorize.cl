/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of extended.cl::colorize — replaces chrominance with
    constant a/b and rescales L.
*/

#include "dt_vulkan_common.h"

kernel void colorize(global const float4 *in,
                     global float4 *out,
                     const int width,
                     const int height,
                     const float mix,
                     const float L,
                     const float a,
                     const float b)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);
  float4 p = in[i];

  float4 o;
  o.x = p.x * mix + L - 50.0f * mix;
  o.y = a;
  o.z = b;
  o.w = p.w;
  out[i] = o;
}
