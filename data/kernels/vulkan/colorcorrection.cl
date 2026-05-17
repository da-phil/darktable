/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::colorcorrection — Lab-space split-toning
    style adjustment that shifts a/b axes proportionally to L.
*/

#include "dt_vulkan_common.h"

kernel void colorcorrection(global const float4 *in,
                            global float4 *out,
                            const int width,
                            const int height,
                            const float saturation,
                            const float a_scale,
                            const float a_base,
                            const float b_scale,
                            const float b_base)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);
  float4 p = in[i];
  float4 o;
  o.x = p.x;
  o.y = saturation * (p.y + p.x * a_scale + a_base);
  o.z = saturation * (p.z + p.x * b_scale + b_base);
  o.w = p.w;
  out[i] = o;
}
