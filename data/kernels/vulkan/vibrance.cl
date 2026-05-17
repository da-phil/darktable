/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of extended.cl::vibrance — Lab-space saturation boost
    weighted by chroma magnitude.
*/

#include "dt_vulkan_common.h"

kernel void vibrance(global const float4 *in,
                     global float4 *out,
                     const int width,
                     const int height,
                     const float amount)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);
  float4 p = in[i];

  const float sw = sqrt(p.y * p.y + p.z * p.z) / 256.0f;
  const float ls = 1.0f - amount * sw * 0.25f;
  const float ss = 1.0f + amount * sw;

  float4 o;
  o.x = p.x * ls;
  o.y = p.y * ss;
  o.z = p.z * ss;
  o.w = p.w;
  out[i] = o;
}
