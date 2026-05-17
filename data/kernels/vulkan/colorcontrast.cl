/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of extended.cl::colorcontrast — per-channel scale +
    offset on Lab inputs, optionally clamped to ±128.
*/

#include "dt_vulkan_common.h"

kernel void colorcontrast(global const float4 *in,
                          global float4 *out,
                          const int width,
                          const int height,
                          const float scale_l,
                          const float scale_a,
                          const float scale_b,
                          const float offset_l,
                          const float offset_a,
                          const float offset_b,
                          const int unbound)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);
  float4 p = in[i];

  float4 o;
  o.x = p.x * scale_l + offset_l;
  o.y = p.y * scale_a + offset_a;
  o.z = p.z * scale_b + offset_b;
  o.w = p.w;

  if(!unbound)
  {
    o.y = clamp(o.y, -128.0f, 128.0f);
    o.z = clamp(o.z, -128.0f, 128.0f);
  }
  out[i] = o;
}
