/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::profilegamma_log — the log-encoded
    "unbreak input profile" variant. Pure per-pixel math, no LUTs.
*/

#include "dt_vulkan_common.h"

kernel void profilegamma_log(global const float4 *in,
                             global float4 *out,
                             const int width,
                             const int height,
                             const float dynamic_range,
                             const float shadows_range,
                             const float grey)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  float4 i = in[idx];

  const float4 noise    = pow((float4)2.0f, (float4)-16.0f);
  const float4 dynamic4 = (float4)dynamic_range;
  const float4 shadows4 = (float4)shadows_range;
  const float4 grey4    = (float4)grey;

  float4 o = (i < noise) ? noise : i / grey4;
  o = (log2(o) - shadows4) / dynamic4;
  o = (o < noise) ? noise : o;
  i.xyz = o.xyz;
  out[idx] = i;
}
