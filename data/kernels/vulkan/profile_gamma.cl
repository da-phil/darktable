/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::profilegamma (the gamma-curve variant).
    Uses a 65536-entry float LUT for the in-gamut range plus 3
    unbounded-extrapolation coefficients, all bound at binding 2.

    Push constants carry (w, h) plus the 3 unbounded coefficients.
*/

#include "dt_vulkan_common.h"

kernel void profilegamma(global const float4 *in,
                         global float4 *out,
                         global const float *lut,
                         const int width,
                         const int height,
                         const float ta0,
                         const float ta1,
                         const float ta2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  const float4 i = in[idx];

  float4 o;
  o.x = vk_lookup_unbounded(lut, i.x, ta0, ta1, ta2);
  o.y = vk_lookup_unbounded(lut, i.y, ta0, ta1, ta2);
  o.z = vk_lookup_unbounded(lut, i.z, ta0, ta1, ta2);
  o.w = i.w;
  out[idx] = o;
}
