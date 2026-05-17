/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan/clspv port of basic.cl::velvia — line-for-line equivalent
    to the OpenCL kernel; any drift is a bug.
*/

#include "dt_vulkan_common.h"

kernel void velvia(global const float4 *in,
                   global float4 *out,
                   const int width,
                   const int height,
                   const float strength,
                   const float bias)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);
  float4 pixel = in[i];

  const float pmax = fmax(pixel.x, fmax(pixel.y, pixel.z));
  const float pmin = fmin(pixel.x, fmin(pixel.y, pixel.z));
  const float plum = (pmax + pmin) * 0.5f;
  const float psat = (plum <= 0.5f)
                       ? (pmax - pmin) / (1e-5f + pmax + pmin)
                       : (pmax - pmin) / (1e-5f + fmax(0.0f, 2.0f - pmax - pmin));
  const float pweight = clipf(((1.0f - 1.5f * psat)
                            + ((1.0f + fabs(plum - 0.5f) * 2.0f) * (1.0f - bias)))
                            / (1.0f + (1.0f - bias)));
  const float sat = strength * pweight;

  float4 o;
  o.x = clipf(pixel.x + sat * (pixel.x - 0.5f * (pixel.y + pixel.z)));
  o.y = clipf(pixel.y + sat * (pixel.y - 0.5f * (pixel.z + pixel.x)));
  o.z = clipf(pixel.z + sat * (pixel.z - 0.5f * (pixel.x + pixel.y)));
  o.w = pixel.w;
  out[i] = o;
}
