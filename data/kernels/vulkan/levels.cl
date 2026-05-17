/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::levels — Lab L-channel level remapping
    via a 65536-entry LUT in the [low, high] range, with gamma-
    extrapolated tails. Chroma scales proportionally to luminance
    change.

    Bindings:
      0 = input float4 buffer (Lab)
      1 = output float4 buffer (Lab)
      2 = curve LUT (global float[65536])
    Push constants: (width, height, in_low, in_high, in_inv_gamma).
*/

#include "dt_vulkan_common.h"

kernel void levels(global const float4 *in,
                   global float4 *out,
                   global const float *lut,
                   const int width,
                   const int height,
                   const float in_low,
                   const float in_high,
                   const float in_inv_gamma)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  float4 pixel = in[idx];
  const float L = pixel.x;
  const float L_in = pixel.x / 100.0f;

  if(L_in <= in_low)
    pixel.x = 0.0f;
  else if(L_in >= in_high)
    pixel.x = 100.0f * pow((L_in - in_low) / (in_high - in_low), in_inv_gamma);
  else
    pixel.x = vk_lookup(lut, (L_in - in_low) / (in_high - in_low));

  if(L_in > 0.01f)
  {
    pixel.y *= pixel.x / L;
    pixel.z *= pixel.x / L;
  }
  else
  {
    pixel.y *= pixel.x;
    pixel.z *= pixel.x;
  }

  out[idx] = pixel;
}
