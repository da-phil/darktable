/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of extended.cl::relight — Gaussian-windowed exposure
    boost / cut centred at a chosen lightness value.

    Bindings:
      0 = input  float4 buffer (Lab data, .x is L*)
      1 = output float4 buffer
    Push constants: width, height, center, wings, ev (2 ints + 3
                    floats = 20 bytes).
*/

#include "dt_vulkan_common.h"

static inline float vk_gauss(const float center, const float wings, const float x)
{
  const float b = -1.0f + center * 2.0f;
  const float c = (wings / 10.0f) / 2.0f;
  return exp(-(x - b) * (x - b) / (c * c));
}

kernel void relight(global const float4 *in,
                    global float4 *out,
                    const int width,
                    const int height,
                    const float center,
                    const float wings,
                    const float ev)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);

  float4 pixel = in[i];

  const float lightness = pixel.x / 100.0f;
  const float value = -1.0f + (lightness * 2.0f);
  float gauss = vk_gauss(center, wings, value);
  if(isnan(gauss) || isinf(gauss)) gauss = 0.0f;

  float relight = 1.0f / exp2(-ev * clipf(gauss));
  if(isnan(relight) || isinf(relight)) relight = 1.0f;

  pixel.x = 100.0f * clipf(lightness * relight);
  out[i] = pixel;
}
