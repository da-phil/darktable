/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of extended.cl::splittoning — hue/saturation injection
    into shadow and highlight regions of an RGB image, via HSL
    intermediate space. Uses the RGB↔HSL helpers in
    dt_vulkan_common.h.
*/

#include "dt_vulkan_common.h"

kernel void splittoning(global const float4 *in,
                        global float4 *out,
                        const int width,
                        const int height,
                        const float compress,
                        const float balance,
                        const float shadow_hue,
                        const float shadow_saturation,
                        const float highlight_hue,
                        const float highlight_saturation)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  float4 pixel = in[idx];
  float4 hsl = vk_RGB_to_HSL(pixel);

  if(hsl.z < balance - compress || hsl.z > balance + compress)
  {
    hsl.x = (hsl.z < balance) ? shadow_hue        : highlight_hue;
    hsl.y = (hsl.z < balance) ? shadow_saturation : highlight_saturation;
    const float ra = (hsl.z < balance)
                       ? clipf(2.0f * fabs(-balance + compress + hsl.z))
                       : clipf(2.0f * fabs(-balance - compress + hsl.z));

    float4 mixrgb = vk_HSL_to_RGB(hsl);
    pixel.xyz = clamp(pixel * (1.0f - ra) + mixrgb * ra, (float4)0.0f, (float4)1.0f).xyz;
  }

  out[idx] = pixel;
}
