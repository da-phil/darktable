/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of extended.cl::graduatedndp / graduatedndm — graduated
    neutral-density filter applied along a screen-space gradient.

    The OpenCL build ships two kernels: graduatedndp for positive
    density (divide path) and graduatedndm for negative density
    (multiply path). The host selects one or the other based on the
    sign of `density`. Here we collapse both into a single entry
    point and branch on the sign of `density` inside the kernel —
    the branch is uniform across the dispatch, so clspv and glslang
    both hoist it out cleanly, and we keep the entire module
    available under the single-entry-point glslang fallback.

    Bindings:
      0 = input  float4 buffer
      1 = output float4 buffer
    Push constants: width, height, density, length_base, length_inc_x,
                    length_inc_y, color[4].
*/

#include "dt_vulkan_common.h"

kernel void graduatednd(global const float4 *in,
                        global float4 *out,
                        const int width,
                        const int height,
                        const float density,
                        const float length_base,
                        const float length_inc_x,
                        const float length_inc_y,
                        const float color_r,
                        const float color_g,
                        const float color_b,
                        const float color_a)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);

  const float4 pixel = in[i];
  const float4 color = (float4)(color_r, color_g, color_b, color_a);
  const float len = length_base + y * length_inc_y + x * length_inc_x;

  float4 result;
  if(density > 0.0f)
  {
    const float dens = exp2(density * clipf(0.5f + len));
    result = pixel / (color + ((float4)1.0f - color) * (float4)dens);
  }
  else
  {
    const float dens = exp2(-density * clipf(0.5f - len));
    result = pixel * (color + ((float4)1.0f - color) * (float4)dens);
  }
  result.w = pixel.w;
  out[i] = result;
}
