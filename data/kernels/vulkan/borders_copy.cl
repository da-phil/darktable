/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Standalone borders_copy kernel used as a GLSL-fallback companion
    program to borders.cl. clspv builds emit borders_copy as a
    second entry point inside borders.spv (the primary program);
    glslang's one-entry-per-module limit drops everything but the
    primary entry there. Shipping this as a separate program gives
    glslang-only builds full coverage of the borders module instead
    of forcing a CL/CPU fallback for the inner image copy.

    Bindings (matches the second-entry layout in borders.cl):
      0 = input  float4 buffer (sized in_width * in_height)
      1 = output float4 buffer (sized out_width * out_height)
    Push constants: in_width, out_width, dst_x, dst_y, region_w,
                    region_h = 6 ints = 24 bytes.
*/

#include "dt_vulkan_common.h"

kernel void borders_copy(global const float4 *in,
                         global float4 *out,
                         const int in_width,
                         const int out_width,
                         const int dst_x,
                         const int dst_y,
                         const int region_w,
                         const int region_h)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= region_w || y >= region_h) return;
  out[(dst_y + y) * out_width + (dst_x + x)] = in[y * in_width + x];
}
