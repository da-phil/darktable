/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Sub-region buffer-to-buffer copy used by the borders module to
    slot the original input image into the framed canvas. Lives in
    its own .cl (rather than alongside borders_fill in borders.cl)
    so the glslang fallback — one entry point per .spv — can ship it
    too: each toolchain compiles this file to borders_copy.spv with
    the single entry `borders_copy`.

    Dispatched exactly over the copied rectangle; the kernel
    translates (gid_x, gid_y) to a write at (dst_x + x, dst_y + y).

    Bindings:
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
