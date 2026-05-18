/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::borders_fill plus a sub-region
    buffer-to-buffer copy used by the borders module to slot the
    original input image into the framed canvas.

    The OpenCL build uses image2d_t and dispatches the fill kernel
    over the full canvas, returning early for pixels outside the
    target rectangle. Buffers don't get the "early return outside
    region" trick for free — we'd be wasting a work-item per pixel
    in the unframed area — so the Vulkan kernels are dispatched
    exactly over the target rectangle and the kernel just translates
    (lid_x, lid_y) into a write at (dst_x + lid_x, dst_y + lid_y).

    Bindings (borders_fill):
      0 = output float4 buffer
    Push constants: out_width, out_height, dst_x, dst_y, region_w,
                    region_h, color[4] = 6 ints + 4 floats = 40 bytes.

    Bindings (borders_copy):
      0 = input  float4 buffer (sized in_width * in_height)
      1 = output float4 buffer (sized out_width * out_height)
    Push constants: in_width, out_width, dst_x, dst_y, region_w,
                    region_h = 6 ints = 24 bytes.
*/

#include "dt_vulkan_common.h"

kernel void borders_fill(global float4 *out,
                         const int out_width,
                         const int out_height,
                         const int dst_x,
                         const int dst_y,
                         const int region_w,
                         const int region_h,
                         const float color_r,
                         const float color_g,
                         const float color_b,
                         const float color_a)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= region_w || y >= region_h) return;
  const int ox = dst_x + x;
  const int oy = dst_y + y;
  if(ox >= out_width || oy >= out_height) return;
  out[oy * out_width + ox] = (float4)(color_r, color_g, color_b, color_a);
}

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
