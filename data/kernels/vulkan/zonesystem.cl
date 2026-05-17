/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::zonesystem — Lab zone-system tonal
    remapping. Each pixel's L channel is bucketed into one of `size`
    zones; the per-zone offset / scale tables are passed as small
    storage buffers (24 floats each — they're too big for the
    128-byte push-constant limit alongside scalars).

    Bindings:
      0 = input float4 buffer (Lab)
      1 = output float4 buffer (Lab)
      2 = zonemap_offset (global float[size])
      3 = zonemap_scale  (global float[size])
    Push constants: (width, height, size).
*/

#include "dt_vulkan_common.h"

kernel void zonesystem(global const float4 *in,
                       global float4 *out,
                       global const float *zonemap_offset,
                       global const float *zonemap_scale,
                       const int width,
                       const int height,
                       const int size)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  float4 pixel = in[idx];

  const float rzscale = (float)(size - 1) / 100.0f;
  const int rz = clamp((int)(pixel.x * rzscale), 0, size - 2);
  const float zs = ((rz > 0) ? (zonemap_offset[rz] / pixel.x) : 0.0f)
                   + zonemap_scale[rz];

  pixel.xyz *= zs;
  out[idx] = pixel;
}
