/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::flip — coordinate-remap kernel. Input
    has roi_in dims (width x height); output has roi_out dims
    (owidth x oheight). For swap-XY (rotate 90°) those differ.
*/

#include "dt_vulkan_common.h"

kernel void flip(global const float4 *in,
                 global float4 *out,
                 const int width,
                 const int height,
                 const int owidth,
                 const int oheight,
                 const int orientation)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  // ORIENTATION_FLIP_X = 2
  int ox = (orientation & 2) ? width  - x - 1 : x;
  // ORIENTATION_FLIP_Y = 1
  int oy = (orientation & 1) ? height - y - 1 : y;
  // ORIENTATION_SWAP_XY = 4
  if(orientation & 4)
  {
    const int tmp = ox;
    ox = oy;
    oy = tmp;
  }

  if(ox >= 0 && oy >= 0 && ox < owidth && oy < oheight)
  {
    out[oy * owidth + ox] = in[y * width + x];
  }
}
