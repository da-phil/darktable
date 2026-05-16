/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan compute version of the exposure kernel.

    Operates on a flat RGBA float buffer (the pixelpipe stages the
    cl_mem/image2d_t input to a host-visible buffer before dispatch;
    see src/develop/pixelpipe_hb.c). Functionally identical to the
    `exposure` kernel in data/kernels/basic.cl.

    Compiled with clspv to SPIR-V at build time; loaded as
    "vulkan/exposure.spv" by src/iop/exposure.c.
*/

kernel void exposure(global const float4 *in,
                     global float4 *out,
                     const int width,
                     const int height,
                     const float black,
                     const float scale)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int idx = y * width + x;
  float4 pixel = in[idx];
  pixel.x = (pixel.x - black) * scale;
  pixel.y = (pixel.y - black) * scale;
  pixel.z = (pixel.z - black) * scale;
  out[idx] = pixel;
}
