/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of basic.cl::colisa — contrast / brightness /
    saturation curves applied via two unbounded 65536-entry LUTs
    plus a flat saturation scale.

    Demonstrates the "module-uploaded LUT" pattern: small lookup
    tables live in storage buffers bound on dispatch, the linear-
    extrapolation coefficients ride along in the push-constant
    block, and the kernel uses vk_lookup_unbounded from
    dt_vulkan_common.h to mirror the OpenCL colour_conversion.h
    lookup_unbounded helper byte-for-byte.

    Bindings:
      0 = input  float4 buffer (Lab data, .x is L*)
      1 = output float4 buffer
      2 = ctable LUT (65536 floats — contrast curve)
      3 = ltable LUT (65536 floats — lightness/brightness curve)
    Push constants: width, height, saturation, ca[3], la[3]
                    = 2 ints + 7 floats = 36 bytes.
*/

#include "dt_vulkan_common.h"

kernel void colisa(global const float4 *in,
                   global float4 *out,
                   global const float *ctable,
                   global const float *ltable,
                   const int width,
                   const int height,
                   const float saturation,
                   const float ca0,
                   const float ca1,
                   const float ca2,
                   const float la0,
                   const float la1,
                   const float la2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);

  const float4 p = in[i];
  float4 o;
  // Two cascaded unbounded curve lookups on L (Lab), saturation
  // scale on a/b. Matches basic.cl::colisa exactly.
  o.x = vk_lookup_unbounded(ctable, p.x / 100.0f, ca0, ca1, ca2);
  o.x = vk_lookup_unbounded(ltable, o.x / 100.0f, la0, la1, la2);
  o.y = p.y * saturation;
  o.z = p.z * saturation;
  o.w = p.w;
  out[i] = o;
}
