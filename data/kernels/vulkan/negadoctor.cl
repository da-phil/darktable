/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of negadoctor.cl::negadoctor — analogue-film inversion
    and gamma in a single pass. Line-for-line equivalent to the
    OpenCL kernel.
*/

#include "dt_vulkan_common.h"

kernel void negadoctor(global const float4 *in,
                       global float4 *out,
                       const int width,
                       const int height,
                       const float Dmin_r, const float Dmin_g, const float Dmin_b,
                       const float wb_high_r, const float wb_high_g, const float wb_high_b,
                       const float offset_r, const float offset_g, const float offset_b,
                       const float exposure,
                       const float black,
                       const float gamma,
                       const float soft_clip,
                       const float soft_clip_comp)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  const float4 i = in[idx];

  const float4 Dmin    = (float4)(Dmin_r,    Dmin_g,    Dmin_b,    1.0f);
  const float4 wb_high = (float4)(wb_high_r, wb_high_g, wb_high_b, 1.0f);
  const float4 offset  = (float4)(offset_r,  offset_g,  offset_b,  0.0f);

  // Density in log space, threshold to -32 EV to avoid -inf.
  float4 o = -log10(Dmin / fmax(i, (float4)2.3283064365386963e-10f));
  o = wb_high * o + offset;

  // Print density on paper.
  o = -((float4)exposure * pow((float4)10.0f, o) + (float4)black);
  o = pow(fmax(o, (float4)0.0f), (float4)gamma);

  // Compress highlights / clip negatives.
  const float4 sc  = (float4)soft_clip;
  const float4 scc = (float4)soft_clip_comp;
  const int4 hi = isgreater(o, sc);
  const float4 hi_v = sc + ((float4)1.0f - exp(-(o - sc) / scc)) * scc;
  o = select(o, hi_v, hi);

  o.w = i.w;
  out[idx] = o;
}
