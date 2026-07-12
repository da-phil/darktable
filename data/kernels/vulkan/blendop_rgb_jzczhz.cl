/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan compute twin of blendop.cl::blendop_rgb_jzczhz — the
    scene-referred RGB blend-apply kernel, all blend modes.

    Buffer-shape differences vs the OpenCL original (math unchanged):
    - image2d_t reads become flat float4 buffer loads; `in_a` is
      roi_in-sized so it carries its own row width `iwidth`, while
      `out` and `mask` are roi_out-sized (`width` pixels per row).
    - OpenCL 1.2 images can't be read_write, so the CL host copies
      dev_out to a dev_tmp and the kernel reads b from the copy.
      Storage buffers have no such limitation and the blend is purely
      per-pixel (each invocation reads and writes only index `idx`),
      so `out` is read (module result, "b") and overwritten in place —
      no tmp copy, no extra bandwidth.
*/

#include "dt_vulkan_blendop.h"

kernel void blendop_rgb_jzczhz(global const float4 *in_a,
                               global float4 *out,
                               global const float *mask,
                               const int width,
                               const int height,
                               const int iwidth,
                               const unsigned int blend_mode,
                               const float blend_parameter,
                               const int offx,
                               const int offy,
                               const int mask_display)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  const int idx = y * width + x;
  const int iidx = (y + offy) * iwidth + (x + offx);

  float4 a, b, o;

  if((blend_mode & DEVELOP_BLEND_REVERSE) == DEVELOP_BLEND_REVERSE)
  {
    b = in_a[iidx]; // see comment in blend.c:dt_develop_blend_process_cl()
    a = out[idx];
  }
  else
  {
    a = in_a[iidx];
    b = out[idx];
  }
  float opacity = mask[idx];
  float norm_a;
  float norm_b;

  /* select the blend operator */
  switch(blend_mode & DEVELOP_BLEND_MODE_MASK)
  {
    case DEVELOP_BLEND_MULTIPLY:
      o = a * (1.0f - opacity) + a * b * blend_parameter * opacity;
      break;

    case DEVELOP_BLEND_AVERAGE:
      o = a * (1.0f - opacity) + 0.5f * (a + b) * opacity;
      break;

    case DEVELOP_BLEND_ADD:
      o = a * (1.0f - opacity) + (a + blend_parameter * b) * opacity;
      break;

    case DEVELOP_BLEND_SUBTRACT:
      o = a * (1.0f - opacity) + fmax(a - blend_parameter * b, 0.0f) * opacity;
      break;

    case DEVELOP_BLEND_SUBTRACT_INVERSE:
      o = a * (1.0f - opacity) + fmax(b - blend_parameter * a, 0.0f) * opacity;
      break;

    case DEVELOP_BLEND_DIFFERENCE:
    case DEVELOP_BLEND_DIFFERENCE2:
      o = a * (1.0f - opacity) + fabs(a - b) * opacity;
      break;

    case DEVELOP_BLEND_LIGHTNESS:
      norm_a = fmax(sqrt(a.x * a.x + a.y * a.y + a.z * a.z), 1e-6f);
      norm_b = fmax(sqrt(b.x * b.x + b.y * b.y + b.z * b.z), 1e-6f);
      o = a * (1.0f - opacity) + a * norm_b / norm_a * opacity;
      break;

    case DEVELOP_BLEND_CHROMA:
      norm_a = fmax(sqrt(a.x * a.x + a.y * a.y + a.z * a.z), 1e-6f);
      norm_b = fmax(sqrt(b.x * b.x + b.y * b.y + b.z * b.z), 1e-6f);
      o = a * (1.0f - opacity) + b * norm_a / norm_b * opacity;
      break;

    case DEVELOP_BLEND_RGB_R:
      o.x = (a.x * (1.0f - opacity)) + (blend_parameter * b.x * opacity);
      o.y = a.y;
      o.z = a.z;
      o.w = a.w;
      break;

    case DEVELOP_BLEND_RGB_G:
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (blend_parameter * b.y * opacity);
      o.z = a.z;
      o.w = a.w;
      break;

    case DEVELOP_BLEND_RGB_B:
      o.x = a.x;
      o.y = a.y;
      o.z = (a.z * (1.0f - opacity)) + (blend_parameter * b.z * opacity);
      o.w = a.w;
      break;

    case DEVELOP_BLEND_DIVIDE:
      o = a * (1.0f - opacity) + a / fmax(b * blend_parameter, 1e-6f) * opacity;
      break;

    case DEVELOP_BLEND_DIVIDE_INVERSE:
      o = a * (1.0f - opacity) + b / fmax(a * blend_parameter, 1e-6f) * opacity;
      break;

    case DEVELOP_BLEND_GEOMETRIC_MEAN:
      o = a * (1.0f - opacity) + sqrt(fmax(a * b, 0.0f)) * opacity;
      break;

    case DEVELOP_BLEND_HARMONIC_MEAN:
      o = a * (1.0f - opacity) + 2.0f * a * b / (fmax(a, 5e-7f) + fmax(b, 5e-7f)) * opacity;
      break;

    /* fallback to normal blend */
    case DEVELOP_BLEND_BOUNDED:
    case DEVELOP_BLEND_NORMAL2:
    default:
      o = (a * (1.0f - opacity)) + (b * opacity);
      break;
  }

  /* we transfer alpha channel of input if mask_display is set, else we save opacity into alpha channel */
  o.w = mask_display ? a.w : opacity;

  out[idx] = o;
}
