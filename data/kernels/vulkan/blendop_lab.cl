/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan compute twin of blendop.cl::blendop_Lab — the Lab
    blend-apply kernel, all blend modes.

    Buffer-shape differences vs the OpenCL original (math unchanged):
    see the header comment in blendop_rgb_jzczhz.cl — same in-place
    `out` convention, same iwidth/offx/offy indexing for `in_a`.

    The vector ternaries (e.g. `a > b ? a : b`) are OpenCL C
    component-wise selects, exactly as in the original.
*/

#include "dt_vulkan_common.h"
#include "dt_vulkan_blendop.h"

kernel void blendop_lab(global const float4 *in_a,
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
  float4 ta, tb, to;
  float d, s;

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

  /* scale L down to [0; 1] and a,b to [-1; 1] */
  const float4 scale = (float4)(100.0f, 128.0f, 128.0f, 1.0f);
  a /= scale;
  b /= scale;

  const float4 min = (float4)(0.0f, -1.0f, -1.0f, 0.0f);
  const float4 max = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
  const float4 lmin = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
  const float4 lmax = (float4)(1.0f, 2.0f, 2.0f, 1.0f);       /* max + fabs(min) */
  const float4 halfmax = (float4)(0.5f, 1.0f, 1.0f, 0.5f);    /* lmax / 2.0f */
  const float4 doublemax = (float4)(2.0f, 4.0f, 4.0f, 2.0f);  /* lmax * 2.0f */
  const float opacity2 = opacity * opacity;

  float4 la = clamp(a + fabs(min), lmin, lmax);
  float4 lb = clamp(b + fabs(min), lmin, lmax);

  /* select the blend operator */
  switch(blend_mode & DEVELOP_BLEND_MODE_MASK)
  {
    case DEVELOP_BLEND_LIGHTEN:
      o = clamp(a * (1.0f - opacity) + (a > b ? a : b) * opacity, min, max);
      o.y = clamp(a.y * (1.0f - fabs(o.x - a.x)) + 0.5f * (a.y + b.y) * fabs(o.x - a.x), min.y, max.y);
      o.z = clamp(a.z * (1.0f - fabs(o.x - a.x)) + 0.5f * (a.z + b.z) * fabs(o.x - a.x), min.z, max.z);
      break;

    case DEVELOP_BLEND_DARKEN:
      o = clamp(a * (1.0f - opacity) + (a < b ? a : b) * opacity, min, max);
      o.y = clamp(a.y * (1.0f - fabs(o.x - a.x)) + 0.5f * (a.y + b.y) * fabs(o.x - a.x), min.y, max.y);
      o.z = clamp(a.z * (1.0f - fabs(o.x - a.x)) + 0.5f * (a.z + b.z) * fabs(o.x - a.x), min.z, max.z);
      break;

    case DEVELOP_BLEND_MULTIPLY:
      o = clamp(a * (1.0f - opacity) + a * b * opacity, min, max);
      if(a.x > 0.01f)
      {
        o.y = clamp(a.y * (1.0f - opacity) + (a.y + b.y) * o.x / a.x * opacity, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity) + (a.z + b.z) * o.x / a.x * opacity, min.z, max.z);
      }
      else
      {
        o.y = clamp(a.y * (1.0f - opacity) + (a.y + b.y) * o.x / 0.01f * opacity, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity) + (a.z + b.z) * o.x / 0.01f * opacity, min.z, max.z);
      }
      break;

    case DEVELOP_BLEND_AVERAGE:
      o = clamp(a * (1.0f - opacity) + (a + b) / 2.0f * opacity, min, max);
      break;

    case DEVELOP_BLEND_ADD:
      o = clamp(a * (1.0f - opacity) + (a + b) * opacity, min, max);
      break;

    case DEVELOP_BLEND_SUBTRACT:
      o = clamp(a * (1.0f - opacity) + (b + a - fabs(min + max)) * opacity, min, max);
      break;

    case DEVELOP_BLEND_DIFFERENCE:
      o = clamp(la * (1.0f - opacity) + fabs(la - lb) * opacity, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_DIFFERENCE2:
      to = fabs(a - b) / fabs(max - min);
      to.x = fmax(to.x, fmax(to.y, to.z));
      o = clamp(la * (1.0f - opacity) + to * opacity, lmin, lmax);
      o.y = o.z = 0.0f;
      break;

    case DEVELOP_BLEND_SCREEN:
      o = clamp(la * (1.0f - opacity) + (lmax - (lmax - la) * (lmax - lb)) * opacity, lmin, lmax) - fabs(min);
      if(a.x > 0.01f)
      {
        o.y = clamp(a.y * (1.0f - opacity) + 0.5f * (a.y + b.y) * o.x / a.x * opacity, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity) + 0.5f * (a.z + b.z) * o.x / a.x * opacity, min.z, max.z);
      }
      else
      {
        o.y = clamp(a.y * (1.0f - opacity) + 0.5f * (a.y + b.y) * o.x / 0.01f * opacity, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity) + 0.5f * (a.z + b.z) * o.x / 0.01f * opacity, min.z, max.z);
      }
      break;

    case DEVELOP_BLEND_OVERLAY:
      o = clamp(la * (1.0f - opacity2) + (la > halfmax ? lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb) : doublemax * la * lb) * opacity2, lmin, lmax) - fabs(min);
      if(a.x > 0.01f)
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / a.x * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / a.x * opacity2, min.z, max.z);
      }
      else
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / 0.01f * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / 0.01f * opacity2, min.z, max.z);
      }
      break;

    case DEVELOP_BLEND_SOFTLIGHT:
      o = clamp(la * (1.0f - opacity2) + (lb > halfmax ? lmax - (lmax - la) * (lmax - (lb - halfmax)) : la * (lb + halfmax)) * opacity2, lmin, lmax) - fabs(min);
      if(a.x > 0.01f)
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / a.x * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / a.x * opacity2, min.z, max.z);
      }
      else
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / 0.01f * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / 0.01f * opacity2, min.z, max.z);
      }
      break;

    case DEVELOP_BLEND_HARDLIGHT:
      o = clamp(la * (1.0f - opacity2) + (lb > halfmax ? lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb) : doublemax * la * lb) * opacity2, lmin, lmax) - fabs(min);
      if(a.x > 0.01f)
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / a.x * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / a.x * opacity2, min.z, max.z);
      }
      else
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / 0.01f * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / 0.01f * opacity2, min.z, max.z);
      }
      break;

    case DEVELOP_BLEND_VIVIDLIGHT:
      o = clamp(la * (1.0f - opacity2) + (lb > halfmax ? (lb >= lmax ? lmax : la / (doublemax * (lmax - lb))) : (lb <= lmin ? lmin : lmax - (lmax - la) / (doublemax * lb))) * opacity2, lmin, lmax) - fabs(min);
      if(a.x > 0.01f)
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / a.x * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / a.x * opacity2, min.z, max.z);
      }
      else
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / 0.01f * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / 0.01f * opacity2, min.z, max.z);
      }
      break;

    case DEVELOP_BLEND_LINEARLIGHT:
      o = clamp(la * (1.0f - opacity2) + (la + doublemax * lb - lmax) * opacity2, lmin, lmax) - fabs(min);
      if(a.x > 0.01f)
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / a.x * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / a.x * opacity2, min.z, max.z);
      }
      else
      {
        o.y = clamp(a.y * (1.0f - opacity2) + (a.y + b.y) * o.x / 0.01f * opacity2, min.y, max.y);
        o.z = clamp(a.z * (1.0f - opacity2) + (a.z + b.z) * o.x / 0.01f * opacity2, min.z, max.z);
      }
      break;

    case DEVELOP_BLEND_PINLIGHT:
      o = clamp(la * (1.0f - opacity2) + (lb > halfmax ? fmax(la, doublemax * (lb - halfmax)) : fmin(la, doublemax * lb)) * opacity2, lmin, lmax) - fabs(min);
      o.y = clamp(a.y, min.y, max.y);
      o.z = clamp(a.z, min.z, max.z);
      break;

    case DEVELOP_BLEND_LIGHTNESS:
      // no need to transfer to LCH as we only work on L, which is the same as in Lab
      o.x = clamp((a.x * (1.0f - opacity)) + (b.x * opacity), min.x, max.x);
      o.y = clamp(a.y, min.y, max.y);
      o.z = clamp(a.z, min.z, max.z);
      o.w = a.w;
      break;

    case DEVELOP_BLEND_CHROMA:
      ta = vk_Lab_2_LCH(clamp(a, min, max));
      tb = vk_Lab_2_LCH(clamp(b, min, max));
      to.x = ta.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = ta.z;
      to.w = a.w;
      o = clamp(vk_LCH_2_Lab(to), min, max);
      break;

    case DEVELOP_BLEND_HUE:
      ta = vk_Lab_2_LCH(clamp(a, min, max));
      tb = vk_Lab_2_LCH(clamp(b, min, max));
      to.x = ta.x;
      to.y = ta.y;
      d = fabs(ta.z - tb.z);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.z = fmod((ta.z * (1.0f - s)) + (tb.z * s) + 1.0f, 1.0f);
      to.w = a.w;
      o = clamp(vk_LCH_2_Lab(to), min, max);
      break;

    case DEVELOP_BLEND_COLOR:
      ta = vk_Lab_2_LCH(clamp(a, min, max));
      tb = vk_Lab_2_LCH(clamp(b, min, max));
      to.x = ta.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      d = fabs(ta.z - tb.z);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.z = fmod((ta.z * (1.0f - s)) + (tb.z * s) + 1.0f, 1.0f);
      to.w = a.w;
      o = clamp(vk_LCH_2_Lab(to), min, max);
      break;

    case DEVELOP_BLEND_COLORADJUST:
      ta = vk_Lab_2_LCH(clamp(a, min, max));
      tb = vk_Lab_2_LCH(clamp(b, min, max));
      to.x = tb.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      d = fabs(ta.z - tb.z);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.z = fmod((ta.z * (1.0f - s)) + (tb.z * s) + 1.0f, 1.0f);
      to.w = a.w;
      o = clamp(vk_LCH_2_Lab(to), min, max);
      break;

    case DEVELOP_BLEND_BOUNDED:
      o = clamp((a * (1.0f - opacity)) + (b * opacity), min, max);
      break;

    case DEVELOP_BLEND_LAB_LIGHTNESS:
    case DEVELOP_BLEND_LAB_L:
      o.x = (a.x * (1.0f - opacity)) + (b.x * opacity);
      o.y = a.y;
      o.z = a.z;
      o.w = a.w;
      break;

    case DEVELOP_BLEND_LAB_A:
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (b.y * opacity);
      o.z = a.z;
      o.w = a.w;
      break;

    case DEVELOP_BLEND_LAB_B:
      o.x = a.x;
      o.y = a.y;
      o.z = (a.z * (1.0f - opacity)) + (b.z * opacity);
      o.w = a.w;
      break;

    case DEVELOP_BLEND_LAB_COLOR:
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (b.y * opacity);
      o.z = (a.z * (1.0f - opacity)) + (b.z * opacity);
      o.w = a.w;
      break;

    case DEVELOP_BLEND_HSV_LIGHTNESS:
    case DEVELOP_BLEND_HSV_COLOR:
    case DEVELOP_BLEND_RGB_R:
    case DEVELOP_BLEND_RGB_G:
    case DEVELOP_BLEND_RGB_B:
      o = a; // Noop for Lab (without clamping)
      break;

    /* fallback to normal blend */
    case DEVELOP_BLEND_NORMAL2:
    default:
      o = (a * (1.0f - opacity)) + (b * opacity);
      break;
  }

  /* scale L back to [0; 100] and a,b to [-128; 128] */
  o *= scale;

  /* we transfer alpha channel of input if mask_display is set, else we save opacity into alpha channel */
  o.w = mask_display ? a.w : opacity;

  out[idx] = o;
}
