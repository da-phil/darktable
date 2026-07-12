/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan compute twin of blendop.cl::blendop_rgb_hsl — the
    display-referred RGB blend-apply kernel, all blend modes.

    Buffer-shape differences vs the OpenCL original (math unchanged):
    see the header comment in blendop_rgb_jzczhz.cl — same in-place
    `out` convention, same iwidth/offx/offy indexing for `in_a`.

    The vector ternaries (e.g. `la > halfmax ? X : Y`) are OpenCL C
    component-wise selects, exactly as in the original; keep them
    intact so clspv emits per-lane OpSelect.
*/

#include "dt_vulkan_common.h"
#include "dt_vulkan_blendop.h"

kernel void blendop_rgb_hsl(global const float4 *in_a,
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

  const float4 min = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
  const float4 max = (float4)(1.0f, 1.0f, 1.0f, 1.0f);
  const float4 lmin = (float4)(0.0f, 0.0f, 0.0f, 1.0f);
  const float4 lmax = (float4)(1.0f, 1.0f, 1.0f, 1.0f);       /* max + fabs(min) */
  const float4 halfmax = (float4)(0.5f, 0.5f, 0.5f, 1.0f);    /* lmax / 2.0f */
  const float4 doublemax = (float4)(2.0f, 2.0f, 2.0f, 1.0f);  /* lmax * 2.0f */
  const float opacity2 = opacity * opacity;

  float4 la = clamp(a + fabs(min), lmin, lmax);
  float4 lb = clamp(b + fabs(min), lmin, lmax);

  /* select the blend operator */
  switch(blend_mode & DEVELOP_BLEND_MODE_MASK)
  {
    case DEVELOP_BLEND_LIGHTEN:
      o = clamp(a * (1.0f - opacity) + fmax(a, b) * opacity, min, max);
      break;

    case DEVELOP_BLEND_DARKEN:
      o = clamp(a * (1.0f - opacity) + fmin(a, b) * opacity, min, max);
      break;

    case DEVELOP_BLEND_MULTIPLY:
      o = clamp(a * (1.0f - opacity) + a * b * opacity, min, max);
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
    case DEVELOP_BLEND_DIFFERENCE2:
      o = clamp(la * (1.0f - opacity) + fabs(la - lb) * opacity, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_SCREEN:
      o = clamp(la * (1.0f - opacity) + (lmax - (lmax - la) * (lmax - lb)) * opacity, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_OVERLAY:
      o = clamp(la * (1.0f - opacity2) + (la > halfmax ? lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb) : doublemax * la * lb) * opacity2, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_SOFTLIGHT:
      o = clamp(la * (1.0f - opacity2) + (lb > halfmax ? lmax - (lmax - la) * (lmax - (lb - halfmax)) : la * (lb + halfmax)) * opacity2, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_HARDLIGHT:
      o = clamp(la * (1.0f - opacity2) + (lb > halfmax ? lmax - (lmax - doublemax * (la - halfmax)) * (lmax - lb) : doublemax * la * lb) * opacity2, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_VIVIDLIGHT:
      o = clamp(la * (1.0f - opacity2) + (lb > halfmax ? (lb >= lmax ? lmax : la / (doublemax * (lmax - lb))) : (lb <= lmin ? lmin : lmax - (lmax - la) / (doublemax * lb))) * opacity2, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_LINEARLIGHT:
      o = clamp(la * (1.0f - opacity2) + (la + doublemax * lb - lmax) * opacity2, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_PINLIGHT:
      o = clamp(la * (1.0f - opacity2) + (lb > halfmax ? fmax(la, doublemax * (lb - halfmax)) : fmin(la, doublemax * lb)) * opacity2, lmin, lmax) - fabs(min);
      break;

    case DEVELOP_BLEND_LIGHTNESS:
      ta = vk_RGB_to_HSL(clamp(a, min, max));
      tb = vk_RGB_to_HSL(clamp(b, min, max));
      to.x = ta.x;
      to.y = ta.y;
      to.z = (ta.z * (1.0f - opacity)) + (tb.z * opacity);
      to.w = a.w;
      o = clamp(vk_HSL_to_RGB(to), min, max);
      break;

    case DEVELOP_BLEND_CHROMA:
      ta = vk_RGB_to_HSL(clamp(a, min, max));
      tb = vk_RGB_to_HSL(clamp(b, min, max));
      to.x = ta.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = ta.z;
      to.w = a.w;
      o = clamp(vk_HSL_to_RGB(to), min, max);
      break;

    case DEVELOP_BLEND_HUE:
      ta = vk_RGB_to_HSL(clamp(a, min, max));
      tb = vk_RGB_to_HSL(clamp(b, min, max));
      d = fabs(ta.x - tb.x);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.x = fmod((ta.x * (1.0f - s)) + (tb.x * s) + 1.0f, 1.0f);
      to.y = ta.y;
      to.z = ta.z;
      to.w = a.w;
      o = clamp(vk_HSL_to_RGB(to), min, max);
      break;

    case DEVELOP_BLEND_COLOR:
      ta = vk_RGB_to_HSL(clamp(a, min, max));
      tb = vk_RGB_to_HSL(clamp(b, min, max));
      d = fabs(ta.x - tb.x);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.x = fmod((ta.x * (1.0f - s)) + (tb.x * s) + 1.0f, 1.0f);
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = ta.z;
      to.w = a.w;
      o = clamp(vk_HSL_to_RGB(to), min, max);
      break;

    case DEVELOP_BLEND_COLORADJUST:
      ta = vk_RGB_to_HSL(clamp(a, min, max));
      tb = vk_RGB_to_HSL(clamp(b, min, max));
      d = fabs(ta.x - tb.x);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.x = fmod((ta.x * (1.0f - s)) + (tb.x * s) + 1.0f, 1.0f);
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = tb.z;
      to.w = a.w;
      o = clamp(vk_HSL_to_RGB(to), min, max);
      break;

    case DEVELOP_BLEND_BOUNDED:
      o = clamp((a * (1.0f - opacity)) + (b * opacity), min, max);
      break;

    case DEVELOP_BLEND_LAB_LIGHTNESS:
    case DEVELOP_BLEND_LAB_COLOR:
    case DEVELOP_BLEND_LAB_L:
    case DEVELOP_BLEND_LAB_A:
    case DEVELOP_BLEND_LAB_B:
      o = a; // Noop for RGB (without clamping)
      break;

    case DEVELOP_BLEND_HSV_LIGHTNESS:
      ta = vk_RGB_to_HSV(a);
      tb = vk_RGB_to_HSV(b);
      to.x = ta.x;
      to.y = ta.y;
      to.z = (ta.z * (1.0f - opacity)) + (tb.z * opacity);
      to.w = a.w;
      o = vk_HSV_to_RGB(to);
      break;

    case DEVELOP_BLEND_HSV_COLOR:
      ta = vk_RGB_to_HSV(a);
      tb = vk_RGB_to_HSV(b);
      // blend color vectors of input and output
      d = ta.y * cos(DT_2PI_F * ta.x) * (1.0f - opacity) + tb.y * cos(DT_2PI_F * tb.x) * opacity;
      s = ta.y * sin(DT_2PI_F * ta.x) * (1.0f - opacity) + tb.y * sin(DT_2PI_F * tb.x) * opacity;
      to.x = fmod(atan2(s, d) / DT_2PI_F + 1.0f, 1.0f);
      to.y = vk_dt_fast_hypot(s, d);
      to.z = ta.z;
      to.w = a.w;
      o = vk_HSV_to_RGB(to);
      break;

    case DEVELOP_BLEND_RGB_R:
      o.x = (a.x * (1.0f - opacity)) + (b.x * opacity);
      o.y = a.y;
      o.z = a.z;
      o.w = a.w;
      break;

    case DEVELOP_BLEND_RGB_G:
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (b.y * opacity);
      o.z = a.z;
      o.w = a.w;
      break;

    case DEVELOP_BLEND_RGB_B:
      o.x = a.x;
      o.y = a.y;
      o.z = (a.z * (1.0f - opacity)) + (b.z * opacity);
      o.w = a.w;
      break;

    /* fallback to normal blend */
    case DEVELOP_BLEND_NORMAL2:
    default:
      o = (a * (1.0f - opacity)) + (b * opacity);
      break;
  }

  /* we transfer alpha channel of input if mask_display is set, else we save opacity into alpha channel */
  o.w = mask_display ? a.w : opacity;

  out[idx] = o;
}
