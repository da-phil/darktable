/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/
/*
 * cmocka tests for the Vulkan blendop kernels (DAG milestone M0,
 * dev-doc/gpu_resident_pixelpipe_dag.md §9).
 *
 * The kernel-level tests compare the production .spv modules (the
 * same files darktable ships) against independent C references
 * hand-translated from data/kernels/blendop.cl — the ground truth
 * the whole Vulkan port validates against — for a representative
 * mode set per blend colorspace covering every structural feature:
 * the REVERSE operand swap, roi offsets, per-pixel opacity, the
 * mask_display alpha transfer, component-wise vector selects,
 * blend_parameter, and the HSL/HSV/LCH helper round-trips.
 *
 * The function-level tests drive dt_develop_blend_process_vk with a
 * scaffolded pipe piece: subset-gate refusals (non-uniform masks,
 * colorspace mismatch, RAW) and an end-to-end uniform blend.
 *
 * Runs on any Vulkan implementation (lavapipe in CI); all tests skip
 * cleanly when no device is available.
 */
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmocka.h>

#include "common/darktable.h"
#include "common/vulkan.h"
#include "common/iop_profile.h"
#include "control/conf.h"
#include "develop/blend.h"
#include "develop/imageop.h"
#include "develop/pixelpipe_hb.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

#ifndef HAVE_VULKAN
#error "test_vulkan_blendop requires USE_VULKAN builds"
#endif

// odd sizes exercise the workgroup bounds checks
#define TW 61
#define TH 37
#define TN ((size_t)TW * TH)

typedef struct v4
{
  float x, y, z, w;
} v4;

typedef struct pc_blend_t
{
  int32_t width, height, iwidth;
  uint32_t blend_mode;
  float blend_parameter;
  int32_t offx, offy, mask_display;
} pc_blend_t;

typedef struct pc_setmask_t
{
  int32_t width, height;
  float value;
} pc_setmask_t;

static dt_vulkan_t s_vk;
static dt_conf_t s_conf;
static gboolean s_have_device = FALSE;
static dt_vk_module_kernel_t k_set_mask = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t k_lab = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t k_hsl = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t k_jzczhz = DT_VK_MODULE_KERNEL_INIT;

#define REQUIRE_DEVICE() do { if(!s_have_device) skip(); } while(0)

/*
 * small vec4 helpers for the C references
 */
static inline v4 v4s(float s) { return (v4){ s, s, s, s }; }
static inline v4 v4add(v4 a, v4 b) { return (v4){ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w }; }
static inline v4 v4sub(v4 a, v4 b) { return (v4){ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w }; }
static inline v4 v4mul(v4 a, v4 b) { return (v4){ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w }; }
static inline v4 v4div(v4 a, v4 b) { return (v4){ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w }; }
static inline v4 v4scale(v4 a, float s) { return (v4){ a.x * s, a.y * s, a.z * s, a.w * s }; }
static inline v4 v4abs(v4 a) { return (v4){ fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w) }; }
static inline v4 v4max(v4 a, v4 b) { return (v4){ fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w) }; }
static inline v4 v4min(v4 a, v4 b) { return (v4){ fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w) }; }
static inline v4 v4clamp(v4 a, v4 lo, v4 hi) { return v4min(v4max(a, lo), hi); }
static inline v4 v4sqrt(v4 a) { return (v4){ sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w) }; }
// OpenCL component-wise ternary `c ? t : f` (per-lane bitwise select)
static inline v4 v4sel(v4 t, v4 f, v4 a, v4 b, int cmp /*0:> 1:>= 2:<=*/)
{
  v4 o;
  const float *ta = (const float *)&t, *fa = (const float *)&f;
  const float *aa = (const float *)&a, *ba = (const float *)&b;
  float *oa = (float *)&o;
  for(int i = 0; i < 4; i++)
  {
    gboolean c = cmp == 0 ? aa[i] > ba[i] : cmp == 1 ? aa[i] >= ba[i] : aa[i] <= ba[i];
    oa[i] = c ? ta[i] : fa[i];
  }
  return o;
}

/*
 * colorspace helpers — translated from
 * data/kernels/vulkan/dt_vulkan_common.h (which mirrors
 * data/kernels/colorspace.h byte-for-byte)
 */
#define REF_2PI 6.28318530717958647692f

static v4 ref_RGB_2_HSL(v4 RGB)
{
  float H = 0.0f, S = 0.0f;
  const float R = RGB.x, G = RGB.y, B = RGB.z;
  const float var_Min = fminf(R, fminf(G, B));
  const float var_Max = fmaxf(R, fmaxf(G, B));
  const float del_Max = var_Max - var_Min;
  const float L = (var_Max + var_Min) * 0.5f;

  if(fabsf(var_Max) > 1e-6f && fabsf(del_Max) > 1e-6f)
  {
    S = (L < 0.5f) ? del_Max / (var_Max + var_Min) : del_Max / (2.0f - var_Max - var_Min);
    const float del_R = (((var_Max - R) / 6.0f) + (del_Max / 2.0f)) / del_Max;
    const float del_G = (((var_Max - G) / 6.0f) + (del_Max / 2.0f)) / del_Max;
    const float del_B = (((var_Max - B) / 6.0f) + (del_Max / 2.0f)) / del_Max;

    if(R == var_Max) H = del_B - del_G;
    else if(G == var_Max) H = (1.0f / 3.0f) + del_R - del_B;
    else if(B == var_Max) H = (2.0f / 3.0f) + del_G - del_R;

    if(H < 0.0f) H += 1.0f;
    if(H > 1.0f) H -= 1.0f;
  }
  return (v4){ H, S, L, RGB.w };
}

static float ref_hue_to_rgb(float v1, float v2, float vH)
{
  if(vH < 0.0f) vH += 1.0f;
  if(vH > 1.0f) vH -= 1.0f;
  if(6.0f * vH < 1.0f) return v1 + (v2 - v1) * 6.0f * vH;
  if(2.0f * vH < 1.0f) return v2;
  if(3.0f * vH < 2.0f) return v1 + (v2 - v1) * ((2.0f / 3.0f) - vH) * 6.0f;
  return v1;
}

static v4 ref_HSL_2_RGB(v4 HSL)
{
  const float H = HSL.x, S = HSL.y, L = HSL.z;
  if(S < 1e-6f) return (v4){ L, L, L, HSL.w };
  const float var_2 = (L < 0.5f) ? L * (1.0f + S) : (L + S) - (S * L);
  const float var_1 = 2.0f * L - var_2;
  return (v4){ ref_hue_to_rgb(var_1, var_2, H + 1.0f / 3.0f),
               ref_hue_to_rgb(var_1, var_2, H),
               ref_hue_to_rgb(var_1, var_2, H - 1.0f / 3.0f), HSL.w };
}

static v4 ref_RGB_2_HSV(v4 RGB)
{
  v4 HSV;
  const float minv = fminf(RGB.x, fminf(RGB.y, RGB.z));
  const float maxv = fmaxf(RGB.x, fmaxf(RGB.y, RGB.z));
  const float delta = maxv - minv;
  HSV.z = maxv;
  HSV.w = RGB.w;
  if(fabsf(maxv) > 1e-6f && fabsf(delta) > 1e-6f)
    HSV.y = delta / maxv;
  else
  {
    HSV.x = 0.0f;
    HSV.y = 0.0f;
    return HSV;
  }
  if(RGB.x == maxv) HSV.x = (RGB.y - RGB.z) / delta;
  else if(RGB.y == maxv) HSV.x = 2.0f + (RGB.z - RGB.x) / delta;
  else HSV.x = 4.0f + (RGB.x - RGB.y) / delta;
  HSV.x /= 6.0f;
  HSV.x -= floorf(HSV.x);
  return HSV;
}

static v4 ref_HSV_2_RGB(v4 HSV)
{
  if(fabsf(HSV.y) < 1e-6f) return (v4){ HSV.z, HSV.z, HSV.z, HSV.w };
  const int i = (int)floorf(6.0f * HSV.x);
  const float vv = HSV.z, w = HSV.w;
  const float p = vv * (1.0f - HSV.y);
  const float q = vv * (1.0f - HSV.y * (6.0f * HSV.x - (float)i));
  const float t = vv * (1.0f - HSV.y * (1.0f - (6.0f * HSV.x - (float)i)));
  switch(i)
  {
    case 0: return (v4){ vv, t, p, w };
    case 1: return (v4){ q, vv, p, w };
    case 2: return (v4){ p, vv, t, w };
    case 3: return (v4){ p, q, vv, w };
    case 4: return (v4){ t, p, vv, w };
    default: return (v4){ vv, p, q, w };
  }
}

static v4 ref_Lab_2_LCH(v4 Lab)
{
  float H = atan2f(Lab.z, Lab.y);
  H = (H > 0.0f) ? H / REF_2PI : 1.0f - fabsf(H) / REF_2PI;
  const float C = sqrtf(Lab.y * Lab.y + Lab.z * Lab.z);
  return (v4){ Lab.x, C, H, Lab.w };
}

static v4 ref_LCH_2_Lab(v4 LCH)
{
  return (v4){ LCH.x, cosf(REF_2PI * LCH.z) * LCH.y, sinf(REF_2PI * LCH.z) * LCH.y, LCH.w };
}

/*
 * C references translated from data/kernels/blendop.cl
 */

static v4 ref_blend_jzczhz(v4 a0, v4 b0, float opacity, uint32_t blend_mode,
                           float blend_parameter, int mask_display)
{
  v4 a = a0, b = b0, o;
  if((blend_mode & 0x80000000u) == 0x80000000u) { a = b0; b = a0; }
  float norm_a, norm_b;
  switch(blend_mode & 0xFFu)
  {
    case 0x04: // MULTIPLY
      o = v4add(v4scale(a, 1.0f - opacity), v4scale(v4mul(a, b), blend_parameter * opacity));
      break;
    case 0x25: // SUBTRACT_INVERSE
      o = v4add(v4scale(a, 1.0f - opacity),
                v4scale(v4max(v4sub(b, v4scale(a, blend_parameter)), v4s(0.0f)), opacity));
      break;
    case 0x11: // CHROMA
      norm_a = fmaxf(sqrtf(a.x * a.x + a.y * a.y + a.z * a.z), 1e-6f);
      norm_b = fmaxf(sqrtf(b.x * b.x + b.y * b.y + b.z * b.z), 1e-6f);
      o = v4add(v4scale(a, 1.0f - opacity), v4scale(b, norm_a / norm_b * opacity));
      break;
    case 0x26: // DIVIDE
      o = v4add(v4scale(a, 1.0f - opacity),
                v4scale(v4div(a, v4max(v4scale(b, blend_parameter), v4s(1e-6f))), opacity));
      break;
    case 0x29: // HARMONIC_MEAN
      o = v4add(v4scale(a, 1.0f - opacity),
                v4scale(v4div(v4scale(v4mul(a, b), 2.0f),
                              v4add(v4max(a, v4s(5e-7f)), v4max(b, v4s(5e-7f)))), opacity));
      break;
    case 0x18: // NORMAL2
    default:
      o = v4add(v4scale(a, 1.0f - opacity), v4scale(b, opacity));
      break;
  }
  o.w = mask_display ? a.w : opacity;
  return o;
}

static v4 ref_blend_rgb_hsl(v4 a0, v4 b0, float opacity, uint32_t blend_mode,
                            int mask_display)
{
  v4 a = a0, b = b0, o, ta, tb, to;
  float d, s;
  if((blend_mode & 0x80000000u) == 0x80000000u) { a = b0; b = a0; }

  const v4 mn = { 0.0f, 0.0f, 0.0f, 1.0f };
  const v4 mx = { 1.0f, 1.0f, 1.0f, 1.0f };
  const v4 lmin = { 0.0f, 0.0f, 0.0f, 1.0f };
  const v4 lmax = { 1.0f, 1.0f, 1.0f, 1.0f };
  const v4 halfmax = { 0.5f, 0.5f, 0.5f, 1.0f };
  const v4 doublemax = { 2.0f, 2.0f, 2.0f, 1.0f };
  const float opacity2 = opacity * opacity;

  const v4 la = v4clamp(v4add(a, v4abs(mn)), lmin, lmax);
  const v4 lb = v4clamp(v4add(b, v4abs(mn)), lmin, lmax);

  switch(blend_mode & 0xFFu)
  {
    case 0x04: // MULTIPLY
      o = v4clamp(v4add(v4scale(a, 1.0f - opacity), v4scale(v4mul(a, b), opacity)), mn, mx);
      break;
    case 0x09: // SCREEN
      o = v4sub(v4clamp(v4add(v4scale(la, 1.0f - opacity),
                              v4scale(v4sub(lmax, v4mul(v4sub(lmax, la), v4sub(lmax, lb))), opacity)),
                        lmin, lmax), v4abs(mn));
      break;
    case 0x0A: // OVERLAY
      o = v4sub(v4clamp(v4add(v4scale(la, 1.0f - opacity2),
                v4scale(v4sel(v4sub(lmax, v4mul(v4sub(lmax, v4mul(doublemax, v4sub(la, halfmax))), v4sub(lmax, lb))),
                              v4mul(doublemax, v4mul(la, lb)),
                              la, halfmax, 0),
                        opacity2)), lmin, lmax), v4abs(mn));
      break;
    case 0x0D: // VIVIDLIGHT
    {
      const v4 hi = v4sel(lmax, v4div(la, v4mul(doublemax, v4sub(lmax, lb))), lb, lmax, 1);
      const v4 lo = v4sel(lmin, v4sub(lmax, v4div(v4sub(lmax, la), v4mul(doublemax, lb))), lb, lmin, 2);
      o = v4sub(v4clamp(v4add(v4scale(la, 1.0f - opacity2),
                              v4scale(v4sel(hi, lo, lb, halfmax, 0), opacity2)),
                        lmin, lmax), v4abs(mn));
      break;
    }
    case 0x12: // HUE
      ta = ref_RGB_2_HSL(v4clamp(a, mn, mx));
      tb = ref_RGB_2_HSL(v4clamp(b, mn, mx));
      d = fabsf(ta.x - tb.x);
      s = d > 0.5f ? -opacity * (1.0f - d) / d : opacity;
      to.x = fmodf((ta.x * (1.0f - s)) + (tb.x * s) + 1.0f, 1.0f);
      to.y = ta.y;
      to.z = ta.z;
      to.w = a.w;
      o = v4clamp(ref_HSL_2_RGB(to), mn, mx);
      break;
    case 0x1D: // HSV_COLOR
      ta = ref_RGB_2_HSV(a);
      tb = ref_RGB_2_HSV(b);
      d = ta.y * cosf(REF_2PI * ta.x) * (1.0f - opacity) + tb.y * cosf(REF_2PI * tb.x) * opacity;
      s = ta.y * sinf(REF_2PI * ta.x) * (1.0f - opacity) + tb.y * sinf(REF_2PI * tb.x) * opacity;
      to.x = fmodf(atan2f(s, d) / REF_2PI + 1.0f, 1.0f);
      to.y = sqrtf(s * s + d * d);
      to.z = ta.z;
      to.w = a.w;
      o = ref_HSV_2_RGB(to);
      break;
    case 0x21: // RGB_R
      o.x = (a.x * (1.0f - opacity)) + (b.x * opacity);
      o.y = a.y;
      o.z = a.z;
      o.w = a.w;
      break;
    case 0x18: // NORMAL2
    default:
      o = v4add(v4scale(a, 1.0f - opacity), v4scale(b, opacity));
      break;
  }
  o.w = mask_display ? a.w : opacity;
  return o;
}

static v4 ref_blend_lab(v4 a0, v4 b0, float opacity, uint32_t blend_mode,
                        int mask_display)
{
  v4 a = a0, b = b0, o, ta, tb, to;
  if((blend_mode & 0x80000000u) == 0x80000000u) { a = b0; b = a0; }

  const v4 scale = { 100.0f, 128.0f, 128.0f, 1.0f };
  a = v4div(a, scale);
  b = v4div(b, scale);

  const v4 mn = { 0.0f, -1.0f, -1.0f, 0.0f };
  const v4 mx = { 1.0f, 1.0f, 1.0f, 1.0f };
  const v4 lmin = { 0.0f, 0.0f, 0.0f, 1.0f };
  const v4 lmax = { 1.0f, 2.0f, 2.0f, 1.0f };
  const v4 halfmax = { 0.5f, 1.0f, 1.0f, 0.5f };
  const v4 doublemax = { 2.0f, 4.0f, 4.0f, 2.0f };
  const float opacity2 = opacity * opacity;

  const v4 la = v4clamp(v4add(a, v4abs(mn)), lmin, lmax);
  const v4 lb = v4clamp(v4add(b, v4abs(mn)), lmin, lmax);

  switch(blend_mode & 0xFFu)
  {
    case 0x02: // LIGHTEN
      o = v4clamp(v4add(v4scale(a, 1.0f - opacity), v4scale(v4sel(a, b, a, b, 0), opacity)), mn, mx);
      o.y = fminf(fmaxf(a.y * (1.0f - fabsf(o.x - a.x)) + 0.5f * (a.y + b.y) * fabsf(o.x - a.x), mn.y), mx.y);
      o.z = fminf(fmaxf(a.z * (1.0f - fabsf(o.x - a.x)) + 0.5f * (a.z + b.z) * fabsf(o.x - a.x), mn.z), mx.z);
      break;
    case 0x04: // MULTIPLY
      o = v4clamp(v4add(v4scale(a, 1.0f - opacity), v4scale(v4mul(a, b), opacity)), mn, mx);
      if(a.x > 0.01f)
      {
        o.y = fminf(fmaxf(a.y * (1.0f - opacity) + (a.y + b.y) * o.x / a.x * opacity, mn.y), mx.y);
        o.z = fminf(fmaxf(a.z * (1.0f - opacity) + (a.z + b.z) * o.x / a.x * opacity, mn.z), mx.z);
      }
      else
      {
        o.y = fminf(fmaxf(a.y * (1.0f - opacity) + (a.y + b.y) * o.x / 0.01f * opacity, mn.y), mx.y);
        o.z = fminf(fmaxf(a.z * (1.0f - opacity) + (a.z + b.z) * o.x / 0.01f * opacity, mn.z), mx.z);
      }
      break;
    case 0x0F: // PINLIGHT
    {
      const v4 hi = v4max(la, v4mul(doublemax, v4sub(lb, halfmax)));
      const v4 lo = v4min(la, v4mul(doublemax, lb));
      o = v4sub(v4clamp(v4add(v4scale(la, 1.0f - opacity2),
                              v4scale(v4sel(hi, lo, lb, halfmax, 0), opacity2)),
                        lmin, lmax), v4abs(mn));
      o.y = fminf(fmaxf(a.y, mn.y), mx.y);
      o.z = fminf(fmaxf(a.z, mn.z), mx.z);
      break;
    }
    case 0x11: // CHROMA
      ta = ref_Lab_2_LCH(v4clamp(a, mn, mx));
      tb = ref_Lab_2_LCH(v4clamp(b, mn, mx));
      to.x = ta.x;
      to.y = (ta.y * (1.0f - opacity)) + (tb.y * opacity);
      to.z = ta.z;
      to.w = a.w;
      o = v4clamp(ref_LCH_2_Lab(to), mn, mx);
      break;
    case 0x1F: // LAB_A
      o.x = a.x;
      o.y = (a.y * (1.0f - opacity)) + (b.y * opacity);
      o.z = a.z;
      o.w = a.w;
      break;
    case 0x19: // BOUNDED
      o = v4clamp(v4add(v4scale(a, 1.0f - opacity), v4scale(b, opacity)), mn, mx);
      break;
    case 0x18: // NORMAL2
    default:
      o = v4add(v4scale(a, 1.0f - opacity), v4scale(b, opacity));
      break;
  }

  o = v4mul(o, scale);
  o.w = mask_display ? a.w : opacity;
  return o;
}

/*
 * fixtures
 */

static int group_setup(void **state)
{
  (void)state;
  darktable.conf = &s_conf;
  dt_conf_init(darktable.conf, "/nonexistent-dt-vk-blendop-test.rc", FALSE, NULL);
  dt_conf_set_bool("opencl_use_vulkan", TRUE);
  // dt_vulkan_load_program_by_name resolves <datadir>/kernels/vulkan;
  // point it at the build tree so the tests run the production .spv
  darktable.datadir = g_strdup(TEST_VK_DATADIR);

  darktable.vulkan = &s_vk;
  dt_vulkan_init(&s_vk);
  if(!dt_vulkan_running())
  {
    fprintf(stderr, "no Vulkan device available — blendop tests will be skipped\n");
    return 0;
  }

  dt_vulkan_module_kernel_load(&k_set_mask, "blendop_set_mask", "blendop_set_mask",
                               1, sizeof(pc_setmask_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&k_lab, "blendop_lab", "blendop_lab",
                               3, sizeof(pc_blend_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&k_hsl, "blendop_rgb_hsl", "blendop_rgb_hsl",
                               3, sizeof(pc_blend_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&k_jzczhz, "blendop_rgb_jzczhz", "blendop_rgb_jzczhz",
                               3, sizeof(pc_blend_t), 16, 16, 1);
  if(k_set_mask.kernel < 0 || k_lab.kernel < 0 || k_hsl.kernel < 0 || k_jzczhz.kernel < 0)
  {
    fprintf(stderr, "blendop kernels failed to load — tests will be skipped\n");
    return 0;
  }

  s_have_device = TRUE;
  return 0;
}

static int group_teardown(void **state)
{
  (void)state;
  dt_vulkan_cleanup(&s_vk);
  darktable.vulkan = NULL;
  return 0;
}

// deterministic pseudo-random floats
static uint32_t s_rng;
static float frand(float lo, float hi)
{
  s_rng = s_rng * 1664525u + 1013904223u;
  return lo + (hi - lo) * ((s_rng >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
}

static void fill_rgb(v4 *buf, size_t n)
{
  for(size_t i = 0; i < n; i++)
    buf[i] = (v4){ frand(-0.25f, 1.5f), frand(-0.25f, 1.5f), frand(-0.25f, 1.5f), frand(0.0f, 1.0f) };
}

static void fill_lab(v4 *buf, size_t n)
{
  for(size_t i = 0; i < n; i++)
    buf[i] = (v4){ frand(-20.0f, 120.0f), frand(-150.0f, 150.0f), frand(-150.0f, 150.0f), frand(0.0f, 1.0f) };
}

// Run one blend kernel on device over equal rois and compare against
// the given per-pixel reference.
typedef v4 (*ref_fn_t)(v4 a, v4 b, float opacity, uint32_t mode, float param, int mask_display);

static v4 _ref_hsl_adapter(v4 a, v4 b, float op, uint32_t m, float p, int md)
{ (void)p; return ref_blend_rgb_hsl(a, b, op, m, md); }
static v4 _ref_lab_adapter(v4 a, v4 b, float op, uint32_t m, float p, int md)
{ (void)p; return ref_blend_lab(a, b, op, m, md); }

// cmocka asserts longjmp out of the test on failure; doing that while
// g_vk_lock is held would deadlock every following test. So each test
// runs the whole device sequence first (collecting an rc), releases the
// lock, and only then asserts.
// `mask_arr` NULL => uniform `opacity` everywhere (the common case). A
// non-NULL per-pixel mask exercises the apply kernel's spatial mask
// indexing — the property drawn/parametric masks (ME.4) rely on — with
// the reference blended at each pixel's own mask value.
static void run_mode_check(const dt_vk_module_kernel_t *kernel,
                           const v4 *in_a, const v4 *in_b,
                           float opacity, uint32_t mode, float param,
                           int mask_display, ref_fn_t ref, float eps,
                           const char *tag, const float *mask_arr)
{
  const size_t bytes = TN * sizeof(v4);
  float *mask = malloc(TN * sizeof(float));
  v4 *out = malloc(bytes);
  assert_non_null(mask); assert_non_null(out);
  for(size_t i = 0; i < TN; i++) mask[i] = mask_arr ? mask_arr[i] : opacity;

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  dt_vk_mem_t *da = NULL, *db = NULL, *dm = NULL;
  if(!rc)
  {
    da = dt_vulkan_alloc_buffer(dev, bytes);
    db = dt_vulkan_alloc_buffer(dev, bytes);
    dm = dt_vulkan_alloc_buffer(dev, TN * sizeof(float));
    if(!da || !db || !dm) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, da, in_a, bytes);
  if(!rc) rc = dt_vulkan_write_to_device(dev, db, in_b, bytes);
  if(!rc) rc = dt_vulkan_write_to_device(dev, dm, mask, TN * sizeof(float));
  if(!rc)
  {
    const pc_blend_t pc = { TW, TH, TW, mode, param, 0, 0, mask_display };
    dt_vk_mem_t *bufs[3] = { da, db, dm };
    rc = dt_vulkan_dispatch_n(kernel, bufs, 3, TW, TH, &pc, sizeof(pc));
  }
  if(!rc) rc = dt_vulkan_read_from_device(dev, out, db, bytes);
  if(da) dt_vulkan_free_buffer(dev, da);
  if(db) dt_vulkan_free_buffer(dev, db);
  if(dm) dt_vulkan_free_buffer(dev, dm);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  // scale-aware error: absolute for |ref| <= 1, relative above — the
  // HSL/HSV round-trips on out-of-gamut inputs legitimately produce
  // intermediates of magnitude ~100 where GPU fma reassociation costs
  // a few ulp more than an absolute epsilon calibrated near 1.0 allows
#define BLEND_ERR(g, r) (fabsf((g) - (r)) / fmaxf(1.0f, fabsf(r)))
  float maxerr = 0.0f;
  size_t worst = 0;
  v4 wa = v4s(0.0f), wb = v4s(0.0f), wo = v4s(0.0f), wr = v4s(0.0f);
  if(!rc)
    for(size_t i = 0; i < TN; i++)
    {
      const float px_op = mask_arr ? mask_arr[i] : opacity;
      const v4 r = ref(in_a[i], in_b[i], px_op, mode, param, mask_display);
      const float e = fmaxf(fmaxf(BLEND_ERR(out[i].x, r.x), BLEND_ERR(out[i].y, r.y)),
                            fmaxf(BLEND_ERR(out[i].z, r.z), BLEND_ERR(out[i].w, r.w)));
      if(e > maxerr)
      {
        maxerr = e;
        worst = i;
        wa = in_a[i]; wb = in_b[i]; wo = out[i]; wr = r;
      }
    }
#undef BLEND_ERR
  free(mask); free(out);

  if(rc)
    fail_msg("%s: device sequence failed rc=%d (mode 0x%x, opacity %g)",
             tag, rc, mode, (double)opacity);
  if(maxerr > eps)
    fail_msg("%s: max error %g exceeds eps %g (mode 0x%x, opacity %g) at px %zu\n"
             "  a    = (%g, %g, %g, %g)\n"
             "  b    = (%g, %g, %g, %g)\n"
             "  got  = (%g, %g, %g, %g)\n"
             "  want = (%g, %g, %g, %g)",
             tag, (double)maxerr, (double)eps, mode, (double)opacity, worst,
             (double)wa.x, (double)wa.y, (double)wa.z, (double)wa.w,
             (double)wb.x, (double)wb.y, (double)wb.z, (double)wb.w,
             (double)wo.x, (double)wo.y, (double)wo.z, (double)wo.w,
             (double)wr.x, (double)wr.y, (double)wr.z, (double)wr.w);
}

/*
 * TESTS
 */

static void test_set_mask(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  float *mask = malloc(TN * sizeof(float));
  assert_non_null(mask);

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  dt_vk_mem_t *dm = NULL;
  if(!rc)
  {
    dm = dt_vulkan_alloc_buffer(dev, TN * sizeof(float));
    if(!dm) rc = -101;
  }
  if(!rc)
  {
    const pc_setmask_t pc = { TW, TH, 0.4375f };
    dt_vk_mem_t *bufs[1] = { dm };
    rc = dt_vulkan_dispatch_n(&k_set_mask, bufs, 1, TW, TH, &pc, sizeof(pc));
  }
  if(!rc) rc = dt_vulkan_read_from_device(dev, mask, dm, TN * sizeof(float));
  if(dm) dt_vulkan_free_buffer(dev, dm);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  assert_int_equal(rc, 0);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(mask[i], 0.4375f, 0.0f);
  free(mask);
}

static void test_jzczhz_modes(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  s_rng = 0xbeef;
  v4 *a = malloc(TN * sizeof(v4)), *b = malloc(TN * sizeof(v4));
  assert_non_null(a); assert_non_null(b);
  fill_rgb(a, TN); fill_rgb(b, TN);

  const uint32_t modes[] = { 0x18 /*NORMAL2*/, 0x04 /*MULTIPLY*/, 0x25 /*SUB_INV*/,
                             0x11 /*CHROMA*/, 0x26 /*DIVIDE*/, 0x29 /*HARMONIC*/,
                             0x18 | 0x80000000u /*NORMAL2+REVERSE*/ };
  const float ops[] = { 0.0f, 0.35f, 1.0f };
  for(size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++)
    for(size_t p = 0; p < 3; p++)
      run_mode_check(&k_jzczhz, a, b, ops[p], modes[m], 1.3f, 0,
                     ref_blend_jzczhz, 1e-5f, "jzczhz", NULL);

  // mask_display transfers a.w
  run_mode_check(&k_jzczhz, a, b, 0.5f, 0x18, 1.0f, 1, ref_blend_jzczhz, 1e-5f, "jzczhz/md", NULL);

  free(a); free(b);
}

static void test_rgb_hsl_modes(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  s_rng = 0xcafe;
  v4 *a = malloc(TN * sizeof(v4)), *b = malloc(TN * sizeof(v4));
  assert_non_null(a); assert_non_null(b);
  fill_rgb(a, TN); fill_rgb(b, TN);

  const uint32_t modes[] = { 0x18 /*NORMAL2*/, 0x04 /*MULTIPLY*/, 0x09 /*SCREEN*/,
                             0x0A /*OVERLAY*/, 0x0D /*VIVIDLIGHT*/, 0x12 /*HUE*/,
                             0x1D /*HSV_COLOR*/, 0x21 /*RGB_R*/ };
  const float ops[] = { 0.0f, 0.35f, 1.0f };
  for(size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++)
    for(size_t p = 0; p < 3; p++)
      run_mode_check(&k_hsl, a, b, ops[p], modes[m], 1.0f, 0,
                     _ref_hsl_adapter, 1e-4f, "rgb_hsl", NULL);

  free(a); free(b);
}

static void test_lab_modes(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  s_rng = 0xf00d;
  v4 *a = malloc(TN * sizeof(v4)), *b = malloc(TN * sizeof(v4));
  assert_non_null(a); assert_non_null(b);
  fill_lab(a, TN); fill_lab(b, TN);

  const uint32_t modes[] = { 0x18 /*NORMAL2*/, 0x02 /*LIGHTEN*/, 0x04 /*MULTIPLY*/,
                             0x0F /*PINLIGHT*/, 0x11 /*CHROMA*/, 0x1F /*LAB_A*/,
                             0x19 /*BOUNDED*/, 0x18 | 0x80000000u };
  const float ops[] = { 0.0f, 0.35f, 1.0f };
  // the error metric is scale-aware, so the [0,100]/[-128,128] Lab
  // ranges need no special-casing; CHROMA still round-trips through
  // LCH trig where lavapipe's sin/cos/atan2 differ from libm by more
  // ulps than the plain arithmetic modes
  for(size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++)
    for(size_t p = 0; p < 3; p++)
      run_mode_check(&k_lab, a, b, ops[p], modes[m], 1.0f, 0,
                     _ref_lab_adapter, 5e-4f, "lab", NULL);

  free(a); free(b);
}

// ME.4 foundation: the apply kernel must read the mask per pixel, not
// treat it as one global opacity. Feed a spatially-varying mask (the
// shape a drawn/parametric mask produces) and require the device blend
// to match the reference blended at each pixel's own mask value. This is
// what lets a future non-uniform mask be uploaded and applied on device
// instead of forcing the CPU blend.
static void test_per_pixel_mask(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  s_rng = 0x5a5a;
  v4 *a = malloc(TN * sizeof(v4)), *b = malloc(TN * sizeof(v4));
  float *mask = malloc(TN * sizeof(float));
  assert_non_null(a); assert_non_null(b); assert_non_null(mask);
  fill_lab(a, TN); fill_lab(b, TN);
  // clearly non-uniform, deterministic, spans [0,1]
  for(size_t i = 0; i < TN; i++) mask[i] = (float)((i * 37u) % 101u) / 100.0f;

  // opacity arg is ignored when a per-pixel mask is given (passed 0)
  run_mode_check(&k_lab, a, b, 0.0f, 0x18 /*NORMAL2*/, 1.0f, 0,
                 _ref_lab_adapter, 5e-4f, "lab/ppmask", mask);
  run_mode_check(&k_lab, a, b, 0.0f, 0x04 /*MULTIPLY*/, 1.0f, 0,
                 _ref_lab_adapter, 5e-4f, "lab/ppmask-mul", mask);

  free(a); free(b); free(mask);
}

// non-equal rois: in_a is larger and read at an (offx, offy) offset
static void test_roi_offsets(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  const int IW = TW + 16, IH = TH + 11, DX = 9, DY = 7;
  const size_t IN_N = (size_t)IW * IH;

  s_rng = 0x5eed;
  v4 *a = malloc(IN_N * sizeof(v4)), *b = malloc(TN * sizeof(v4));
  v4 *out = malloc(TN * sizeof(v4));
  float *mask = malloc(TN * sizeof(float));
  assert_non_null(a); assert_non_null(b); assert_non_null(out); assert_non_null(mask);
  fill_rgb(a, IN_N); fill_rgb(b, TN);
  for(size_t i = 0; i < TN; i++) mask[i] = 0.6f;

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  dt_vk_mem_t *da = NULL, *db = NULL, *dm = NULL;
  if(!rc)
  {
    da = dt_vulkan_alloc_buffer(dev, IN_N * sizeof(v4));
    db = dt_vulkan_alloc_buffer(dev, TN * sizeof(v4));
    dm = dt_vulkan_alloc_buffer(dev, TN * sizeof(float));
    if(!da || !db || !dm) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, da, a, IN_N * sizeof(v4));
  if(!rc) rc = dt_vulkan_write_to_device(dev, db, b, TN * sizeof(v4));
  if(!rc) rc = dt_vulkan_write_to_device(dev, dm, mask, TN * sizeof(float));
  if(!rc)
  {
    const pc_blend_t pc = { TW, TH, IW, 0x18 /*NORMAL2*/, 1.0f, DX, DY, 0 };
    dt_vk_mem_t *bufs[3] = { da, db, dm };
    rc = dt_vulkan_dispatch_n(&k_jzczhz, bufs, 3, TW, TH, &pc, sizeof(pc));
  }
  if(!rc) rc = dt_vulkan_read_from_device(dev, out, db, TN * sizeof(v4));
  if(da) dt_vulkan_free_buffer(dev, da);
  if(db) dt_vulkan_free_buffer(dev, db);
  if(dm) dt_vulkan_free_buffer(dev, dm);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  assert_int_equal(rc, 0);
  for(int y = 0; y < TH; y++)
    for(int x = 0; x < TW; x++)
    {
      const v4 r = ref_blend_jzczhz(a[(y + DY) * IW + (x + DX)], b[y * TW + x],
                                    0.6f, 0x18, 1.0f, 0);
      const v4 g = out[y * TW + x];
      assert_float_equal(g.x, r.x, 1e-6);
      assert_float_equal(g.y, r.y, 1e-6);
      assert_float_equal(g.z, r.z, 1e-6);
      assert_float_equal(g.w, r.w, 1e-6);
    }

  free(a); free(b); free(out); free(mask);
}

/*
 * function-level tests: dt_develop_blend_process_vk with a
 * scaffolded piece
 */

static void _free_align_notify(gpointer p) { dt_free_align(p); }

// the scaffolded module reports blending support (and crucially not
// IOP_FLAGS_NO_MASKS), which the ME.4 drawn-mask gate queries
static int _blendtest_flags(void) { return IOP_FLAGS_SUPPORTS_BLENDING; }

typedef struct blend_scaffold_t
{
  dt_dev_pixelpipe_t pipe;
  dt_iop_module_t module;
  dt_dev_pixelpipe_iop_t piece;
  dt_develop_blend_params_t params;
} blend_scaffold_t;

static void scaffold_init(blend_scaffold_t *s,
                          dt_develop_blend_colorspace_t csp,
                          uint32_t blend_mode, float opacity_percent)
{
  memset(s, 0, sizeof(*s));
  s->pipe.mask_display = DT_DEV_PIXELPIPE_DISPLAY_NONE;
  s->pipe.store_all_raster_masks = FALSE;
  g_strlcpy(s->module.op, "blendtest", sizeof(s->module.op));
  s->module.flags = _blendtest_flags;
  s->module.raster_mask.source.users = g_hash_table_new(g_direct_hash, g_direct_equal);
  s->piece.pipe = &s->pipe;
  s->piece.module = &s->module;
  s->piece.colors = 4;
  s->piece.raster_masks = g_hash_table_new_full(g_direct_hash, g_direct_equal,
                                                NULL, _free_align_notify);
  s->params.mask_mode = DEVELOP_MASK_ENABLED;
  s->params.blend_cst = csp;
  s->params.blend_mode = blend_mode;
  s->params.blend_parameter = 0.0f; // exp2f(0) == 1
  s->params.opacity = opacity_percent;
  s->piece.blendop_data = &s->params;
}

static void scaffold_cleanup(blend_scaffold_t *s)
{
  g_hash_table_destroy(s->piece.raster_masks);
  g_hash_table_destroy(s->module.raster_mask.source.users);
}

static void test_process_vk_uniform_blend(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  blend_scaffold_t sc;
  scaffold_init(&sc, DEVELOP_BLEND_CS_RGB_SCENE, 0x18 /*NORMAL2*/, 60.0f);

  s_rng = 0x1234;
  v4 *a = malloc(TN * sizeof(v4)), *b = malloc(TN * sizeof(v4)), *out = malloc(TN * sizeof(v4));
  assert_non_null(a); assert_non_null(b); assert_non_null(out);
  fill_rgb(a, TN); fill_rgb(b, TN);

  const dt_iop_roi_t roi = { 0, 0, TW, TH, 1.0f };

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  gboolean blended = FALSE;
  dt_vk_mem_t *da = NULL, *db = NULL;
  if(!rc)
  {
    da = dt_vulkan_alloc_buffer(dev, TN * sizeof(v4));
    db = dt_vulkan_alloc_buffer(dev, TN * sizeof(v4));
    if(!da || !db) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, da, a, TN * sizeof(v4));
  if(!rc) rc = dt_vulkan_write_to_device(dev, db, b, TN * sizeof(v4));
  if(!rc)
  {
    blended = dt_develop_blend_process_vk(&sc.module, &sc.piece, da, db,
                                          &roi, &roi, IOP_CS_RGB, IOP_CS_RGB, NULL);
    if(blended) rc = dt_vulkan_read_from_device(dev, out, db, TN * sizeof(v4));
  }
  if(da) dt_vulkan_free_buffer(dev, da);
  if(db) dt_vulkan_free_buffer(dev, db);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  assert_int_equal(rc, 0);
  assert_true(blended);
  for(size_t i = 0; i < TN; i++)
  {
    const v4 r = ref_blend_jzczhz(a[i], b[i], 0.6f, 0x18, 1.0f, 0);
    assert_float_equal(out[i].x, r.x, 1e-6);
    assert_float_equal(out[i].y, r.y, 1e-6);
    assert_float_equal(out[i].z, r.z, 1e-6);
    assert_float_equal(out[i].w, r.w, 1e-6);
  }

  scaffold_cleanup(&sc);
  free(a); free(b); free(out);
}

// ME.4: a drawn mask (DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK) now runs
// on device instead of forcing the CPU blend. With no form defined the
// build fills 1.0 and applies the global opacity, so the result must
// equal a uniform blend at that opacity — this exercises the new gate,
// the host mask build, the upload and the apply end to end. (A varying
// drawn form reaches the same code path; its per-pixel correctness is
// covered by test_per_pixel_mask + the reused dt_masks build.)
static void test_process_vk_drawn_mask(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  blend_scaffold_t sc;
  scaffold_init(&sc, DEVELOP_BLEND_CS_RGB_SCENE, 0x18 /*NORMAL2*/, 60.0f);
  sc.params.mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK; // drawn

  s_rng = 0x1234;
  v4 *a = malloc(TN * sizeof(v4)), *b = malloc(TN * sizeof(v4)), *out = malloc(TN * sizeof(v4));
  assert_non_null(a); assert_non_null(b); assert_non_null(out);
  fill_rgb(a, TN); fill_rgb(b, TN);

  const dt_iop_roi_t roi = { 0, 0, TW, TH, 1.0f };

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  gboolean blended = FALSE;
  dt_vk_mem_t *da = NULL, *db = NULL;
  if(!rc)
  {
    da = dt_vulkan_alloc_buffer(dev, TN * sizeof(v4));
    db = dt_vulkan_alloc_buffer(dev, TN * sizeof(v4));
    if(!da || !db) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, da, a, TN * sizeof(v4));
  if(!rc) rc = dt_vulkan_write_to_device(dev, db, b, TN * sizeof(v4));
  if(!rc)
  {
    blended = dt_develop_blend_process_vk(&sc.module, &sc.piece, da, db,
                                          &roi, &roi, IOP_CS_RGB, IOP_CS_RGB, NULL);
    if(blended) rc = dt_vulkan_read_from_device(dev, out, db, TN * sizeof(v4));
  }
  if(da) dt_vulkan_free_buffer(dev, da);
  if(db) dt_vulkan_free_buffer(dev, db);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  assert_int_equal(rc, 0);
  assert_true(blended);  // the drawn mask was accepted and ran on device
  for(size_t i = 0; i < TN; i++)
  {
    const v4 r = ref_blend_jzczhz(a[i], b[i], 0.6f, 0x18, 1.0f, 0);
    assert_float_equal(out[i].x, r.x, 1e-6);
    assert_float_equal(out[i].y, r.y, 1e-6);
    assert_float_equal(out[i].z, r.z, 1e-6);
    assert_float_equal(out[i].w, r.w, 1e-6);
  }

  scaffold_cleanup(&sc);
  free(a); free(b); free(out);
}

// sRGB D50 matrices (linear TRC), matching dt_vulkan_common.h — both
// the row-major matrices (GPU) and the transposed copies (CPU path)
static void make_srgb_profile(dt_iop_order_iccprofile_info_t *p)
{
  static const float SRGB_TO_XYZ[9] = {
    0.4360747f, 0.3850649f, 0.1430804f,
    0.2225045f, 0.7168786f, 0.0606169f,
    0.0139322f, 0.0971045f, 0.7141733f };
  static const float XYZ_TO_SRGB[9] = {
     3.1338561f, -1.6168667f, -0.4906146f,
    -0.9787684f,  1.9161415f,  0.0334540f,
     0.0719453f, -0.2289914f,  1.4052427f };
  dt_ioppr_init_profile_info(p, 0);
  p->type = DT_COLORSPACE_SRGB;
  g_strlcpy(p->filename, "srgb-test", sizeof(p->filename));
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
    {
      p->matrix_in[i][j]  = SRGB_TO_XYZ[i * 3 + j];
      p->matrix_out[i][j] = XYZ_TO_SRGB[i * 3 + j];
      p->matrix_in_transposed[j][i]  = SRGB_TO_XYZ[i * 3 + j];
      p->matrix_out_transposed[j][i] = XYZ_TO_SRGB[i * 3 + j];
    }
  p->nonlinearlut = 0;
}

// M2 glue: the buffers are NOT in the blend colorspace, so the device
// must run the _transform_for_blend step itself — convert both into
// temporaries, blend there, land the blend-space result in dev_out and
// report it via blended_cst. Reference: darktable's own CPU colorspace
// transform on host copies + the independent blend reference.
static void run_blend_glue_case(const dt_develop_blend_colorspace_t bcs,
                                const dt_iop_colorspace_type_t buf_cst,
                                const dt_iop_colorspace_type_t want_cst,
                                ref_fn_t ref, const uint32_t mode,
                                const float eps, const char *tag)
{
  blend_scaffold_t sc;
  scaffold_init(&sc, bcs, mode, 70.0f);
  dt_iop_order_iccprofile_info_t prof;
  make_srgb_profile(&prof);
  sc.pipe.work_profile_info = &prof;

  s_rng = 0x9a9a;
  const size_t bytes = TN * sizeof(v4);
  v4 *a = malloc(bytes), *b = malloc(bytes), *out = malloc(bytes);
  v4 *ra = dt_alloc_aligned(bytes), *rb = dt_alloc_aligned(bytes);
  assert_non_null(a); assert_non_null(b); assert_non_null(out);
  assert_non_null(ra); assert_non_null(rb);
  if(buf_cst == IOP_CS_LAB) { fill_lab(a, TN); fill_lab(b, TN); }
  else
    for(size_t i = 0; i < TN; i++)
    {
      a[i] = (v4){ frand(0.0f, 1.0f), frand(0.0f, 1.0f), frand(0.0f, 1.0f), frand(0.0f, 1.0f) };
      b[i] = (v4){ frand(0.0f, 1.0f), frand(0.0f, 1.0f), frand(0.0f, 1.0f), frand(0.0f, 1.0f) };
    }

  const dt_iop_roi_t roi = { 0, 0, TW, TH, 1.0f };
  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  gboolean blended = FALSE;
  dt_iop_colorspace_type_t bcst = IOP_CS_NONE;
  dt_vk_mem_t *da = NULL, *db = NULL;
  if(!rc)
  {
    da = dt_vulkan_alloc_buffer(dev, bytes);
    db = dt_vulkan_alloc_buffer(dev, bytes);
    if(!da || !db) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, da, a, bytes);
  if(!rc) rc = dt_vulkan_write_to_device(dev, db, b, bytes);
  if(!rc)
  {
    blended = dt_develop_blend_process_vk(&sc.module, &sc.piece, da, db,
                                          &roi, &roi, buf_cst, buf_cst, &bcst);
    if(blended) rc = dt_vulkan_read_from_device(dev, out, db, bytes);
  }
  if(da) dt_vulkan_free_buffer(dev, da);
  if(db) dt_vulkan_free_buffer(dev, db);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  // CPU reference: transform host copies with darktable's own function
  // (exactly what _transform_for_blend does), then blend-reference
  dt_iop_colorspace_type_t c1 = buf_cst, c2 = buf_cst;
  dt_ioppr_transform_image_colorspace(&sc.module, (const float *)a, (float *)ra,
                                      TW, TH, buf_cst, want_cst, &c1, &prof);
  dt_ioppr_transform_image_colorspace(&sc.module, (const float *)b, (float *)rb,
                                      TW, TH, buf_cst, want_cst, &c2, &prof);

#define GLUE_ERR(g, r) (fabsf((g) - (r)) / fmaxf(1.0f, fabsf(r)))
  float maxerr = 0.0f;
  size_t worst = 0;
  if(!rc && blended && c1 == want_cst && c2 == want_cst)
    for(size_t i = 0; i < TN; i++)
    {
      const v4 r = ref(ra[i], rb[i], 0.7f, mode, 1.0f, 0);
      const float e = fmaxf(fmaxf(GLUE_ERR(out[i].x, r.x), GLUE_ERR(out[i].y, r.y)),
                            GLUE_ERR(out[i].z, r.z));
      if(e > maxerr) { maxerr = e; worst = i; }
    }
#undef GLUE_ERR

  scaffold_cleanup(&sc);
  dt_ioppr_cleanup_profile_info(&prof);
  free(a); free(b); free(out);
  dt_free_align(ra); dt_free_align(rb);

  assert_int_equal(rc, 0);
  assert_true(blended);
  assert_int_equal(bcst, want_cst);
  assert_int_equal(c1, want_cst);
  assert_int_equal(c2, want_cst);
  if(maxerr > eps)
    fail_msg("%s: max scaled error %g exceeds eps %g at px %zu",
             tag, (double)maxerr, (double)eps, worst);
}

static void test_process_vk_blend_glue_rgb_to_lab(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  // RGB buffers, Lab blend space (e.g. an RGB module blending in LAB)
  run_blend_glue_case(DEVELOP_BLEND_CS_LAB, IOP_CS_RGB, IOP_CS_LAB,
                      _ref_lab_adapter, 0x18 /*NORMAL2*/, 1e-3f,
                      "glue rgb->lab");
}

static void test_process_vk_blend_glue_lab_to_rgb(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  // Lab buffers, scene-RGB blend space (jzczhz kernel after Lab->RGB)
  run_blend_glue_case(DEVELOP_BLEND_CS_RGB_SCENE, IOP_CS_LAB, IOP_CS_RGB,
                      ref_blend_jzczhz, 0x18 /*NORMAL2*/, 1e-3f,
                      "glue lab->rgb");
}

static void test_process_vk_subset_gates(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  const dt_iop_roi_t roi = { 0, 0, TW, TH, 1.0f };
  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  gboolean gate_mask = TRUE, gate_cs = TRUE, gate_raw = TRUE, gate_ch = TRUE;
  gboolean gate_drawn_postop = TRUE;
  dt_vk_mem_t *da = NULL, *db = NULL;
  if(!rc)
  {
    da = dt_vulkan_alloc_buffer(dev, TN * sizeof(v4));
    db = dt_vulkan_alloc_buffer(dev, TN * sizeof(v4));
    if(!da || !db) rc = -101;
  }
  if(!rc)
  {
    blend_scaffold_t sc;

    // parametric masks still refuse: their build reads pixel data (the
    // device blendif port is future work). Drawn masks are now accepted
    // on device — see test_process_vk_drawn_mask.
    scaffold_init(&sc, DEVELOP_BLEND_CS_RGB_SCENE, 0x18, 60.0f);
    sc.params.mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_CONDITIONAL;
    gate_mask = dt_develop_blend_process_vk(&sc.module, &sc.piece, da, db,
                                            &roi, &roi, IOP_CS_RGB, IOP_CS_RGB, NULL);
    scaffold_cleanup(&sc);

    // a drawn mask with a post-op (blur) still refuses — the on-device
    // build only covers the pixel-free, post-op-free case
    scaffold_init(&sc, DEVELOP_BLEND_CS_RGB_SCENE, 0x18, 60.0f);
    sc.params.mask_mode = DEVELOP_MASK_ENABLED | DEVELOP_MASK_MASK;
    sc.params.blur_radius = 5.0f;
    gate_drawn_postop = dt_develop_blend_process_vk(&sc.module, &sc.piece, da, db,
                                                    &roi, &roi, IOP_CS_RGB, IOP_CS_RGB, NULL);
    scaffold_cleanup(&sc);

    // colorspace mismatch WITHOUT a work profile: the M2 glue
    // transforms need the profile matrix, so this must still refuse
    // (the with-profile case is test_process_vk_blend_glue_*)
    scaffold_init(&sc, DEVELOP_BLEND_CS_LAB, 0x18, 60.0f);
    gate_cs = dt_develop_blend_process_vk(&sc.module, &sc.piece, da, db,
                                          &roi, &roi, IOP_CS_RGB, IOP_CS_RGB, NULL);
    scaffold_cleanup(&sc);

    // RAW blend colorspace is not ported
    scaffold_init(&sc, DEVELOP_BLEND_CS_RAW, 0x18, 60.0f);
    gate_raw = dt_develop_blend_process_vk(&sc.module, &sc.piece, da, db,
                                           &roi, &roi, IOP_CS_RAW, IOP_CS_RAW, NULL);
    scaffold_cleanup(&sc);

    // 1-channel buffers are not ported
    scaffold_init(&sc, DEVELOP_BLEND_CS_RGB_SCENE, 0x18, 60.0f);
    sc.piece.colors = 1;
    gate_ch = dt_develop_blend_process_vk(&sc.module, &sc.piece, da, db,
                                          &roi, &roi, IOP_CS_RGB, IOP_CS_RGB, NULL);
    scaffold_cleanup(&sc);
  }
  if(da) dt_vulkan_free_buffer(dev, da);
  if(db) dt_vulkan_free_buffer(dev, db);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  assert_int_equal(rc, 0);
  assert_false(gate_mask);          // parametric refused
  assert_false(gate_drawn_postop);  // drawn + post-op refused
  assert_false(gate_cs);
  assert_false(gate_raw);
  assert_false(gate_ch);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_set_mask),
    cmocka_unit_test(test_jzczhz_modes),
    cmocka_unit_test(test_rgb_hsl_modes),
    cmocka_unit_test(test_lab_modes),
    cmocka_unit_test(test_per_pixel_mask),
    cmocka_unit_test(test_roi_offsets),
    cmocka_unit_test(test_process_vk_uniform_blend),
    cmocka_unit_test(test_process_vk_drawn_mask),
    cmocka_unit_test(test_process_vk_blend_glue_rgb_to_lab),
    cmocka_unit_test(test_process_vk_blend_glue_lab_to_rgb),
    cmocka_unit_test(test_process_vk_subset_gates),
  };
  return cmocka_run_group_tests(tests, group_setup, group_teardown);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
