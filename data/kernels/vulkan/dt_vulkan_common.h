/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Shared helpers for Vulkan-targeted IOP kernels.

    Convention: every kernel here operates on flat RGBA float storage
    buffers (the pixelpipe stages cl_mem / float* inputs through
    dt_vulkan_write_to_device before dispatch; see
    src/develop/pixelpipe_hb.c). The standard signature is:

        kernel void NAME(global const float4 *in,
                         global float4 *out,
                         const int width,
                         const int height,
                         ... scalar params ...);

    Push constants carry width / height plus the module-specific
    scalars. Storage buffers carry input and output (and constant
    tables when needed, on subsequent bindings).
*/

#pragma once

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// Branchless clamp helpers.
static inline float clampf(const float x, const float lo, const float hi)
{
  return fmin(fmax(x, lo), hi);
}

// Match the clipf helper used by data/kernels/common.h so kernel
// ports read 1:1 against their OpenCL counterparts.
static inline float clipf(const float a)
{
  return clamp(a, 0.0f, 1.0f);
}

static inline float4 clampf4(const float4 v, const float lo, const float hi)
{
  return (float4)(clampf(v.x, lo, hi),
                  clampf(v.y, lo, hi),
                  clampf(v.z, lo, hi),
                  v.w);
}

// Linear-index helper.
static inline int idx2d(const int x, const int y, const int width)
{
  return y * width + x;
}

// Bayer filter-color-at helper. Matches data/kernels/common.h::FC
// byte-for-byte; the `filters` bitmask comes from the camera RAW
// metadata. Returns one of {0=R, 1=G, 2=B}.
static inline int vk_FC(const int row, const int col, const uint filters)
{
  return (filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1)) & 3u;
}

// X-Trans pattern lookup. Matches FCxtrans in common.h. The pattern
// is a 6x6 byte matrix; we pass it as a flat 36-element uint storage
// buffer (one byte value per uint slot) to keep std430 layout simple.
// The +600 guard mirrors the OpenCL helper's handling of negative
// row/col values (used by demosaic algorithms that read past edges).
static inline int vk_FCxtrans(const int row, const int col,
                              global const uint *xtrans_flat)
{
  return (int)xtrans_flat[((row + 600) % 6) * 6 + ((col + 600) % 6)];
}

// Read with edge-clamped coords (saves writing the clamp+idx pair on
// every kernel that does spatial neighbourhoods).
static inline float4 read_clamped(global const float4 *buf,
                                  const int x, const int y,
                                  const int width, const int height)
{
  const int cx = clamp(x, 0, width  - 1);
  const int cy = clamp(y, 0, height - 1);
  return buf[cy * width + cx];
}

// Apply a 3x3 matrix to a float4 (alpha preserved).
static inline float4 matmul3(const float4 v, constant const float *m)
{
  // m is 9 floats laid out row-major: r0c0 r0c1 r0c2 / r1c0 ...
  float4 r;
  r.x = m[0] * v.x + m[1] * v.y + m[2] * v.z;
  r.y = m[3] * v.x + m[4] * v.y + m[5] * v.z;
  r.z = m[6] * v.x + m[7] * v.y + m[8] * v.z;
  r.w = v.w;
  return r;
}

// Same, but for a row-major 3x4 padded matrix (the format darktable
// stores ICC transform matrices in).
static inline float4 matmul3_padded(const float4 v, constant const float *m)
{
  float4 r;
  r.x = m[0]  * v.x + m[1]  * v.y + m[2]  * v.z;
  r.y = m[4]  * v.x + m[5]  * v.y + m[6]  * v.z;
  r.z = m[8]  * v.x + m[9]  * v.y + m[10] * v.z;
  r.w = v.w;
  return r;
}

// ---- 1D-LUT lookup (storage-buffer flavour) -------------------------
//
// data/kernels/common.h::lookup_unbounded takes a 256x256 image2d_t
// LUT (treating it as a 65536-entry flat 16-bit-indexed table) plus a
// 3-float coefficient blob describing the linear extrapolation tail.
// Our Vulkan kernels pass the LUT as a flat global float buffer of
// 65536 entries. The maths is line-for-line identical otherwise.
//
//   coeffs[0] < 0   -> identity (curve marked as linear)
//   coeffs[0] >= 0  -> for x in [0,1) integer lookup;
//                      for x >= 1 extrapolation: coeffs[1] * pow(x*coeffs[0], coeffs[2])

#define VK_LUT_SIZE 0x10000  // 65536 entries; matches the OpenCL 256x256 layout

// Plain LUT lookup, no extrapolation. The OpenCL counterpart
// (data/kernels/color_conversion.h::lookup) clamps x*65536 to the
// LUT index range and reads back; we do the same against a flat
// float buffer.
static inline float vk_lookup(global const float *lut, const float x)
{
  const int xi = clamp((int)(x * VK_LUT_SIZE), 0, VK_LUT_SIZE - 1);
  return lut[xi];
}

// Three scalars rather than `constant const float *coeffs` because
// clspv won't accept a private-pointer arg as `constant` and we want
// to pass these coefficients via push constants without going via a
// separate uniform buffer for each call site.
static inline float vk_lookup_unbounded(global const float *lut,
                                        const float x,
                                        const float c0,  // a[0]: <0 = identity
                                        const float c1,  // a[1]: out-of-range scale
                                        const float c2)  // a[2]: out-of-range exponent
{
  if(c0 < 0.0f) return x;
  if(x < 1.0f)
  {
    const int xi = clamp((int)(x * VK_LUT_SIZE), 0, VK_LUT_SIZE - 1);
    return lut[xi];
  }
  return c1 * pow(x * c0, c2);
}

// ---- colour-space conversions ---------------------------------------
//
// Selected pure-math helpers from data/kernels/colorspace.h. We carry
// our own copies rather than #include "colorspace.h" because the
// OpenCL header pulls in common.h which is full of image-sampler
// boilerplate that clspv won't compile under our buffer-only model.
// Keep these byte-for-byte equivalent to the OpenCL versions; if
// colorspace.h changes, mirror the change here.

static inline float4 vk_lab_f(float4 x)
{
  const float4 epsilon = 216.0f / 24389.0f;
  const float4 kappa   = 24389.0f / 27.0f;
  return (x > epsilon) ? cbrt(x) : (kappa * x + (float4)16.0f) / (float4)116.0f;
}

static inline float4 vk_lab_f_inv(float4 x)
{
  const float4 epsilon = 0.20689655172413796f; // cbrt(216/24389)
  const float4 kappa   = 24389.0f / 27.0f;
  return (x > epsilon) ? x * x * x : ((float4)116.0f * x - (float4)16.0f) / kappa;
}

static inline float4 vk_XYZ_to_Lab(float4 xyz)
{
  const float4 d50i = (float4)(1.0f / 0.9642f, 1.0f, 1.0f / 0.8249f, 1.0f);
  float4 f = vk_lab_f(xyz * d50i);
  float4 lab;
  lab.x = 116.0f * f.y - 16.0f;
  lab.y = 500.0f * (f.x - f.y);
  lab.z = 200.0f * (f.y - f.z);
  lab.w = xyz.w;
  return lab;
}

static inline float4 vk_Lab_to_XYZ(float4 Lab)
{
  const float4 d50 = (float4)(0.9642f, 1.0f, 0.8249f, 0.0f);
  float4 f;
  f.y = (Lab.x + 16.0f) / 116.0f;
  f.x = Lab.y / 500.0f + f.y;
  f.z = f.y - Lab.z / 200.0f;
  f.w = 0.0f;
  float4 xyz = d50 * vk_lab_f_inv(f);
  xyz.w = Lab.w;
  return xyz;
}

static inline float4 vk_XYZ_to_sRGB(float4 XYZ)
{
  float4 sRGB;
  sRGB.x =  3.1338561f * XYZ.x - 1.6168667f * XYZ.y - 0.4906146f * XYZ.z;
  sRGB.y = -0.9787684f * XYZ.x + 1.9161415f * XYZ.y + 0.0334540f * XYZ.z;
  sRGB.z =  0.0719453f * XYZ.x - 0.2289914f * XYZ.y + 1.4052427f * XYZ.z;
  sRGB.w = XYZ.w;
  return sRGB;
}

static inline float4 vk_sRGB_to_XYZ(float4 sRGB)
{
  float4 XYZ;
  XYZ.x = 0.4360747f * sRGB.x + 0.3850649f * sRGB.y + 0.1430804f * sRGB.z;
  XYZ.y = 0.2225045f * sRGB.x + 0.7168786f * sRGB.y + 0.0606169f * sRGB.z;
  XYZ.z = 0.0139322f * sRGB.x + 0.0971045f * sRGB.y + 0.7141733f * sRGB.z;
  XYZ.w = sRGB.w;
  return XYZ;
}

// ProPhoto RGB <-> XYZ (D50). Matrices match colorspace.h.
static inline float4 vk_prophotorgb_to_XYZ(float4 rgb)
{
  float4 XYZ = (float4)(0.0f, 0.0f, 0.0f, rgb.w);
  XYZ.x = 0.7976749f * rgb.x + 0.1351917f * rgb.y + 0.0313534f * rgb.z;
  XYZ.y = 0.2880402f * rgb.x + 0.7118741f * rgb.y + 0.0000857f * rgb.z;
  XYZ.z = 0.0000000f * rgb.x + 0.0000000f * rgb.y + 0.8252100f * rgb.z;
  return XYZ;
}

static inline float4 vk_XYZ_to_prophotorgb(float4 XYZ)
{
  float4 rgb = (float4)(0.0f, 0.0f, 0.0f, XYZ.w);
  rgb.x =  1.3459433f * XYZ.x - 0.2556075f * XYZ.y - 0.0511118f * XYZ.z;
  rgb.y = -0.5445989f * XYZ.x + 1.5081673f * XYZ.y + 0.0205351f * XYZ.z;
  rgb.z =  0.0000000f * XYZ.x + 0.0000000f * XYZ.y + 1.2118128f * XYZ.z;
  return rgb;
}

static inline float4 vk_prophotorgb_to_Lab(float4 rgb)
{
  return vk_XYZ_to_Lab(vk_prophotorgb_to_XYZ(rgb));
}

static inline float4 vk_Lab_to_prophotorgb(float4 lab)
{
  return vk_XYZ_to_prophotorgb(vk_Lab_to_XYZ(lab));
}

// ---- ICC profile info (storage-buffer flavour) ----------------------
//
// OpenCL passes profile info as a constant buffer + a 256×(256·6)
// image2d_t LUT and reads the LUT through image samplers. clspv
// doesn't surface samplers under our buffer-only kernel convention,
// so we pass the profile info as a flat storage buffer (struct below)
// and the LUT as a 1-D float buffer of 6·lutsize entries laid out as:
//
//   lut[0·lutsize .. 1·lutsize-1]  ->  in_R
//   lut[1·lutsize .. 2·lutsize-1]  ->  in_G
//   lut[2·lutsize .. 3·lutsize-1]  ->  in_B
//   lut[3·lutsize .. 4·lutsize-1]  ->  out_R
//   lut[4·lutsize .. 5·lutsize-1]  ->  out_G
//   lut[5·lutsize .. 6·lutsize-1]  ->  out_B
//
// This struct is byte-for-byte equivalent to
// dt_colorspaces_iccprofile_info_cl_t (156 bytes). The host-side
// helper dt_ioppr_build_iccprofile_params_vk fills both buffers and
// dt_ioppr_free_iccprofile_params_vk frees them.

typedef struct vk_dt_colorspaces_iccprofile_info_t
{
  float matrix_in[9];
  float matrix_out[9];
  int   lutsize;
  float unbounded_coeffs_in[3][3];
  float unbounded_coeffs_out[3][3];
  int   nonlinearlut;
  float grey;
} vk_dt_colorspaces_iccprofile_info_t;

// 1-D linear-interp lookup into the flat 6·lutsize LUT. Mirrors
// data/kernels/color_conversion.h::lerp_lookup_unbounded but reads
// from the flat buffer above instead of a 2-D image (the OpenCL
// helper splits a 65536-entry curve into 256×256 tiles to fit
// image2d_t limits; we don't have that constraint here).
static inline float vk_lerp_lookup_unbounded(global const float *lut,
                                             const float x,
                                             const float c0, const float c1, const float c2,
                                             const int n_lut, const int lutsize)
{
  if(c0 >= 0.0f)
  {
    if(x < 1.0f)
    {
      const float ft = clamp(x * (float)(lutsize - 1), 0.0f, (float)(lutsize - 1));
      const int   t  = (int)((ft < (float)(lutsize - 2)) ? ft : (float)(lutsize - 2));
      const float f  = ft - (float)t;
      const int   base = n_lut * lutsize;
      const float l1 = lut[base + t];
      const float l2 = lut[base + t + 1];
      return l1 * (1.0f - f) + l2 * f;
    }
    return c1 * pow(x * c0, c2);
  }
  return x;
}

static inline float4 vk_apply_trc_in(const float4 rgb_in,
                                     global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                     global const float *lut)
{
  const float R = vk_lerp_lookup_unbounded(lut, rgb_in.x,
                                            profile_info->unbounded_coeffs_in[0][0],
                                            profile_info->unbounded_coeffs_in[0][1],
                                            profile_info->unbounded_coeffs_in[0][2],
                                            0, profile_info->lutsize);
  const float G = vk_lerp_lookup_unbounded(lut, rgb_in.y,
                                            profile_info->unbounded_coeffs_in[1][0],
                                            profile_info->unbounded_coeffs_in[1][1],
                                            profile_info->unbounded_coeffs_in[1][2],
                                            1, profile_info->lutsize);
  const float B = vk_lerp_lookup_unbounded(lut, rgb_in.z,
                                            profile_info->unbounded_coeffs_in[2][0],
                                            profile_info->unbounded_coeffs_in[2][1],
                                            profile_info->unbounded_coeffs_in[2][2],
                                            2, profile_info->lutsize);
  return (float4)(R, G, B, rgb_in.w);
}

static inline float4 vk_apply_trc_out(const float4 rgb_in,
                                      global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                      global const float *lut)
{
  const float R = vk_lerp_lookup_unbounded(lut, rgb_in.x,
                                            profile_info->unbounded_coeffs_out[0][0],
                                            profile_info->unbounded_coeffs_out[0][1],
                                            profile_info->unbounded_coeffs_out[0][2],
                                            3, profile_info->lutsize);
  const float G = vk_lerp_lookup_unbounded(lut, rgb_in.y,
                                            profile_info->unbounded_coeffs_out[1][0],
                                            profile_info->unbounded_coeffs_out[1][1],
                                            profile_info->unbounded_coeffs_out[1][2],
                                            4, profile_info->lutsize);
  const float B = vk_lerp_lookup_unbounded(lut, rgb_in.z,
                                            profile_info->unbounded_coeffs_out[2][0],
                                            profile_info->unbounded_coeffs_out[2][1],
                                            profile_info->unbounded_coeffs_out[2][2],
                                            5, profile_info->lutsize);
  return (float4)(R, G, B, rgb_in.w);
}

static inline float vk_get_rgb_matrix_luminance(const float4 rgb,
                                                global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                                global const float *lut)
{
  // matrix is row-major 3x3 in 9 floats. Y-row is matrix[3..5].
  float luminance;
  if(profile_info->nonlinearlut)
  {
    const float4 linear_rgb = vk_apply_trc_in(rgb, profile_info, lut);
    luminance = profile_info->matrix_in[3] * linear_rgb.x
              + profile_info->matrix_in[4] * linear_rgb.y
              + profile_info->matrix_in[5] * linear_rgb.z;
  }
  else
  {
    luminance = profile_info->matrix_in[3] * rgb.x
              + profile_info->matrix_in[4] * rgb.y
              + profile_info->matrix_in[5] * rgb.z;
  }
  return luminance;
}

static inline float vk_dt_camera_rgb_luminance(const float4 rgb)
{
  // sRGB Y row, used as fallback when no work profile is available.
  const float4 coeffs = (float4)(0.2225045f, 0.7168786f, 0.0606169f, 0.0f);
  return dot(rgb, coeffs);
}

// dt_iop_rgb_norms_t mirror. Keep in sync with rgb_norms.h.
#define VK_RGB_NORM_NONE      0
#define VK_RGB_NORM_LUMINANCE 1
#define VK_RGB_NORM_MAX       2
#define VK_RGB_NORM_AVERAGE   3
#define VK_RGB_NORM_SUM       4
#define VK_RGB_NORM_NORM      5
#define VK_RGB_NORM_POWER     6

static inline float vk_dt_rgb_norm(const float4 in, const int norm, const int work_profile,
                                   global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                   global const float *lut)
{
  if(norm == VK_RGB_NORM_LUMINANCE)
    return (work_profile == 0) ? vk_dt_camera_rgb_luminance(in)
                               : vk_get_rgb_matrix_luminance(in, profile_info, lut);
  if(norm == VK_RGB_NORM_MAX)     return fmax(in.x, fmax(in.y, in.z));
  if(norm == VK_RGB_NORM_AVERAGE) return (in.x + in.y + in.z) / 3.0f;
  if(norm == VK_RGB_NORM_SUM)     return in.x + in.y + in.z;
  if(norm == VK_RGB_NORM_NORM)    return pow(in.x * in.x + in.y * in.y + in.z * in.z, 0.5f);
  if(norm == VK_RGB_NORM_POWER)
  {
    const float R = in.x * in.x;
    const float G = in.y * in.y;
    const float B = in.z * in.z;
    return (in.x * R + in.y * G + in.z * B) / (R + G + B);
  }
  return (in.x + in.y + in.z) / 3.0f;
}

// ---- atomic float add (CAS-loop flavour) ----------------------------
//
// Vulkan / SPIR-V 1.3 doesn't guarantee a native atomicAdd(float, float)
// on storage buffers; VK_EXT_shader_atomic_float lifts that but we
// can't rely on the extension on every target driver. The CAS loop
// on the uint reinterpretation of the float is the portable form
// (same pattern as data/kernels/common.h::atomic_add_f minus the
// NVIDIA-PTX fast path), and is what the bilateral splat kernel
// needs to accumulate contributions from many work-items into the
// same 3-D grid cell.
//
// `val` must point at storage-buffer (global) memory.

static inline void vk_atomic_add_f(global float *val, const float delta)
{
  global volatile uint *ival = (global volatile uint *)val;
  uint old_i = *ival;
  union { float f; uint i; } u;
  while(1)
  {
    u.i = old_i;
    u.f += delta;
    const uint witness = atomic_cmpxchg(ival, old_i, u.i);
    if(witness == old_i) break;
    old_i = witness;
  }
}

// RGB <-> HSL — needed by splittoning et al. Matches the OpenCL
// implementations in data/kernels/colorspace.h byte-for-byte.

static inline float4 vk_RGB_to_HSL(const float4 RGB)
{
  float H = 0.0f, S = 0.0f;
  const float R = RGB.x, G = RGB.y, B = RGB.z;
  const float var_Min = fmin(R, fmin(G, B));
  const float var_Max = fmax(R, fmax(G, B));
  const float del_Max = var_Max - var_Min;
  const float L = (var_Max + var_Min) * 0.5f;

  if(fabs(var_Max) > 1e-6f && fabs(del_Max) > 1e-6f)
  {
    S = (L < 0.5f)
        ? del_Max / (var_Max + var_Min)
        : del_Max / (2.0f - var_Max - var_Min);

    const float del_R = (((var_Max - R) / 6.0f) + (del_Max / 2.0f)) / del_Max;
    const float del_G = (((var_Max - G) / 6.0f) + (del_Max / 2.0f)) / del_Max;
    const float del_B = (((var_Max - B) / 6.0f) + (del_Max / 2.0f)) / del_Max;

    if(R == var_Max)      H = del_B - del_G;
    else if(G == var_Max) H = (1.0f / 3.0f) + del_R - del_B;
    else if(B == var_Max) H = (2.0f / 3.0f) + del_G - del_R;

    if(H < 0.0f) H += 1.0f;
    if(H > 1.0f) H -= 1.0f;
  }
  return (float4)(H, S, L, RGB.w);
}

static inline float vk_hue_to_rgb(const float v1, const float v2, float vH)
{
  if(vH < 0.0f) vH += 1.0f;
  if(vH > 1.0f) vH -= 1.0f;
  if(6.0f * vH < 1.0f) return v1 + (v2 - v1) * 6.0f * vH;
  if(2.0f * vH < 1.0f) return v2;
  if(3.0f * vH < 2.0f) return v1 + (v2 - v1) * ((2.0f / 3.0f) - vH) * 6.0f;
  return v1;
}

static inline float4 vk_HSL_to_RGB(const float4 HSL)
{
  const float H = HSL.x, S = HSL.y, L = HSL.z;
  if(S < 1e-6f)
    return (float4)(L, L, L, HSL.w);

  const float var_2 = (L < 0.5f) ? L * (1.0f + S) : (L + S) - (S * L);
  const float var_1 = 2.0f * L - var_2;
  return (float4)(vk_hue_to_rgb(var_1, var_2, H + 1.0f / 3.0f),
                  vk_hue_to_rgb(var_1, var_2, H),
                  vk_hue_to_rgb(var_1, var_2, H - 1.0f / 3.0f),
                  HSL.w);
}

// RGB <-> HSV — mirrors RGB_2_HSV / HSV_2_RGB in colorspace.h. agx
// uses these for the optional hue-restore step.
static inline float4 vk_RGB_to_HSV(const float4 RGB)
{
  float4 HSV;
  const float minv = fmin(RGB.x, fmin(RGB.y, RGB.z));
  const float maxv = fmax(RGB.x, fmax(RGB.y, RGB.z));
  const float delta = maxv - minv;

  HSV.z = maxv;
  HSV.w = RGB.w;

  if(fabs(maxv) > 1e-6f && fabs(delta) > 1e-6f)
  {
    HSV.y = delta / maxv;
  }
  else
  {
    HSV.x = 0.0f;
    HSV.y = 0.0f;
    return HSV;
  }

  if(RGB.x == maxv)      HSV.x = (RGB.y - RGB.z) / delta;
  else if(RGB.y == maxv) HSV.x = 2.0f + (RGB.z - RGB.x) / delta;
  else                   HSV.x = 4.0f + (RGB.x - RGB.y) / delta;

  HSV.x /= 6.0f;
  HSV.x -= floor(HSV.x);
  return HSV;
}

static inline float4 vk_HSV_to_RGB(const float4 HSV)
{
  if(fabs(HSV.y) < 1e-6f)
    return (float4)(HSV.z, HSV.z, HSV.z, HSV.w);

  const int   i = (int)floor(6.0f * HSV.x);
  const float v = HSV.z;
  const float w = HSV.w;
  const float p = v * (1.0f - HSV.y);
  const float q = v * (1.0f - HSV.y * (6.0f * HSV.x - (float)i));
  const float t = v * (1.0f - HSV.y * (1.0f - (6.0f * HSV.x - (float)i)));

  switch(i)
  {
    case 0:  return (float4)(v, t, p, w);
    case 1:  return (float4)(q, v, p, w);
    case 2:  return (float4)(p, v, t, w);
    case 3:  return (float4)(p, q, v, w);
    case 4:  return (float4)(t, p, v, w);
    default: return (float4)(v, p, q, w);  // case 5 + wrap
  }
}
