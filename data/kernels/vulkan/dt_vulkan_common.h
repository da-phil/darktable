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
