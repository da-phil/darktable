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

// Kahan-compensated running sum, lvalue form (matches common.h::Kahan_sum).
// Used by the guided-filter box-mean kernels.
#ifndef vk_kahan_sum
#define vk_kahan_sum(m, c, add)       \
  {                                   \
    const float _kt1 = (add) - (c);   \
    const float _kt2 = (m) + _kt1;    \
    (c) = (_kt2 - (m)) - _kt1;        \
    (m) = _kt2;                       \
  }
#endif

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

// NOTE on clspv: the float->uint pointer reinterpret below is the
// portable OpenCL idiom for a float atomic-add (OpenCL C 1.2 has no
// native float atomics), and is exactly what common.h::atomic_add_f
// does. clspv (at least the LLVM-23-era build we tested) cannot fully
// lower it: it keeps `ival` pointing at the buffer's float element
// type, so the uint atomics emit SPIR-V that spirv-val rejects with a
// pointer/result type mismatch. The two kernels that use this
// (bilateral_splat, colorreconstruction_splat) therefore ship via
// their glslang .comp twins, which express the same accumulation with
// GLSL's atomicCompSwap over a uint-typed buffer view. Fully fixing
// the clspv path would mean binding those splat buffers as uint* at
// the kernel signature and bitcasting values (not the pointer) — a
// host-side change tracked separately. The body below still matches
// common.h byte-for-byte so the CPU/OpenCL semantics are identical.
static inline void vk_atomic_add_f(global float *val, const float delta)
{
  global volatile uint *ival = (global volatile uint *)val;
  union { float f; uint i; } old_val, new_val;
  do
  {
    // atomic read (atomic_add of 0) rather than a plain load: OpenCL's
    // relaxed global-memory consistency means `*ival` may be stale.
    old_val.i = atomic_add(ival, 0u);
    new_val.f = old_val.f + delta;
  }
  while(atomic_cmpxchg(ival, old_val.i, new_val.i) != old_val.i);
}

// Float atomic min/max via the same reinterpret-CAS idiom (no native
// float atomics under our buffer-only kernel convention). The early
// return means a lane that isn't extending the extremum touches memory
// only through the initial atomic read, so uncontended pixels are
// cheap; contention degrades to a retry loop, not a wrong result.
static inline void vk_atomic_min_f(global float *val, const float v)
{
  global volatile uint *ival = (global volatile uint *)val;
  union { float f; uint i; } old_val, new_val;
  do
  {
    old_val.i = atomic_add(ival, 0u);
    if(old_val.f <= v) return;
    new_val.f = v;
  }
  while(atomic_cmpxchg(ival, old_val.i, new_val.i) != old_val.i);
}

static inline void vk_atomic_max_f(global float *val, const float v)
{
  global volatile uint *ival = (global volatile uint *)val;
  union { float f; uint i; } old_val, new_val;
  do
  {
    old_val.i = atomic_add(ival, 0u);
    if(old_val.f >= v) return;
    new_val.f = v;
  }
  while(atomic_cmpxchg(ival, old_val.i, new_val.i) != old_val.i);
}

// Fast 2^-x approximation. Matches data/kernels/common.h::fast_mexp2f
// byte-for-byte (the union bit-pun is clspv-safe — same idiom as
// vk_atomic_add_f / colorchecker's fastlog2). Used by nlmeans.
static inline float vk_fast_mexp2f(const float x)
{
  const float i1 = (float)0x3f800000u; // 2^0
  const float i2 = (float)0x3f000000u; // 2^-1
  const float k0 = i1 + x * (i2 - i1);
  union { float f; uint i; } k;
  k.i = (k0 >= (float)0x800000u) ? (uint)k0 : 0u;
  return k.f;
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

// dt UCS color appearance model — used by colorharmonizer (and any
// future module that wants the JCH perceptual triple). Mirrors the
// definitions in data/kernels/colorspace.h byte-for-byte.
#define DT_UCS_L_STAR_RANGE         2.098883786377f
#define DT_UCS_L_STAR_UPPER_LIMIT   2.09885f
#define DT_UCS_Y_UPPER_LIMIT        1e8f

static inline float4 vk_D65_XYZ_to_xyY(const float4 sXYZ)
{
  float4 XYZ = fmax(0.0f, sXYZ);
  float4 xyY;
  const float sum = XYZ.x + XYZ.y + XYZ.z;
  if(sum > 0.0f) { xyY.x = XYZ.x / sum; xyY.y = XYZ.y / sum; }
  else           { xyY.x = 0.31271f;   xyY.y = 0.32902f; }
  xyY.z = XYZ.y;
  xyY.w = sXYZ.w;
  return xyY;
}

static inline float4 vk_xyY_to_XYZ(const float4 xyY)
{
  float4 XYZ = (float4)(0.0f, 0.0f, 0.0f, xyY.w);
  if(xyY.y != 0.0f)
  {
    XYZ.x = xyY.z * xyY.x / xyY.y;
    XYZ.y = xyY.z;
    XYZ.z = xyY.z * (1.0f - xyY.x - xyY.y) / xyY.y;
  }
  return XYZ;
}

static inline float vk_Y_to_dt_UCS_L_star(const float Y)
{
  const float Y_hat = pow(Y, 0.631651345306265f);
  return DT_UCS_L_STAR_RANGE * Y_hat / (Y_hat + 1.12426773749357f);
}

static inline float vk_dt_UCS_L_star_to_Y(const float L_star)
{
  return pow((1.12426773749357f * L_star / (DT_UCS_L_STAR_RANGE - L_star)),
             1.5831518565279648f);
}

static inline float2 vk_xyY_to_dt_UCS_UV(const float4 xyY)
{
  const float4 x_factors = (float4)(-0.783941002840055f,  0.745273540913283f, 0.318707282433486f, 0.0f);
  const float4 y_factors = (float4)( 0.277512987809202f, -0.205375866083878f, 2.16743692732158f,  0.0f);
  const float4 offsets   = (float4)( 0.153836578598858f, -0.165478376301988f, 0.291320554395942f, 0.0f);

  float4 UVD = x_factors * xyY.x + y_factors * xyY.y + offsets;
  const float div = (UVD.z >= 0.0f) ? fmax(FLT_MIN, UVD.z) : fmin(-FLT_MIN, UVD.z);
  UVD.x /= div;
  UVD.y /= div;

  const float2 factors     = (float2)(1.39656225667f, 1.4513954287f);
  const float2 half_values = (float2)(1.49217352929f, 1.52488637914f);
  const float2 UV_star = (float2)(
    factors.x * UVD.x / (fabs(UVD.x) + half_values.x),
    factors.y * UVD.y / (fabs(UVD.y) + half_values.y));
  // Final 2x2 matrix product giving UV_star_prime.
  return (float2)(
    -1.124983854323892f * UV_star.x - 0.980483721769325f * UV_star.y,
     1.86323315098672f  * UV_star.x + 1.971853092390862f * UV_star.y);
}

static inline float4 vk_xyY_to_dt_UCS_JCH(const float4 xyY, const float L_white)
{
  const float2 UV_star_prime = vk_xyY_to_dt_UCS_UV(xyY);

  const float L_star = vk_Y_to_dt_UCS_L_star(clamp(xyY.z, 0.0f, DT_UCS_Y_UPPER_LIMIT));
  const float M2 = UV_star_prime.x * UV_star_prime.x
                 + UV_star_prime.y * UV_star_prime.y;

  return (float4)(L_star / L_white,
                  15.932993652962535f * pow(L_star, 0.6523997524738018f)
                                      * pow(M2, 0.6007557017508491f) / L_white,
                  atan2(UV_star_prime.y, UV_star_prime.x),
                  0.0f);
}

static inline float4 vk_dt_UCS_JCH_to_xyY(const float4 JCH, const float L_white)
{
  const float L_star = clamp(JCH.x * L_white, 0.0f, DT_UCS_L_STAR_UPPER_LIMIT);
  const float M = L_star != 0.0f
    ? pow(JCH.y * L_white / (15.932993652962535f * pow(L_star, 0.6523997524738018f)),
          0.8322850678616855f)
    : 0.0f;

  const float U_star_prime = M * cos(JCH.z);
  const float V_star_prime = M * sin(JCH.z);

  const float2 UV_star = (float2)(
    -5.037522385190711f * U_star_prime - 2.504856328185843f * V_star_prime,
     4.760029407436461f * U_star_prime + 2.874012963239247f * V_star_prime);

  const float2 factors     = (float2)(1.39656225667f, 1.4513954287f);
  const float2 half_values = (float2)(1.49217352929f, 1.52488637914f);
  const float2 UV = (float2)(
    -half_values.x * UV_star.x / (fabs(UV_star.x) - factors.x),
    -half_values.y * UV_star.y / (fabs(UV_star.y) - factors.y));

  const float4 U_factors = (float4)( 0.167171472114775f,   -0.150959086409163f,    0.940254742367256f,  0.0f);
  const float4 V_factors = (float4)( 0.141299802443708f,   -0.155185060382272f,    1.000000000000000f,  0.0f);
  const float4 offsets   = (float4)(-0.00801531300850582f, -0.00843312433578007f, -0.0256325967652889f, 0.0f);

  const float4 xyD = U_factors * UV.x + V_factors * UV.y + offsets;
  const float div = (xyD.z >= 0.0f) ? fmax(FLT_MIN, xyD.z) : fmin(-FLT_MIN, xyD.z);
  return (float4)(xyD.x / div, xyD.y / div, vk_dt_UCS_L_star_to_Y(L_star), 0.0f);
}

// dt UCS brightness/colorfulness cylindrical forms — used by
// colorbalancergb's dt-UCS saturation branch. Mirror colorspace.h.
static inline float4 vk_dt_UCS_JCH_to_HSB(const float4 JCH)
{
  float4 HSB;
  HSB.z = JCH.x * (pow(JCH.y, 1.33654221029386f) + 1.0f);
  HSB.y = (HSB.z > 0.0f) ? JCH.y / HSB.z : 0.0f;
  HSB.x = JCH.z;
  HSB.w = 0.0f;
  return HSB;
}

static inline float4 vk_dt_UCS_HSB_to_JCH(const float4 HSB)
{
  float4 JCH;
  JCH.z = HSB.x;
  JCH.y = HSB.y * HSB.z;
  JCH.x = HSB.z / (pow(JCH.y, 1.33654221029386f) + 1.0f);
  JCH.w = 0.0f;
  return JCH;
}

static inline float4 vk_dt_UCS_JCH_to_HCB(const float4 JCH)
{
  float4 HCB;
  HCB.z = JCH.x * (pow(JCH.y, 1.33654221029386f) + 1.0f);
  HCB.y = JCH.y;
  HCB.x = JCH.z;
  HCB.w = 0.0f;
  return HCB;
}

static inline float4 vk_dt_UCS_HCB_to_JCH(const float4 HCB)
{
  float4 JCH;
  JCH.z = HCB.x;
  JCH.y = HCB.y;
  JCH.x = HCB.z / (pow(HCB.y, 1.33654221029386f) + 1.0f);
  JCH.w = 0.0f;
  return JCH;
}

// ---- CIE 2006 LMS / Filmlight Yrg-Ych / JzAzBz colour science -------
//
// Ported byte-for-byte from data/kernels/colorspace.h (matrix_dot rows
// inlined, dtcl_* -> std intrinsics). These back colorbalancergb and
// are reusable by the future filmicrgb / sigmoid Yrg ports. Keep the
// matrix coefficients identical to the OpenCL originals.

#ifndef DT_2PI_F
#define DT_2PI_F 6.28318530717958647692f
#endif
#ifndef LUT_ELEM
#define LUT_ELEM 512  // gamut LUT element count; matches data/kernels/common.h
#endif

static inline float vk_dt_fast_hypot(const float x, const float y)
{
  return sqrt(x * x + y * y);
}

static inline float4 vk_LMS_to_XYZ(const float4 LMS)
{
  return (float4)( 1.80794659f * LMS.x - 1.29971660f * LMS.y + 0.34785879f * LMS.z,
                   0.61783960f * LMS.x + 0.39595453f * LMS.y - 0.04104687f * LMS.z,
                  -0.12546960f * LMS.x + 0.20478038f * LMS.y + 1.74274183f * LMS.z,
                   LMS.w);
}

static inline float4 vk_gradingRGB_to_LMS(const float4 RGB)
{
  return (float4)(0.95f * RGB.x + 0.38f * RGB.y + 0.00f * RGB.z,
                  0.05f * RGB.x + 0.62f * RGB.y + 0.03f * RGB.z,
                  0.00f * RGB.x + 0.00f * RGB.y + 0.97f * RGB.z,
                  RGB.w);
}

static inline float4 vk_LMS_to_gradingRGB(const float4 LMS)
{
  return (float4)( 1.0877193f  * LMS.x - 0.66666667f * LMS.y + 0.02061856f * LMS.z,
                  -0.0877193f  * LMS.x + 1.66666667f * LMS.y - 0.05154639f * LMS.z,
                                                              1.03092784f * LMS.z,
                   LMS.w);
}

static inline float4 vk_LMS_to_Yrg(const float4 LMS)
{
  const float Y = 0.68990272f * LMS.x + 0.34832189f * LMS.y;
  const float a = LMS.x + LMS.y + LMS.z;
  const float4 lms = (a == 0.0f) ? (float4)(0.0f) : LMS / a;
  const float4 rgb = vk_LMS_to_gradingRGB(lms);
  return (float4)(Y, rgb.x, rgb.y, LMS.w);
}

static inline float4 vk_Yrg_to_LMS(const float4 Yrg)
{
  const float Y = Yrg.x;
  const float r = Yrg.y;
  const float g = Yrg.z;
  const float b = 1.0f - r - g;
  const float4 rgb = (float4)(r, g, b, 0.0f);
  const float4 lms = vk_gradingRGB_to_LMS(rgb);
  const float denom = (0.68990272f * lms.x + 0.34832189f * lms.y);
  const float a = (denom == 0.0f) ? 0.0f : Y / denom;
  return lms * a;
}

static inline float4 vk_Yrg_to_Ych(const float4 Yrg)
{
  const float Y = Yrg.x;
  const float r = Yrg.y - 0.21902143f;
  const float g = Yrg.z - 0.54371398f;
  const float c = vk_dt_fast_hypot(g, r);
  const float cos_h = c != 0.0f ? r / c : 1.0f;
  const float sin_h = c != 0.0f ? g / c : 0.0f;
  return (float4)(Y, c, cos_h, sin_h);
}

static inline float4 vk_Ych_to_Yrg(const float4 Ych)
{
  const float Y = Ych.x;
  const float c = Ych.y;
  const float cos_h = Ych.z;
  const float sin_h = Ych.w;
  const float r = c * cos_h + 0.21902143f;
  const float g = c * sin_h + 0.54371398f;
  return (float4)(Y, r, g, 0.0f);
}

static inline float4 vk_gamut_check_Yrg(float4 Ych)
{
  const float4 Yrg = vk_Ych_to_Yrg(Ych);
  const float D65_r = 0.21902143f;
  const float D65_g = 0.54371398f;
  float max_c = Ych.y;
  const float cos_h = Ych.z;
  const float sin_h = Ych.w;

  if(Yrg.y < 0.0f)
    max_c = fmin(-D65_r / cos_h, max_c);
  if(Yrg.z < 0.0f)
    max_c = fmin(-D65_g / sin_h, max_c);
  if(Yrg.y + Yrg.z > 1.0f)
    max_c = fmin((1.0f - D65_r - D65_g) / (cos_h + sin_h), max_c);

  Ych.y = max_c;
  return Ych;
}

static inline float4 vk_XYZ_to_JzAzBz(const float4 XYZ_D65)
{
  float4 temp1, temp2;
  // XYZ -> X'Y'Z
  temp1.x = 1.15f * XYZ_D65.x - 0.15f * XYZ_D65.z;
  temp1.y = 0.66f * XYZ_D65.y + 0.34f * XYZ_D65.x;
  temp1.z = XYZ_D65.z;
  temp1.w = 0.0f;
  // X'Y'Z -> LMS
  temp2.x =  0.41478972f * temp1.x + 0.579999f * temp1.y + 0.0146480f * temp1.z;
  temp2.y = -0.2015100f  * temp1.x + 1.120649f * temp1.y + 0.0531008f * temp1.z;
  temp2.z = -0.0166008f  * temp1.x + 0.264800f * temp1.y + 0.6684799f * temp1.z;
  temp2.w = 0.0f;
  // LMS -> L'M'S'
  temp2 = pow(fmax(temp2 / 10000.0f, 0.0f), (float4)(0.159301758f));
  temp2 = pow((0.8359375f + 18.8515625f * temp2) / (1.0f + 18.6875f * temp2), (float4)(134.034375f));
  // L'M'S' -> Izazbz
  temp1.x = 0.5f * temp2.x + 0.5f * temp2.y;
  temp1.y = 3.524000f * temp2.x - 4.066708f * temp2.y + 0.542708f * temp2.z;
  temp1.z = 0.199076f * temp2.x + 1.096799f * temp2.y - 1.295875f * temp2.z;
  // Iz -> Jz
  temp1.x = fmax(0.44f * temp1.x / (1.0f - 0.56f * temp1.x) - 1.6295499532821566e-11f, 0.0f);
  return temp1;
}

// XYZ D50 -> XYZ D65 Bradford adaptation, matching the CPU
// dt_XYZ_D50_2_XYZ_D65 (transposed Bradford matrix from Lindbloom).
static inline float4 vk_XYZ_D50_to_XYZ_D65(const float4 XYZ_D50)
{
  float4 out;
  out.x =  0.9555766f * XYZ_D50.x - 0.0230393f * XYZ_D50.y + 0.0631636f * XYZ_D50.z;
  out.y = -0.0282895f * XYZ_D50.x + 1.0099416f * XYZ_D50.y + 0.0210077f * XYZ_D50.z;
  out.z =  0.0122982f * XYZ_D50.x - 0.0204830f * XYZ_D50.y + 1.3299098f * XYZ_D50.z;
  out.w = XYZ_D50.w;
  return out;
}

// JzAzBz -> JzCzhz (polar), matching the CPU dt_JzAzBz_2_JzCzhz: hue
// in turns wrapped to [0,1), chroma = hypot(az, bz).
static inline float4 vk_JzAzBz_to_JzCzhz(const float4 JzAzBz)
{
  float H = atan2(JzAzBz.z, JzAzBz.y) / DT_2PI_F;
  H = (H >= 0.0f) ? H : 1.0f + H;
  return (float4)(JzAzBz.x, vk_dt_fast_hypot(JzAzBz.y, JzAzBz.z), H, JzAzBz.w);
}

static inline float4 vk_JzAzBz_2_XYZ(const float4 JzAzBz)
{
  const float b = 1.15f;
  const float g = 0.66f;
  const float c1 = 0.8359375f;
  const float c2 = 18.8515625f;
  const float c3 = 18.6875f;
  const float n_inv = 1.0f / 0.159301758f;
  const float p_inv = 1.0f / 134.034375f;
  const float d = -0.56f;
  const float d0 = 1.6295499532821566e-11f;

  float4 XYZ, LMS, IzAzBz;
  // Jz -> Iz
  IzAzBz = JzAzBz;
  IzAzBz.x += d0;
  IzAzBz.x = fmax(IzAzBz.x / (1.0f + d - d * IzAzBz.x), 0.0f);
  // IzAzBz -> L'M'S'
  LMS.x = 1.0f * IzAzBz.x + 0.1386050432715393f * IzAzBz.y + 0.0580473161561189f * IzAzBz.z;
  LMS.y = 1.0f * IzAzBz.x - 0.1386050432715393f * IzAzBz.y - 0.0580473161561189f * IzAzBz.z;
  LMS.z = 1.0f * IzAzBz.x - 0.0960192420263190f * IzAzBz.y - 0.8118918960560390f * IzAzBz.z;
  LMS.w = 0.0f;
  // L'M'S' -> LMS
  LMS = pow(fmax(LMS, 0.0f), (float4)(p_inv));
  LMS = 10000.0f * pow(fmax((c1 - LMS) / (c3 * LMS - c2), 0.0f), (float4)(n_inv));
  // LMS -> X'Y'Z
  XYZ.x =  1.9242264357876067f * LMS.x - 1.0047923125953657f * LMS.y + 0.0376514040306180f * LMS.z;
  XYZ.y =  0.3503167620949991f * LMS.x + 0.7264811939316552f * LMS.y - 0.0653844229480850f * LMS.z;
  XYZ.z = -0.0909828109828475f * LMS.x - 0.3127282905230739f * LMS.y + 1.5227665613052603f * LMS.z;
  XYZ.w = 0.0f;
  // X'Y'Z -> XYZ_D65
  float4 XYZ_D65;
  XYZ_D65.x = (XYZ.x + (b - 1.0f) * XYZ.z) / b;
  XYZ_D65.y = (XYZ.y + (g - 1.0f) * XYZ_D65.x) / g;
  XYZ_D65.z = XYZ.z;
  XYZ_D65.w = JzAzBz.w;
  return XYZ_D65;
}

static inline float vk_soft_clip(const float x, const float soft_threshold, const float hard_threshold)
{
  // exponential soft clipping above soft_threshold (hard > soft)
  const float norm = hard_threshold - soft_threshold;
  return (x > soft_threshold) ? soft_threshold + (1.0f - exp(-(x - soft_threshold) / norm)) * norm : x;
}

static inline float vk_lookup_gamut(global const float *gamut_lut, const float x)
{
  // linearly interpolate the gamut LUT at the hue angle in radians
  const float x_test = (float)LUT_ELEM * (x + M_PI_F) / DT_2PI_F;
  const float x_prev = floor(x_test);
  const float x_next = ceil(x_test);
  const int xi  = ((int)x_prev) & (LUT_ELEM - 1);
  const int xii = ((int)x_next) & (LUT_ELEM - 1);
  const float y_prev = gamut_lut[xi];
  return y_prev + ((xi != xii) ? (x_test - x_prev) * (gamut_lut[xii] - y_prev) : 0.0f);
}

// Lab <-> LCH (hue normalized to [0,1)) — used by colorzones. Mirrors
// data/kernels/colorspace.h::Lab_2_LCH / LCH_2_Lab byte-for-byte.
static inline float4 vk_Lab_2_LCH(const float4 Lab)
{
  float H = atan2(Lab.z, Lab.y);
  H = (H > 0.0f) ? H / DT_2PI_F : 1.0f - fabs(H) / DT_2PI_F;
  const float L = Lab.x;
  const float C = vk_dt_fast_hypot(Lab.y, Lab.z);
  return (float4)(L, C, H, Lab.w);
}

static inline float4 vk_LCH_2_Lab(const float4 LCH)
{
  const float L = LCH.x;
  const float a = cos(DT_2PI_F * LCH.z) * LCH.y;
  const float b = sin(DT_2PI_F * LCH.z) * LCH.y;
  return (float4)(L, a, b, LCH.w);
}

// ---- Interpolation helpers for geometric ports -----------------------
//
// Mirror data/kernels/basic.cl's interpolation_func_bicubic /
// interpolation_func_lanczos / sinf_fast / clip_mirror byte-for-byte.
// Used by ashift and (future) clip_rotate / lens, where the kernel
// does its own multi-tap reconstruction rather than using a hardware
// sampler.

// Mirror coordinate for image-edge handling: reflects negative or
// past-edge indices back into [0, edge].
static inline int vk_clip_mirror(const int x, const int edge)
{
  // Same body as common.h::clip_mirror — the standard 2-fold mirror.
  int v = x;
  if(v < 0) v = -v;
  if(v > edge) v = 2 * edge - v;
  if(v < 0) v = 0;
  if(v > edge) v = edge;
  return v;
}

// Fast sin approximation for t ∈ [-π, π]. Mirrors basic.cl::sinf_fast
// (and src/common/math.h's sinf_fast — both must change together).
static inline float vk_sinf_fast(float t)
{
  const float a = 4.0f / (M_PI_F * M_PI_F);
  const float p = 0.225f;
  t = a * t * (M_PI_F - fabs(t));
  return p * (t * fabs(t) - t) + t;
}

// Bicubic spline weight, B = 0.5. Mirrors interpolation_func_bicubic.
static inline float vk_interpolation_bicubic(float t)
{
  t = fabs(t);
  if(t >= 2.0f) return 0.0f;
  if(t >  1.0f) return 0.5f * (t * (-t * t + 5.0f * t - 8.0f) + 4.0f);
  return 0.5f * (t * (3.0f * t * t - 5.0f * t) + 2.0f);
}

// Lanczos windowed sinc, width = 2 or 3. The sign-bit trick from
// basic.cl::interpolation_func_lanczos is reproduced via a union
// (clspv-safe; the .comp twin uses uintBitsToFloat).
#define VK_LANCZOS_EPSILON (1e-9f)
static inline float vk_interpolation_lanczos(const float width, const float t)
{
  // Reduce t into [-1, 1] for the sin approximation.
  const int an = (int)t;
  const float r = t - (float)an;
  // Sign for sinf(pi * an) — even an → +1, odd an → -1.
  union { float f; unsigned int i; } sign;
  sign.i = (((unsigned int)an & 1u) << 31u) | 0x3f800000u;
  return (VK_LANCZOS_EPSILON
          + width * sign.f
                  * vk_sinf_fast(M_PI_F * r)
                  * vk_sinf_fast(M_PI_F * t / width))
       / (VK_LANCZOS_EPSILON + M_PI_F * M_PI_F * t * t);
}

// ---- colour-science helpers for colorequal -------------------------
//
// Mirror colorspace.h::matrix_dot / dt_D65_XYZ_to_xyY / dt_xyY_to_XYZ /
// dt_UCS_LUV_to_JCH / dt_UCS_HSB_to_XYZ / gamut_map_HSB byte-for-byte.
// Used by sample_input / process_data / write_output kernels.
//
// NOTE: dt_UCS_L_star_to_Y / dt_UCS_HSB_to_JCH / dt_UCS_JCH_to_xyY are
// defined once, earlier in this header (the dt-UCS block near
// DT_UCS_L_STAR_RANGE); the functions below reuse those canonical
// definitions. They are NOT re-declared here — doing so was a latent
// redefinition bug that broke every clspv compile (clspv is the only
// path that #includes this header; the glslang .comp twins are
// self-contained, so the duplication was invisible to glslang-only
// builds).

// 3-row matrix-vector multiply. The OpenCL `matrix` is a float4[3]
// where each row's .xyz is the matrix row, .w is unused. The Vulkan
// port stores the matrix as a 12-float storage buffer; rows accessed
// via base+0, base+4, base+8 (4 floats apart for alignment with the
// OpenCL `cl_mem_t copy_host_to_device_constant` 16-byte stride).
static inline float4 vk_matrix_dot(const float4 v, global const float *m)
{
  // m is laid out [m0r, m0g, m0b, m0w,  m1r, m1g, m1b, m1w,  m2r, m2g, m2b, m2w].
  const float4 vc = (float4)(v.x, v.y, v.z, 0.0f);
  const float4 r0 = (float4)(m[0], m[1], m[2],  m[3]);
  const float4 r1 = (float4)(m[4], m[5], m[6],  m[7]);
  const float4 r2 = (float4)(m[8], m[9], m[10], m[11]);
  return (float4)(dot(vc, r0), dot(vc, r1), dot(vc, r2), v.w);
}

static inline float4 vk_dt_D65_XYZ_to_xyY(const float4 sXYZ)
{
  float4 XYZ = fmax(sXYZ, (float4)(0.0f));
  float4 xyY;
  const float sum = XYZ.x + XYZ.y + XYZ.z;
  if(sum > 0.0f)
  {
    xyY.x = XYZ.x / sum;
    xyY.y = XYZ.y / sum;
  }
  else
  {
    xyY.x = 0.31271f;  // D65 white-point x
    xyY.y = 0.32902f;  // D65 white-point y
  }
  xyY.z = XYZ.y;
  xyY.w = XYZ.w;
  return xyY;
}

static inline float4 vk_dt_xyY_to_XYZ(const float4 xyY)
{
  float4 XYZ = (float4)(0.0f);
  if(xyY.y != 0.0f)
  {
    XYZ.x = xyY.z * xyY.x / xyY.y;
    XYZ.y = xyY.z;
    XYZ.z = xyY.z * (1.0f - xyY.x - xyY.y) / xyY.y;
  }
  XYZ.w = xyY.w;
  return XYZ;
}

static inline float4 vk_dt_UCS_LUV_to_JCH(const float L_star,
                                          const float L_white,
                                          const float2 UV_star_prime)
{
  const float M2 = UV_star_prime.x * UV_star_prime.x
                 + UV_star_prime.y * UV_star_prime.y;
  return (float4)(
    L_star / L_white,
    15.932993652962535f * pow(L_star, 0.6523997524738018f)
                        * pow(M2,     0.6007557017508491f) / L_white,
    atan2(UV_star_prime.y, UV_star_prime.x),
    0.0f);
}

static inline float4 vk_dt_UCS_HSB_to_XYZ(const float4 HSB, const float L_w)
{
  const float4 JCH = vk_dt_UCS_HSB_to_JCH(HSB);
  const float4 xyY = vk_dt_UCS_JCH_to_xyY(JCH, L_w);
  return vk_dt_xyY_to_XYZ(xyY);
}

static inline float vk_gamut_map_HSB(const float4 HSB,
                                     global const float *gamut_LUT,
                                     const float L_white)
{
  const float4 JCH = vk_dt_UCS_HSB_to_JCH(HSB);
  const float max_colorfulness = vk_lookup_gamut(gamut_LUT, JCH.z);
  const float max_chroma =
      15.932993652962535f
    * pow(JCH.x * L_white, 0.6523997524738018f)
    * pow(max_colorfulness, 0.6007557017508491f) / L_white;
  const float4 JCH_gb = (float4)(JCH.x, max_chroma, JCH.z, 0.0f);
  const float4 HSB_gb = vk_dt_UCS_JCH_to_HSB(JCH_gb);
  return vk_soft_clip(HSB.y, 0.8f * HSB_gb.y, HSB_gb.y);
}
