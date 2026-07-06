/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Full Vulkan port of channelmixer.cl's five adaptation kernels,
    collapsed into a single entry point (`channelmixerrgb`) that
    switches on pc.adaptation at runtime. The OpenCL build keeps five
    specialized kernels so the compiler can constant-fold each mode;
    here the switch is uniform across the dispatch (all invocations
    read the same push constant) so there is no divergence cost — and
    a single entry point means the glslang fallback, which exports
    exactly one entry per .spv, covers every adaptation mode.

    The shared helpers below mirror the relevant subset of
    data/kernels/colorspace.h and data/kernels/channelmixer.cl
    byte-for-byte. Any drift from the OpenCL maths is a bug.

    Binding shape:

      binding 0: input float4 buffer  (pipe RGB)
      binding 1: output float4 buffer (pipe RGB)
      binding 2: matrices buffer (36 floats, padded 3x4 rows: RGB_to_XYZ,
                 XYZ_to_RGB, MIX — same layout the OpenCL kernels
                 receive via three constant cl_mem buffers).

    Push constants (≤128 bytes):
      int width, height, version, clip, apply_grey, adaptation
      float p, gamut
      float illuminant[4], saturation[4], lightness[4], grey[4]
*/

#include "dt_vulkan_common.h"

// ---- shared helpers --------------------------------------------------

#define NORM_MIN         1.52587890625e-05f  // 2^-16
#define INVERSE_SQRT_3   0.5773502691896258f

// Adaptation enum — values match dt_iop_channelmixer_rgb_module.h.
#define VK_ADAPT_LINEAR_BRADFORD 0
#define VK_ADAPT_CAT16           1
#define VK_ADAPT_FULL_BRADFORD   2
#define VK_ADAPT_XYZ             3
#define VK_ADAPT_RGB             4

// Version enum — values match dt_iop_channelmixer_rgb_version_t.
#define VK_CHMIX_V1 0
#define VK_CHMIX_V2 1
#define VK_CHMIX_V3 2

static inline float vk_sqf(const float x) { return x * x; }

static inline float vk_euclidean_norm4(const float4 v)
{
  return fmax(sqrt(vk_sqf(v.x) + vk_sqf(v.y) + vk_sqf(v.z)), NORM_MIN);
}

// Apply a padded 3x4 (row-major) matrix to a float4. The 4th column
// of each row is unused (corresponds to alpha and is set to 0 by the
// host). Alpha of the result is left unchanged.
static inline float4 vk_matmul_padded(const float4 v, global const float *m)
{
  float4 r;
  r.x = m[0]  * v.x + m[1]  * v.y + m[2]  * v.z;
  r.y = m[4]  * v.x + m[5]  * v.y + m[6]  * v.z;
  r.z = m[8]  * v.x + m[9]  * v.y + m[10] * v.z;
  r.w = v.w;
  return r;
}

// chroma_adapt helpers operate on the matrices buffer:
//   m[0..11]  = RGB_to_XYZ
//   m[12..23] = XYZ_to_RGB
//   m[24..35] = MIX

static inline float4 vk_chroma_RGB_to_XYZ(const float4 v, global const float *m)
{ return vk_matmul_padded(v, m); }
static inline float4 vk_chroma_XYZ_to_RGB(const float4 v, global const float *m)
{ return vk_matmul_padded(v, m + 12); }
static inline float4 vk_chroma_MIX(const float4 v, global const float *m)
{ return vk_matmul_padded(v, m + 24); }

// LMS conversions, copied byte-for-byte from colorspace.h.

static inline float4 vk_XYZ_to_bradford_LMS(const float4 XYZ)
{
  float4 r;
  r.x =  0.8951f * XYZ.x + 0.2664f * XYZ.y - 0.1614f * XYZ.z;
  r.y = -0.7502f * XYZ.x + 1.7135f * XYZ.y + 0.0367f * XYZ.z;
  r.z =  0.0389f * XYZ.x - 0.0685f * XYZ.y + 1.0296f * XYZ.z;
  r.w = XYZ.w;
  return r;
}

static inline float4 vk_bradford_LMS_to_XYZ(const float4 LMS)
{
  float4 r;
  r.x =  0.9870f * LMS.x - 0.1471f * LMS.y + 0.1600f * LMS.z;
  r.y =  0.4323f * LMS.x + 0.5184f * LMS.y + 0.0493f * LMS.z;
  r.z = -0.0085f * LMS.x + 0.0400f * LMS.y + 0.9685f * LMS.z;
  r.w = LMS.w;
  return r;
}

static inline float4 vk_XYZ_to_CAT16_LMS(const float4 XYZ)
{
  float4 r;
  r.x =  0.401288f * XYZ.x + 0.650173f * XYZ.y - 0.051461f * XYZ.z;
  r.y = -0.250268f * XYZ.x + 1.204414f * XYZ.y + 0.045854f * XYZ.z;
  r.z = -0.002079f * XYZ.x + 0.048952f * XYZ.y + 0.953127f * XYZ.z;
  r.w = XYZ.w;
  return r;
}

static inline float4 vk_CAT16_LMS_to_XYZ(const float4 LMS)
{
  float4 r;
  r.x =  1.862068f * LMS.x - 1.011255f * LMS.y + 0.149187f * LMS.z;
  r.y =  0.38752f  * LMS.x + 0.621447f * LMS.y - 0.008974f * LMS.z;
  r.z = -0.015841f * LMS.x - 0.034123f * LMS.y + 1.049964f * LMS.z;
  r.w = LMS.w;
  return r;
}

// Chromatic adaptation helpers (in-place mutations matching the
// OpenCL signatures).

static inline float4 vk_bradford_adapt_D50(float4 lms_in,
                                            const float4 origin_illuminant,
                                            const float p, const int full)
{
  const float4 D50 = (float4)(0.996078f, 1.020646f, 0.818155f, 0.f);
  if(full)
  {
    float4 t = lms_in / origin_illuminant;
    t.z = (t.z > 0.0f) ? pow(t.z, p) : t.z;
    return D50 * t;
  }
  return lms_in * (D50 / origin_illuminant);
}

static inline float4 vk_CAT16_adapt_D50(float4 lms_in,
                                         const float4 origin_illuminant,
                                         const float D, const int full)
{
  const float4 D50 = (float4)(0.994535f, 1.000997f, 0.833036f, 0.f);
  if(full) return lms_in * (D50 / origin_illuminant);
  return lms_in * (D * D50 / origin_illuminant + (float4)(1.0f - D));
}

static inline float4 vk_XYZ_adapt_D50(float4 v, const float4 origin_illuminant)
{
  const float4 D50 = (float4)(0.9642119944211994f, 1.0f, 0.8251882845188288f, 0.f);
  return v * (D50 / origin_illuminant);
}

// xyY <-> uvY <-> XYZ — used by gamut_mapping below.

static inline float4 vk_xyY_to_uvY(const float4 xyY)
{
  const float den = -2.f * xyY.x + 12.f * xyY.y + 3.f;
  return (float4)(4.f * xyY.x / den, 9.f * xyY.y / den, xyY.z, xyY.w);
}

static inline float4 vk_uvY_to_xyY(const float4 uvY)
{
  const float den = 6.f * uvY.x - 16.f * uvY.y + 12.f;
  return (float4)(9.f * uvY.x / den, 4.f * uvY.y / den, uvY.z, uvY.w);
}

static inline float4 vk_xyY_to_XYZ(const float4 xyY)
{
  float4 XYZ = (float4)(0.0f, 0.0f, 0.0f, xyY.w);
  if(xyY.y != 0.0f)
  {
    XYZ.x = xyY.z * xyY.x / xyY.y;
    XYZ.y = xyY.z;
    XYZ.z = xyY.z * (1.f - xyY.x - xyY.y) / xyY.y;
  }
  return XYZ;
}

// Gamut mapping in xyY space: compresses chroma proportionally to
// distance-from-D50². Used by all 5 kernels. Translation of
// channelmixer.cl::gamut_mapping.

static inline float4 vk_gamut_mapping(const float4 input,
                                       const float compression,
                                       const int clip)
{
  const float sum = input.x + input.y + input.z;
  const float Y = input.y;
  float4 xyY = (float4)(sum > 0.0f ? input.x / sum : 0.34567f,
                        sum > 0.0f ? input.y / sum : 0.35850f,
                        Y, 0.0f);
  float4 uvY = vk_xyY_to_uvY(xyY);

  const float2 D50 = (float2)(0.20915914598542354f, 0.488075320769787f);
  const float2 delta = D50 - uvY.xy;
  const float Delta = Y * (vk_sqf(delta.x) + vk_sqf(delta.y));
  const float correction = (compression == 0.0f) ? 0.f : pow(Delta, compression);
  const float2 tmp = correction * delta + uvY.xy;
  uvY.xy = (uvY.xy > D50) ? fmax(tmp, D50) : fmin(tmp, D50);

  xyY = vk_uvY_to_xyY(uvY);
  if(clip) xyY.xy = fmax(xyY.xy, 0.0f);
  xyY.y = fmax(xyY.y, NORM_MIN);

  // sanity x+y < 1
  const float scale = xyY.x + xyY.y;
  if(scale >= 1.f) xyY.xy /= scale;

  return vk_xyY_to_XYZ(xyY);
}

// luma_chroma — version-branched saturation / lightness mix. Lifted
// from channelmixer.cl::luma_chroma; v_1 / v_2 / v_3 paths preserved.

static inline float4 vk_luma_chroma(const float4 input,
                                     const float4 saturation,
                                     const float4 lightness,
                                     const int version)
{
  float norm = vk_euclidean_norm4(input);
  const float avg = fmax((input.x + input.y + input.z) / 3.0f, NORM_MIN);
  if(!(norm > 0.f && avg > 0.f)) return input;

  const float mix = dot(input, lightness);
  if(version == VK_CHMIX_V3) norm *= INVERSE_SQRT_3;
  float4 output = input / (float4)norm;

  float coeff_ratio = (version == VK_CHMIX_V1)
                        ? dot((float4)1.0f - output, saturation)
                        : dot(output, saturation) / 3.f;

  const float4 min_ratio = (output < 0.0f) ? output : (float4)0.0f;
  const float4 output_inv = (float4)1.0f - output;
  output = fmax(output_inv * coeff_ratio + output, min_ratio);

  if(version == VK_CHMIX_V3) norm /= vk_euclidean_norm4(output) * INVERSE_SQRT_3;
  norm *= fmax(1.f + mix / avg, 0.f);
  output *= (float4)norm;
  return output;
}

// chroma_adapt_* helpers — wrap the LMS conversion + adaptation
// + matrix product. Return adapted XYZ ready for gamut + luma_chroma.

static inline float4 vk_chroma_adapt_bradford(const float4 RGB,
                                               global const float *matrices,
                                               const float4 illuminant,
                                               const float p, const int full)
{
  float4 XYZ = vk_chroma_RGB_to_XYZ(RGB, matrices);
  const float Y = XYZ.y;
  float4 LMS = vk_XYZ_to_bradford_LMS(XYZ);
  // downscale_vector
  const float scale_down = (Y > NORM_MIN && !isnan(Y)) ? Y + NORM_MIN : NORM_MIN;
  LMS /= (float4)scale_down;
  LMS = vk_bradford_adapt_D50(LMS, illuminant, p, full);
  // upscale_vector
  const float scale_up = (Y > NORM_MIN && !isnan(Y)) ? Y + NORM_MIN : NORM_MIN;
  LMS *= (float4)scale_up;
  const float4 LMS_mixed = vk_chroma_MIX(LMS, matrices);
  return vk_bradford_LMS_to_XYZ(LMS_mixed);
}

static inline float4 vk_chroma_adapt_CAT16(const float4 RGB,
                                            global const float *matrices,
                                            const float4 illuminant,
                                            const float p, const int full)
{
  float4 XYZ = vk_chroma_RGB_to_XYZ(RGB, matrices);
  const float Y = XYZ.y;
  float4 LMS = vk_XYZ_to_CAT16_LMS(XYZ);
  const float scale_down = (Y > NORM_MIN && !isnan(Y)) ? Y + NORM_MIN : NORM_MIN;
  LMS /= (float4)scale_down;
  LMS = vk_CAT16_adapt_D50(LMS, illuminant, 1.0f, full);
  const float scale_up = (Y > NORM_MIN && !isnan(Y)) ? Y + NORM_MIN : NORM_MIN;
  LMS *= (float4)scale_up;
  const float4 LMS_mixed = vk_chroma_MIX(LMS, matrices);
  return vk_CAT16_LMS_to_XYZ(LMS_mixed);
}

static inline float4 vk_chroma_adapt_XYZ(const float4 RGB,
                                          global const float *matrices,
                                          const float4 illuminant)
{
  float4 XYZ_mixed = vk_chroma_RGB_to_XYZ(RGB, matrices);
  const float Y = XYZ_mixed.y;
  const float scale_down = (Y > NORM_MIN && !isnan(Y)) ? Y + NORM_MIN : NORM_MIN;
  XYZ_mixed /= (float4)scale_down;
  XYZ_mixed = vk_XYZ_adapt_D50(XYZ_mixed, illuminant);
  const float scale_up = (Y > NORM_MIN && !isnan(Y)) ? Y + NORM_MIN : NORM_MIN;
  XYZ_mixed *= (float4)scale_up;
  return vk_chroma_MIX(XYZ_mixed, matrices);
}

static inline float4 vk_chroma_adapt_RGB(const float4 RGB,
                                          global const float *matrices)
{
  // No white balance — mix in pipe RGB then push to XYZ.
  float4 RGB_mixed = vk_chroma_MIX(RGB, matrices);
  return vk_chroma_RGB_to_XYZ(RGB_mixed, matrices);
}

// Shared "back end" applied by every variant after the chroma_adapt:
// gamut map, convert XYZ to working space (the adaptation's LMS or
// XYZ or pipe RGB), apply luma_chroma, optionally collapse to grey,
// convert back to pipe RGB.

typedef struct vk_chmix_args_t
{
  float4 illuminant;
  float4 saturation;
  float4 lightness;
  float4 grey;
  float p;
  float gamut;
  int clip;
  int apply_grey;
  int version;
  int adaptation;
} vk_chmix_args_t;

static inline float4 vk_chmix_finalize(float4 XYZ,
                                       const float4 pix_in,
                                       global const float *matrices,
                                       vk_chmix_args_t a)
{
  if(a.clip) XYZ = fmax(XYZ, (float4)0.0f);
  XYZ = vk_gamut_mapping(XYZ, a.gamut, a.clip);

  // Convert XYZ to the working LMS/XYZ/pipe-RGB space.
  float4 LMS;
  switch(a.adaptation)
  {
    case VK_ADAPT_FULL_BRADFORD:
    case VK_ADAPT_LINEAR_BRADFORD:
      LMS = vk_XYZ_to_bradford_LMS(XYZ); break;
    case VK_ADAPT_CAT16:
      LMS = vk_XYZ_to_CAT16_LMS(XYZ); break;
    case VK_ADAPT_XYZ:
      LMS = XYZ; break;
    case VK_ADAPT_RGB:
    default:
      LMS = vk_chroma_XYZ_to_RGB(XYZ, matrices); break;
  }
  if(a.clip) LMS = fmax(LMS, (float4)0.0f);

  LMS = vk_luma_chroma(LMS, a.saturation, a.lightness, a.version);
  if(a.clip) LMS = fmax(LMS, (float4)0.0f);

  float4 RGB;
  if(a.apply_grey)
  {
    const float grey_mix = fmax(dot(LMS, a.grey), 0.0f);
    RGB = (float4)(grey_mix, grey_mix, grey_mix, pix_in.w);
  }
  else
  {
    switch(a.adaptation)
    {
      case VK_ADAPT_FULL_BRADFORD:
      case VK_ADAPT_LINEAR_BRADFORD:
        XYZ = vk_bradford_LMS_to_XYZ(LMS); break;
      case VK_ADAPT_CAT16:
        XYZ = vk_CAT16_LMS_to_XYZ(LMS); break;
      case VK_ADAPT_XYZ:
        XYZ = LMS; break;
      case VK_ADAPT_RGB:
      default:
        XYZ = vk_chroma_RGB_to_XYZ(LMS, matrices); break;
    }
    if(a.clip) XYZ = fmax(XYZ, (float4)0.0f);
    RGB = vk_chroma_XYZ_to_RGB(XYZ, matrices);
    if(a.clip) RGB = fmax(RGB, (float4)0.0f);
    RGB.w = pix_in.w;
  }
  return RGB;
}

// ---- shared push-constant layout ------------------------------------

typedef struct vk_chmix_pc_t
{
  int width;
  int height;
  int version;
  int clip;
  int apply_grey;
  int adaptation;   // VK_ADAPT_* (values match dt_adaptation_t)
  float p;
  float gamut;
  float illuminant[4];
  float saturation[4];
  float lightness[4];
  float grey[4];
} vk_chmix_pc_t;

static inline vk_chmix_args_t vk_chmix_load_args(const vk_chmix_pc_t pc,
                                                  const int adaptation)
{
  vk_chmix_args_t a;
  a.illuminant = (float4)(pc.illuminant[0], pc.illuminant[1], pc.illuminant[2], pc.illuminant[3]);
  a.saturation = (float4)(pc.saturation[0], pc.saturation[1], pc.saturation[2], pc.saturation[3]);
  a.lightness  = (float4)(pc.lightness[0],  pc.lightness[1],  pc.lightness[2],  pc.lightness[3]);
  a.grey       = (float4)(pc.grey[0],       pc.grey[1],       pc.grey[2],       pc.grey[3]);
  a.p          = pc.p;
  a.gamut      = pc.gamut;
  a.clip       = pc.clip;
  a.apply_grey = pc.apply_grey;
  a.version    = pc.version;
  a.adaptation = adaptation;
  return a;
}

// ---- entry point -----------------------------------------------------
//
// One kernel switching on pc.adaptation at runtime instead of five
// per-adaptation entries. The branch is uniform across the dispatch
// (every invocation reads the same push constant), so there is no
// divergence cost, and a single entry point means the glslang
// fallback — which can only export one entry per .spv — covers every
// adaptation mode instead of just linear Bradford.

kernel void channelmixerrgb(global const float4 *in,
                            global float4 *out,
                            global const float *matrices,
                            const vk_chmix_pc_t pc)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= pc.width || y >= pc.height) return;
  const int idx = idx2d(x, y, pc.width);
  float4 pix_in = in[idx];
  float4 RGB = pix_in;
  RGB.w = 0.0f;
  if(pc.clip) RGB = fmax(RGB, (float4)0.0f);

  const vk_chmix_args_t a = vk_chmix_load_args(pc, pc.adaptation);
  float4 XYZ;
  switch(pc.adaptation)
  {
    case VK_ADAPT_FULL_BRADFORD:
      XYZ = vk_chroma_adapt_bradford(RGB, matrices, a.illuminant, a.p, 1);
      break;
    case VK_ADAPT_LINEAR_BRADFORD:
      XYZ = vk_chroma_adapt_bradford(RGB, matrices, a.illuminant, a.p, 0);
      break;
    case VK_ADAPT_CAT16:
      XYZ = vk_chroma_adapt_CAT16(RGB, matrices, a.illuminant, 1.0f, 0);
      break;
    case VK_ADAPT_XYZ:
      XYZ = vk_chroma_adapt_XYZ(RGB, matrices, a.illuminant);
      break;
    case VK_ADAPT_RGB:
    default:
      XYZ = vk_chroma_adapt_RGB(RGB, matrices);
      break;
  }
  out[idx] = vk_chmix_finalize(XYZ, pix_in, matrices, a);
}
