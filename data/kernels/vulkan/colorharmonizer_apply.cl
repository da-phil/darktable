// Vulkan port of colorharmonizer.cl::colorharmonizer_apply.
//
// Second pass: apply the (Gaussian-smoothed) corrections from the
// map kernel to the cached JCH, modulate by chroma_weight to protect
// near-achromatic pixels, then convert back JCH→xyY→XYZ_D65→RGB.
//
// Bindings (4 storage buffers):
//   0: out         (float4 RGBA)
//   1: matrix_out  (9 floats, XYZ_D65 → work_profile RGB)
//   2: jch_in      (float4 per pixel — cached J, chroma, hue, alpha)
//   3: corrections (float4 per pixel — hue_shift, sat_delta, 0, 0;
//                   padded to float4 to share the 4-channel Gaussian
//                   helper. Only .xy is read here)
//
// Push constants: 20 B (2 ints + 3 floats).

#include "dt_vulkan_common.h"

#ifndef M_PI_F
#define M_PI_F   3.14159265358979323846f
#endif
#ifndef DT_2PI_F
#define DT_2PI_F 6.28318530717958647692f
#endif

static inline float _ch_wrap_hue(float h)
{
  h = fmod(h, 1.0f);
  if(h < 0.0f) h += 1.0f;
  return h;
}

static inline float4 _mat3x3_apply(const float4 v, global const float *m)
{
  const float R = m[0] * v.x + m[1] * v.y + m[2] * v.z;
  const float G = m[3] * v.x + m[4] * v.y + m[5] * v.z;
  const float B = m[6] * v.x + m[7] * v.y + m[8] * v.z;
  return (float4)(R, G, B, v.w);
}

kernel void colorharmonizer_apply(global       float4 *out,
                                  global const float  *matrix_out,
                                  global const float4 *jch_in,
                                  global const float4 *corrections,
                                  const int   width,
                                  const int   height,
                                  const float effect_strength,
                                  const float protect_neutral,
                                  const float L_white)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int    k      = idx2d(x, y, width);
  const float4 cached = jch_in[k];
  const float  J      = cached.x;
  const float  chroma = cached.y;
  const float  hue    = cached.z;

  const float4 corr_pad = corrections[k];
  const float2 corr = (float2)(corr_pad.x, corr_pad.y);

  const float t             = protect_neutral;
  const float cutoff        = t * t * t * 0.03f;
  const float chroma_weight = chroma / (chroma + cutoff + 1.0e-5f);

  float4 JCH;
  JCH.x = J;
  JCH.y = fmax(chroma * (1.0f + corr.y * chroma_weight), 0.0f);
  JCH.z = _ch_wrap_hue(hue + corr.x * effect_strength * chroma_weight) * DT_2PI_F - M_PI_F;

  const float4 xyY     = vk_dt_UCS_JCH_to_xyY(JCH, L_white);
  const float4 XYZ_D65 = vk_xyY_to_XYZ(xyY);

  float4 pix_out = _mat3x3_apply(XYZ_D65, matrix_out);
  pix_out.w = cached.w;
  out[k] = pix_out;
}
