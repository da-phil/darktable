// Vulkan port of filmic.cl :: filmicrgb_split + filmicrgb_chroma.
//
// The scene-referred filmic tone mapper. The OpenCL module has two
// kernels (filmicrgb_split for per-channel, filmicrgb_chroma for
// chroma-preserving) selected host-side, each switching on the
// colour-science version (v1..v5). They share a deep helper tree:
// norms, the filmic S-curve spline, log tone mapping, Kirk/Filmlight
// Yrg-Ych conversions and the v4/v5 gamut-mapping subsystem.
//
// This single kernel folds both entry points together and reproduces
// the host's split-vs-chroma decision internally
// (split == preserve_color NONE && version != v5), so the maths is
// byte-for-byte the OpenCL path. The colour-science cohort
// (LMS/Yrg/Ych/gamut_check) is reused from dt_vulkan_common.h — the
// same helpers colorbalancergb landed.
//
// The highlight-reconstruction path (inpaint + a-trous wavelets on
// image2d_t) and the GUI clipped-pixel mask are NOT ported; they need
// the §8.5 sampled-image bindings and stay on OpenCL/CPU via a
// process_vk_ready gate (mirrors basecurve's exposure-fusion gate).
//
// Bindings (6 storage buffers):
//   0: in           (float4 RGBA, pipeline/work profile)
//   1: out          (float4 RGBA)
//   2: params       (vk_filmicrgb_params_t — all 4-byte scalars)
//   3: matrices     (48 floats: 4 packed 3x4 dt_colormatrix_t —
//                    in @0, out @12, export_in @24, export_out @36)
//   4: profile_info (vk_dt_colorspaces_iccprofile_info_t)
//   5: profile_lut  (work-profile TRC LUT, flat)
//
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

#ifndef NORM_MIN
#define NORM_MIN 1.52587890625e-05f // norm can't be < to 2^(-16)
#endif
#define INVERSE_SQRT_3 0.5773502691896258f

// CIE Y 1931 -> CIE Y 2006 (Kirk Ych uses 2006); achromatic only.
#define CIE_Y_1931_TO_2006(x) (1.05785528f * (x))

// dt_iop_filmicrgb_methods_type_t
#define FR_METHOD_NONE          0
#define FR_METHOD_MAX_RGB       1
#define FR_METHOD_LUMINANCE     2
#define FR_METHOD_POWER_NORM    3
#define FR_METHOD_EUCLIDEAN_V1  4
#define FR_METHOD_EUCLIDEAN_V2  5

// dt_iop_filmicrgb_colorscience_type_t
#define FR_SCIENCE_V1 0
#define FR_SCIENCE_V2 1
#define FR_SCIENCE_V3 2
#define FR_SCIENCE_V4 3
#define FR_SCIENCE_V5 4

// dt_iop_filmicrgb_curve_type_t
#define FR_CURVE_POLY_4   0
#define FR_CURVE_POLY_3   1
#define FR_CURVE_RATIONAL 2

// packed 3x4 matrix bases inside the matrices buffer
#define FR_MAT_IN    0
#define FR_MAT_OUT   12
#define FR_MAT_EXIN  24
#define FR_MAT_EXOUT 36

typedef struct vk_filmicrgb_params_t
{
  float dynamic_range;
  float black_exposure;
  float grey_value;
  float sigma_toe;
  float sigma_shoulder;
  float saturation;
  float latitude_min;
  float latitude_max;
  float output_power;
  float display_black;
  float display_white;
  float norm_min;
  float norm_max;
  int   use_work_profile;
  int   color_science;       // version v1..v5
  int   variant;             // preserve_color norm method
  int   type_1;              // toe curve type
  int   type_2;              // shoulder curve type
  int   use_output_profile;
  float M1[4];
  float M2[4];
  float M3[4];
  float M4[4];
  float M5[4];
} vk_filmicrgb_params_t;

static inline float4 fr_ld4(const float a[4])
{
  return (float4)(a[0], a[1], a[2], a[3]);
}

// 3x4 matrix product, same packing as colorspace.h::matrix_product_float4.
static inline float4 fr_matrix_product(const float4 v, global const float *m, const int base)
{
  const float R = m[base + 0] * v.x + m[base + 1]  * v.y + m[base + 2]  * v.z;
  const float G = m[base + 4] * v.x + m[base + 5]  * v.y + m[base + 6]  * v.z;
  const float B = m[base + 8] * v.x + m[base + 9]  * v.y + m[base + 10] * v.z;
  return (float4)(R, G, B, v.w);
}

static inline float fr_clipf(const float a) { return clamp(a, 0.0f, 1.0f); }
static inline float4 fr_clip4(const float4 a) { return clamp(a, (float4)0.0f, (float4)1.0f); }

/* Norms */
static inline float fr_norm_power(const float4 pixel)
{
  const float4 RGB = fabs(pixel);
  const float4 RGB_square = RGB * RGB;
  const float4 RGB_cubic = RGB_square * RGB;
  return (RGB_cubic.x + RGB_cubic.y + RGB_cubic.z) / fmax(RGB_square.x + RGB_square.y + RGB_square.z, 1e-12f);
}

static inline float fr_norm_euclidean(const float4 pixel)
{
  const float4 RGB_square = pixel * pixel;
  return sqrt(RGB_square.x + RGB_square.y + RGB_square.z);
}

static inline float fr_get_pixel_norm(const float4 pixel, const int variant,
                                      global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                      global const float *lut, const int use_work_profile)
{
  switch(variant)
  {
    case FR_METHOD_MAX_RGB:
      return fmax(fmax(pixel.x, pixel.y), pixel.z);
    case FR_METHOD_LUMINANCE:
      return (use_work_profile) ? vk_get_rgb_matrix_luminance(pixel, profile_info, lut)
                                : vk_dt_camera_rgb_luminance(pixel);
    case FR_METHOD_POWER_NORM:
      return fr_norm_power(pixel);
    case FR_METHOD_EUCLIDEAN_V1:
      return fr_norm_euclidean(pixel);
    case FR_METHOD_EUCLIDEAN_V2:
      return fr_norm_euclidean(pixel) * INVERSE_SQRT_3;
    case FR_METHOD_NONE:
    default:
      return (use_work_profile) ? vk_get_rgb_matrix_luminance(pixel, profile_info, lut)
                                : vk_dt_camera_rgb_luminance(pixel);
  }
}

/* Saturation */
static inline float fr_desaturate_v1(const float x, const float sigma_toe, const float sigma_shoulder, const float saturation)
{
  const float radius_toe = x;
  const float radius_shoulder = 1.0f - x;
  const float key_toe = exp(-0.5f * radius_toe * radius_toe / sigma_toe);
  const float key_shoulder = exp(-0.5f * radius_shoulder * radius_shoulder / sigma_shoulder);
  return 1.0f - fr_clipf((key_toe + key_shoulder) / saturation);
}

static inline float fr_desaturate_v2(const float x, const float sigma_toe, const float sigma_shoulder, const float saturation)
{
  const float radius_toe = x;
  const float radius_shoulder = 1.0f - x;
  const float sat2 = 0.5f / sqrt(saturation);
  const float key_toe = exp(-radius_toe * radius_toe / sigma_toe * sat2);
  const float key_shoulder = exp(-radius_shoulder * radius_shoulder / sigma_shoulder * sat2);
  return (saturation - (key_toe + key_shoulder) * (saturation));
}

static inline float4 fr_linear_saturation(const float4 x, const float luminance, const float saturation)
{
  return (float4)luminance + (float4)saturation * (x - (float4)luminance);
}

static inline float fr_spline(const float x,
                              const float4 M1, const float4 M2, const float4 M3, const float4 M4, const float4 M5,
                              const float latitude_min, const float latitude_max,
                              const int type_0, const int type_1)
{
  float result;

  if(x < latitude_min)
  {
    // toe
    if(type_0 == FR_CURVE_POLY_4)
      result = M1.x + x * (M2.x + x * (M3.x + x * (M4.x + x * M5.x)));
    else if(type_0 == FR_CURVE_POLY_3)
      result = M1.x + x * (M2.x + x * (M3.x + x * M4.x));
    else
    {
      const float xi = latitude_min - x;
      const float rat = xi * (xi * M2.x + 1.f);
      result = M4.x - M1.x * rat / (rat + M3.x);
    }
  }
  else if(x > latitude_max)
  {
    // shoulder
    if(type_1 == FR_CURVE_POLY_4)
      result = M1.y + x * (M2.y + x * (M3.y + x * (M4.y + x * M5.y)));
    else if(type_1 == FR_CURVE_POLY_3)
      result = M1.y + x * (M2.y + x * (M3.y + x * M4.y));
    else
    {
      const float xi = x - latitude_max;
      const float rat = xi * (xi * M2.y + 1.f);
      result = M4.y + M1.y * rat / (rat + M3.y);
    }
  }
  else
  {
    // latitude
    result = M1.z + x * M2.z;
  }

  return result;
}

static inline float fr_log_tonemapping_v1(const float x, const float grey, const float black, const float dynamic_range)
{
  const float temp = (log2(x / grey) - black) / dynamic_range;
  return clamp(temp, NORM_MIN, 1.f);
}

static inline float fr_log_tonemapping_v2(const float x, const float grey, const float black, const float dynamic_range)
{
  return fr_clipf((log2(x / grey) - black) / dynamic_range);
}

static inline float4 fr_pipe_RGB_to_Ych(const float4 in, global const float *m, const int base)
{
  const float4 LMS = fr_matrix_product(in, m, base);
  const float4 Yrg = vk_LMS_to_Yrg(LMS);
  return vk_Yrg_to_Ych(Yrg);
}

static inline float4 fr_Ych_to_pipe_RGB(const float4 in, global const float *m, const int base)
{
  const float4 Yrg = vk_Ych_to_Yrg(in);
  const float4 LMS = vk_Yrg_to_LMS(Yrg);
  return fr_matrix_product(LMS, m, base);
}

static inline float4 fr_desaturate_v4(const float4 Ych_original, float4 Ych_final, const float saturation)
{
  const float chroma_original = Ych_original.y * Ych_original.x;
  float chroma_final = Ych_final.y * Ych_final.x;

  const float delta_chroma = saturation * (chroma_original - chroma_final);

  const int filmic_brightens = (Ych_final.x > Ych_original.x);
  const int filmic_resat = (chroma_original < chroma_final);
  const int filmic_desat = (chroma_original > chroma_final);
  const int user_resat = (saturation > 0.f);
  const int user_desat = (saturation < 0.f);

  chroma_final = (filmic_brightens && filmic_resat)
                      ? (chroma_original + chroma_final) / 2.f
                  : ((user_resat && filmic_desat) || user_desat)
                      ? chroma_final + delta_chroma
                      : chroma_final;

  Ych_final.y = fmax(chroma_final / Ych_final.x, 0.f);
  return Ych_final;
}

static inline float fr_clip_chroma_white_raw(global const float *m, const int row,
                                             const float target_white, const float Y,
                                             const float cos_h, const float sin_h)
{
  const float c0 = m[row + 0], c1 = m[row + 1], c2 = m[row + 2];
  const float denominator_Y_coeff = c0 * (0.979381443298969f * cos_h + 0.391752577319588f * sin_h)
                                   + c1 * (0.0206185567010309f * cos_h + 0.608247422680412f * sin_h)
                                   - c2 * (cos_h + sin_h);
  const float denominator_target_term = target_white * (0.68285981628866f * cos_h + 0.482137060515464f * sin_h);

  if(denominator_Y_coeff == 0.f) return FLT_MAX;

  const float Y_asymptote = denominator_target_term / denominator_Y_coeff;
  if(Y <= Y_asymptote) return FLT_MAX;

  const float denominator = Y * denominator_Y_coeff - denominator_target_term;
  const float numerator = -0.427506877216495f
                          * (Y * (c0 + 0.856492345150334f * c1 + 0.554995960637719f * c2)
                             - 0.988237752433297f * target_white);
  return numerator / denominator;
}

static inline float fr_clip_chroma_white(global const float *m, const int row,
                                         const float target_white, const float Y,
                                         const float cos_h, const float sin_h)
{
  const float eps = 1e-3f;
  const float max_Y = CIE_Y_1931_TO_2006(target_white);
  const float delta_Y = fmax(max_Y - Y, 0.f);
  float max_chroma;
  if(delta_Y < eps)
    max_chroma = delta_Y / (eps * max_Y) * fr_clip_chroma_white_raw(m, row, target_white, (1.f - eps) * max_Y, cos_h, sin_h);
  else
    max_chroma = fr_clip_chroma_white_raw(m, row, target_white, Y, cos_h, sin_h);
  return max_chroma >= 0.f ? max_chroma : FLT_MAX;
}

static inline float fr_clip_chroma_black(global const float *m, const int row, const float cos_h, const float sin_h)
{
  const float c0 = m[row + 0], c1 = m[row + 1], c2 = m[row + 2];
  const float denominator = c0 * (0.979381443298969f * cos_h + 0.391752577319588f * sin_h)
                          + c1 * (0.0206185567010309f * cos_h + 0.608247422680412f * sin_h)
                          - c2 * (cos_h + sin_h);
  if(denominator == 0.f) return FLT_MAX;

  const float numerator = -0.427506877216495f * (c0 + 0.856492345150334f * c1 + 0.554995960637719f * c2);
  const float max_chroma = numerator / denominator;
  return max_chroma >= 0.f ? max_chroma : FLT_MAX;
}

static inline float fr_clip_chroma(global const float *m, const int out_base, const float target_white,
                                   const float Y, const float cos_h, const float sin_h, const float chroma)
{
  const float chroma_R_white = fr_clip_chroma_white(m, out_base + 0, target_white, Y, cos_h, sin_h);
  const float chroma_G_white = fr_clip_chroma_white(m, out_base + 4, target_white, Y, cos_h, sin_h);
  const float chroma_B_white = fr_clip_chroma_white(m, out_base + 8, target_white, Y, cos_h, sin_h);
  const float max_chroma_white = fmin(fmin(chroma_R_white, chroma_G_white), chroma_B_white);

  const float chroma_R_black = fr_clip_chroma_black(m, out_base + 0, cos_h, sin_h);
  const float chroma_G_black = fr_clip_chroma_black(m, out_base + 4, cos_h, sin_h);
  const float chroma_B_black = fr_clip_chroma_black(m, out_base + 8, cos_h, sin_h);
  const float max_chroma_black = fmin(fmin(chroma_R_black, chroma_G_black), chroma_B_black);

  return fmin(fmin(chroma, max_chroma_black), max_chroma_white);
}

static inline float4 fr_gamut_check_RGB(global const float *m, const int in_base, const int out_base,
                                        const float display_black, const float display_white, const float4 Ych_in)
{
  float4 RGB_brightened = fr_Ych_to_pipe_RGB(Ych_in, m, out_base);
  const float min_pix = fmin(fmin(RGB_brightened.x, RGB_brightened.y), RGB_brightened.z);
  const float black_offset = fmax(-min_pix, 0.f);
  RGB_brightened += black_offset;
  const float4 Ych_brightened = fr_pipe_RGB_to_Ych(RGB_brightened, m, in_base);

  const float Y = clamp((Ych_in.x + Ych_brightened.x) / 2.f,
                        CIE_Y_1931_TO_2006(display_black), CIE_Y_1931_TO_2006(display_white));

  const float cos_h = Ych_in.z;
  const float sin_h = Ych_in.w;
  const float new_chroma = fr_clip_chroma(m, out_base, display_white, Y, cos_h, sin_h, Ych_in.y);

  const float4 Ych = (float4)(Y, new_chroma, cos_h, sin_h);
  const float4 RGB_out = fr_Ych_to_pipe_RGB(Ych, m, out_base);

  return clamp(RGB_out, 0.f, display_white);
}

static inline float4 fr_gamut_mapping(float4 Ych_final, float4 Ych_original, global const float *m,
                                      const float display_black, const float display_white,
                                      const float saturation, const int use_output_profile)
{
  Ych_final.z = Ych_original.z;
  Ych_final.w = Ych_original.w;

  Ych_final.x = clamp(Ych_final.x, CIE_Y_1931_TO_2006(display_black), CIE_Y_1931_TO_2006(display_white));

  Ych_final = fr_desaturate_v4(Ych_original, Ych_final, saturation);
  Ych_final = vk_gamut_check_Yrg(Ych_final);

  float4 output;
  if(!use_output_profile)
  {
    output = fr_gamut_check_RGB(m, FR_MAT_IN, FR_MAT_OUT, display_black, display_white, Ych_final);
  }
  else
  {
    const float4 export_RGB = fr_gamut_check_RGB(m, FR_MAT_EXIN, FR_MAT_EXOUT, display_black, display_white, Ych_final);
    const float4 LMS = fr_matrix_product(export_RGB, m, FR_MAT_EXIN);
    output = fr_matrix_product(LMS, m, FR_MAT_OUT);
  }
  return output;
}

/* ---- colour-science variants ---- */

static inline float4 fr_chroma_v1(const float4 i, global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                  global const float *lut, const vk_filmicrgb_params_t *p,
                                  const float4 M1, const float4 M2, const float4 M3, const float4 M4, const float4 M5)
{
  float norm = fmax(fr_get_pixel_norm(i, p->variant, profile_info, lut, p->use_work_profile), NORM_MIN);

  float4 o = i / (float4)norm;
  const float min_ratios = fmin(fmin(o.x, o.y), o.z);
  if(min_ratios < 0.0f) o -= (float4)min_ratios;

  norm = fr_log_tonemapping_v1(norm, p->grey_value, p->black_exposure, p->dynamic_range);

  o *= (float4)norm;
  const float luminance = (p->use_work_profile) ? vk_get_rgb_matrix_luminance(o, profile_info, lut)
                                                : vk_dt_camera_rgb_luminance(o);
  const float desaturation = fr_desaturate_v1(norm, p->sigma_toe, p->sigma_shoulder, p->saturation);
  o = fr_linear_saturation(o, luminance, desaturation);
  o /= (float4)norm;

  norm = pow(fr_clipf(fr_spline(norm, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2)),
             p->output_power);
  return o * norm;
}

static inline float4 fr_chroma_v2_v3(const float4 i, global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                     global const float *lut, const vk_filmicrgb_params_t *p,
                                     const float4 M1, const float4 M2, const float4 M3, const float4 M4, const float4 M5)
{
  float norm = fmax(fr_get_pixel_norm(i, p->variant, profile_info, lut, p->use_work_profile), NORM_MIN);

  float4 ratios = i / (float4)norm;
  const float min_ratios = fmin(fmin(ratios.x, ratios.y), ratios.z);
  if(min_ratios < 0.0f) ratios -= (float4)min_ratios;

  norm = fr_log_tonemapping_v2(norm, p->grey_value, p->black_exposure, p->dynamic_range);

  const float4 desaturation = (float4)fr_desaturate_v2(norm, p->sigma_toe, p->sigma_shoulder, p->saturation);

  norm = pow(fr_clipf(fr_spline(norm, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2)),
             p->output_power);

  ratios = fmax(ratios + ((float4)1.0f - ratios) * ((float4)1.0f - desaturation), (float4)0.f);

  if(p->color_science == FR_SCIENCE_V3)
    norm /= fmax(fr_get_pixel_norm(ratios, p->variant, profile_info, lut, p->use_work_profile), NORM_MIN);

  float4 o = (float4)norm * ratios;

  const float max_pix = fmax(fmax(o.x, o.y), o.z);
  const int penalize = (max_pix > 1.0f);
  if(penalize)
  {
    ratios = fmax(ratios + ((float4)1.0f - (float4)max_pix), (float4)0.0f);
    o = fr_clip4((float4)norm * ratios);
  }
  return o;
}

static inline float4 fr_chroma_v4(const float4 i, global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                  global const float *lut, global const float *m, const vk_filmicrgb_params_t *p,
                                  const float4 M1, const float4 M2, const float4 M3, const float4 M4, const float4 M5)
{
  float norm = clamp(fr_get_pixel_norm(i, p->variant, profile_info, lut, p->use_work_profile), p->norm_min, p->norm_max);

  float4 ratios = i / (float4)norm;

  norm = fr_log_tonemapping_v2(norm, p->grey_value, p->black_exposure, p->dynamic_range);

  norm = pow(clamp(fr_spline(norm, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2),
                   p->display_black, p->display_white), p->output_power);

  float4 o = norm * ratios;

  float4 Ych_original = fr_pipe_RGB_to_Ych(i, m, FR_MAT_IN);
  float4 Ych_final = fr_pipe_RGB_to_Ych(o, m, FR_MAT_IN);

  return fr_gamut_mapping(Ych_final, Ych_original, m, p->display_black, p->display_white,
                          p->saturation, p->use_output_profile);
}

static inline float4 fr_split_v4(const float4 i, global const float *m, const vk_filmicrgb_params_t *p,
                                 const float4 M1, const float4 M2, const float4 M3, const float4 M4, const float4 M5)
{
  float4 o;
  o.x = fr_log_tonemapping_v2(i.x, p->grey_value, p->black_exposure, p->dynamic_range);
  o.y = fr_log_tonemapping_v2(i.y, p->grey_value, p->black_exposure, p->dynamic_range);
  o.z = fr_log_tonemapping_v2(i.z, p->grey_value, p->black_exposure, p->dynamic_range);
  o.w = 0.f;

  o.x = fr_spline(o.x, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);
  o.y = fr_spline(o.y, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);
  o.z = fr_spline(o.z, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);

  o = pow(clamp(o, (float4)0.f, (float4)p->display_white), p->output_power);

  float4 Ych_original = fr_pipe_RGB_to_Ych(i, m, FR_MAT_IN);
  float4 Ych_final = fr_pipe_RGB_to_Ych(o, m, FR_MAT_IN);
  Ych_final.y = fmin(Ych_original.y, Ych_final.y);

  return fr_gamut_mapping(Ych_final, Ych_original, m, p->display_black, p->display_white,
                          p->saturation, p->use_output_profile);
}

static inline float4 fr_chroma_v5(const float4 i, global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                  global const float *lut, global const float *m, const vk_filmicrgb_params_t *p,
                                  const float4 M1, const float4 M2, const float4 M3, const float4 M4, const float4 M5)
{
  float norm = clamp(fr_get_pixel_norm(i, FR_METHOD_MAX_RGB, profile_info, lut, p->use_work_profile),
                     p->norm_min, p->norm_max);

  float4 ratios = i / (float4)norm;

  norm = fr_log_tonemapping_v2(norm, p->grey_value, p->black_exposure, p->dynamic_range);
  norm = pow(clamp(fr_spline(norm, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2),
                   p->display_black, p->display_white), p->output_power);

  float4 max_rgb = norm * ratios;

  float4 naive_rgb;
  naive_rgb.x = fr_log_tonemapping_v2(i.x, p->grey_value, p->black_exposure, p->dynamic_range);
  naive_rgb.y = fr_log_tonemapping_v2(i.y, p->grey_value, p->black_exposure, p->dynamic_range);
  naive_rgb.z = fr_log_tonemapping_v2(i.z, p->grey_value, p->black_exposure, p->dynamic_range);
  naive_rgb.w = 0.f;

  naive_rgb.x = fr_spline(naive_rgb.x, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);
  naive_rgb.y = fr_spline(naive_rgb.y, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);
  naive_rgb.z = fr_spline(naive_rgb.z, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);

  naive_rgb = pow(clamp(naive_rgb, (float4)0.f, (float4)p->display_white), p->output_power);

  float4 o = (0.5f - p->saturation) * naive_rgb + (0.5f + p->saturation) * max_rgb;

  float4 Ych_original = fr_pipe_RGB_to_Ych(i, m, FR_MAT_IN);
  float4 Ych_final = fr_pipe_RGB_to_Ych(o, m, FR_MAT_IN);
  Ych_final.y = fmin(Ych_original.y, Ych_final.y);

  return fr_gamut_mapping(Ych_final, Ych_original, m, p->display_black, p->display_white,
                          0.f, p->use_output_profile);
}

static inline float4 fr_split_v1(const float4 i, global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                 global const float *lut, const vk_filmicrgb_params_t *p,
                                 const float4 M1, const float4 M2, const float4 M3, const float4 M4, const float4 M5)
{
  float4 o;
  o.x = fr_log_tonemapping_v1(fmax(i.x, NORM_MIN), p->grey_value, p->black_exposure, p->dynamic_range);
  o.y = fr_log_tonemapping_v1(fmax(i.y, NORM_MIN), p->grey_value, p->black_exposure, p->dynamic_range);
  o.z = fr_log_tonemapping_v1(fmax(i.z, NORM_MIN), p->grey_value, p->black_exposure, p->dynamic_range);
  o.w = 0.f;

  const float luminance = (p->use_work_profile) ? vk_get_rgb_matrix_luminance(o, profile_info, lut)
                                                : vk_dt_camera_rgb_luminance(o);
  const float desaturation = fr_desaturate_v1(luminance, p->sigma_toe, p->sigma_shoulder, p->saturation);
  o = fr_linear_saturation(o, luminance, desaturation);

  o.x = fr_spline(o.x, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);
  o.y = fr_spline(o.y, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);
  o.z = fr_spline(o.z, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);

  o = pow(fr_clip4(o), (float4)p->output_power);
  return o;
}

static inline float4 fr_split_v2_v3(const float4 i, global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                                    global const float *lut, const vk_filmicrgb_params_t *p,
                                    const float4 M1, const float4 M2, const float4 M3, const float4 M4, const float4 M5)
{
  float4 o;
  o.x = fr_log_tonemapping_v2(fmax(i.x, NORM_MIN), p->grey_value, p->black_exposure, p->dynamic_range);
  o.y = fr_log_tonemapping_v2(fmax(i.y, NORM_MIN), p->grey_value, p->black_exposure, p->dynamic_range);
  o.z = fr_log_tonemapping_v2(fmax(i.z, NORM_MIN), p->grey_value, p->black_exposure, p->dynamic_range);
  o.w = 0.f;

  const float luminance = (p->use_work_profile) ? vk_get_rgb_matrix_luminance(o, profile_info, lut)
                                                : vk_dt_camera_rgb_luminance(o);
  const float desaturation = fr_desaturate_v2(luminance, p->sigma_toe, p->sigma_shoulder, p->saturation);
  o = fr_linear_saturation(o, luminance, desaturation);

  o.x = fr_spline(o.x, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);
  o.y = fr_spline(o.y, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);
  o.z = fr_spline(o.z, M1, M2, M3, M4, M5, p->latitude_min, p->latitude_max, p->type_1, p->type_2);

  o = pow(fr_clip4(o), (float4)p->output_power);
  return o;
}

kernel void filmicrgb(global const float4 *in,
                      global       float4 *out,
                      global const vk_filmicrgb_params_t *params,
                      global const float  *matrices,
                      global const vk_dt_colorspaces_iccprofile_info_t *profile_info,
                      global const float  *profile_lut,
                      const int width,
                      const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 i = in[idx];
  const vk_filmicrgb_params_t p = *params;

  const float4 M1 = fr_ld4(p.M1);
  const float4 M2 = fr_ld4(p.M2);
  const float4 M3 = fr_ld4(p.M3);
  const float4 M4 = fr_ld4(p.M4);
  const float4 M5 = fr_ld4(p.M5);

  // Same split-vs-chroma decision the host makes in process_cl.
  const int is_split = (p.variant == FR_METHOD_NONE && p.color_science != FR_SCIENCE_V5);

  float4 o;
  if(is_split)
  {
    if(p.color_science == FR_SCIENCE_V1)
      o = fr_split_v1(i, profile_info, profile_lut, &p, M1, M2, M3, M4, M5);
    else if(p.color_science == FR_SCIENCE_V2 || p.color_science == FR_SCIENCE_V3)
      o = fr_split_v2_v3(i, profile_info, profile_lut, &p, M1, M2, M3, M4, M5);
    else // V4
      o = fr_split_v4(i, matrices, &p, M1, M2, M3, M4, M5);
  }
  else
  {
    if(p.color_science == FR_SCIENCE_V1)
      o = fr_chroma_v1(i, profile_info, profile_lut, &p, M1, M2, M3, M4, M5);
    else if(p.color_science == FR_SCIENCE_V2 || p.color_science == FR_SCIENCE_V3)
      o = fr_chroma_v2_v3(i, profile_info, profile_lut, &p, M1, M2, M3, M4, M5);
    else if(p.color_science == FR_SCIENCE_V4)
      o = fr_chroma_v4(i, profile_info, profile_lut, matrices, &p, M1, M2, M3, M4, M5);
    else // V5
      o = fr_chroma_v5(i, profile_info, profile_lut, matrices, &p, M1, M2, M3, M4, M5);
  }

  o.w = i.w;
  out[idx] = o;
}
