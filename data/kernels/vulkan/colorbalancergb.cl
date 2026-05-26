// Vulkan port of extended.cl::colorbalancergb.
//
// Scene-referred color balance in CIE 2006 LMS / Filmlight Yrg-Ych,
// then a perceptual saturation/brilliance pass in either JzAzBz (2021)
// or darktable UCS (2022). The OpenCL kernel takes ~50 scalar + vector
// args; that exceeds the 128 B push-constant budget, so (like agx) the
// parameter block is migrated into a storage buffer. The work profile
// is baked into the two host-premultiplied 3x4 matrices, so no ICC
// profile_info binding is needed (the OpenCL kernel takes one but never
// reads it).
//
// Bindings (6 storage buffers):
//   0: in         (float4 RGBA, pipeline/work profile, D50)
//   1: out        (float4 RGBA)
//   2: params     (vk_colorbalancergb_params_t — all 4-byte scalars,
//                  std430 == the C struct)
//   3: matrix_in  (12 floats: RGB_D50 -> LMS_2006_D65, premultiplied)
//   4: matrix_out (12 floats: XYZ_D65 -> RGB_D50, premultiplied)
//   5: gamut_lut  (LUT_ELEM floats, hue -> max colorfulness)
//
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

#ifndef DT_COLORBALANCE_SATURATION_JZAZBZ
#define DT_COLORBALANCE_SATURATION_JZAZBZ 0
#endif

typedef struct vk_colorbalancergb_params_t
{
  float shadows_weight;
  float highlights_weight;
  float midtones_weight;
  float mask_grey_fulcrum;
  float chroma_global;
  float chroma[4];
  float vibrance;
  float hue_rotation_matrix[4];
  float global_offset[4];
  float shadows[4];
  float highlights[4];
  float midtones[4];
  float white_fulcrum;
  float midtones_Y;
  float grey_fulcrum;
  float contrast;
  float brilliance_global;
  float brilliance[4];
  float saturation_global;
  float saturation[4];
  float L_white;
  int   saturation_formula;
  int   mask_display;
  int   mask_type;
  int   checker_1;
  int   checker_2;
  float checker_color_1[4];
  float checker_color_2[4];
} vk_colorbalancergb_params_t;

static inline float4 _cb_matrix_product(const float4 v, global const float *m)
{
  // 3x4 matrix, same packing as colorspace.h::matrix_product_float4.
  const float R = m[0] * v.x + m[1] * v.y + m[2]  * v.z;
  const float G = m[4] * v.x + m[5] * v.y + m[6]  * v.z;
  const float B = m[8] * v.x + m[9] * v.y + m[10] * v.z;
  return (float4)(R, G, B, v.w);
}

static inline float4 _cb_ld4(global const float *a)
{
  return (float4)(a[0], a[1], a[2], a[3]);
}

// Mirrors extended.cl::opacity_masks byte-for-byte.
static inline float4 _cb_opacity_masks(const float x,
                                       const float shadows_weight, const float highlights_weight,
                                       const float midtones_weight, const float mask_grey_fulcrum)
{
  const float x_offset = (x - mask_grey_fulcrum);
  const float x_offset_norm = x_offset / mask_grey_fulcrum;
  const float alpha = 1.0f / (1.0f + exp(x_offset_norm * shadows_weight));    // shadows
  const float beta  = 1.0f / (1.0f + exp(-x_offset_norm * highlights_weight)); // highlights
  const float gamma = exp(-x_offset * x_offset * midtones_weight / 4.0f)
                      * (1.0f - alpha) * (1.0f - alpha)
                      * (1.0f - beta)  * (1.0f - beta) * 8.0f;                 // midtones
  return (float4)(alpha, gamma, beta, 0.0f);
}

kernel void colorbalancergb(global const float4 *in,
                            global       float4 *out,
                            global const vk_colorbalancergb_params_t *params,
                            global const float  *matrix_in,
                            global const float  *matrix_out,
                            global const float  *gamut_lut,
                            const int width,
                            const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int idx = idx2d(x, y, width);
  // clip pipeline RGB while reading; this also gives a proper alpha
  const float4 pix_in = fmax(0.0f, in[idx]);

  const float4 chroma        = _cb_ld4(params->chroma);
  const float4 global_offset = _cb_ld4(params->global_offset);
  const float4 shadows       = _cb_ld4(params->shadows);
  const float4 highlights    = _cb_ld4(params->highlights);
  const float4 midtones      = _cb_ld4(params->midtones);
  const float4 brilliance    = _cb_ld4(params->brilliance);
  const float4 saturation    = _cb_ld4(params->saturation);
  const float L_white = params->L_white;

  float4 RGB = pix_in;

  // go to CIE 2006 LMS D65
  float4 LMS = _cb_matrix_product(RGB, matrix_in);

  // go to Filmlight Yrg then Ych
  float4 Yrg = vk_LMS_to_Yrg(LMS);
  float4 Ych = vk_Yrg_to_Ych(Yrg);

  // Sanitize input : no negative luminance
  Ych.x = fmax(Ych.x, 0.0f);
  const float4 opacities = _cb_opacity_masks(pow(Ych.x, 0.4101205819200422f), // center middle grey at 50 %
                                             params->shadows_weight, params->highlights_weight,
                                             params->midtones_weight, params->mask_grey_fulcrum);
  const float4 opacities_comp = (float4)(1.0f) - opacities;

  // Hue shift via the 2x2 rotation matrix (needs the gamut limit at output hue next)
  const float cos_h = Ych.z;
  const float sin_h = Ych.w;
  Ych.z = params->hue_rotation_matrix[0] * cos_h + params->hue_rotation_matrix[1] * sin_h;
  Ych.w = params->hue_rotation_matrix[2] * cos_h + params->hue_rotation_matrix[3] * sin_h;

  // Linear chroma : distance to achromatic at constant luminance
  const float chroma_boost = params->chroma_global + dot(opacities, chroma);
  const float vib = params->vibrance * (1.0f - pow(Ych.y, fabs(params->vibrance)));
  const float chroma_factor = fmax(1.0f + chroma_boost + vib, 0.0f);
  Ych.y *= chroma_factor;

  // clip chroma at constant Y and hue
  Ych = vk_gamut_check_Yrg(Ych);

  // back to Yrg, then LMS, then Filmlight RGB
  Yrg = vk_Ych_to_Yrg(Ych);
  LMS = vk_Yrg_to_LMS(Yrg);
  RGB = vk_LMS_to_gradingRGB(LMS);

  // Color balance
  // global : offset
  RGB += global_offset;

  // highlights, shadows : 2 slopes with masking
  RGB *= opacities_comp.z * (opacities_comp.x + opacities.x * shadows) + opacities.z * highlights;

  // midtones : power with sign preservation
  RGB = sign(RGB) * pow(fabs(RGB) / params->white_fulcrum, midtones) * params->white_fulcrum;

  // non-linear ops in Yrg (RGB doesn't preserve colour)
  LMS = vk_gradingRGB_to_LMS(RGB);
  Yrg = vk_LMS_to_Yrg(LMS);

  // Y midtones power (gamma)
  Yrg.x = pow(fmax(Yrg.x / params->white_fulcrum, 0.0f), params->midtones_Y) * params->white_fulcrum;
  // Y fulcrumed contrast
  Yrg.x = params->grey_fulcrum * pow(Yrg.x / params->grey_fulcrum, params->contrast);

  LMS = vk_Yrg_to_LMS(Yrg);
  float4 XYZ_D65 = vk_LMS_to_XYZ(LMS);

  // Perceptual color adjustments
  if(params->saturation_formula == DT_COLORBALANCE_SATURATION_JZAZBZ)
  {
    float4 Jab = vk_XYZ_to_JzAzBz(XYZ_D65);

    float JC[2] = { Jab.x, vk_dt_fast_hypot(Jab.y, Jab.z) };
    const float h = atan2(Jab.z, Jab.y);

    const float T = atan2(JC[1], JC[0]);
    const float sin_T = sin(T);
    const float cos_T = cos(T);
    const float M_rot_dir[2][2] = { {  cos_T,  sin_T },
                                    { -sin_T,  cos_T } };
    const float M_rot_inv[2][2] = { {  cos_T, -sin_T },
                                    {  sin_T,  cos_T } };
    float SO[2];

    const float boosts[2] = { 1.0f + params->brilliance_global + dot(opacities, brilliance),
                              params->saturation_global + dot(opacities, saturation) };

    SO[0] = JC[0] * M_rot_dir[0][0] + JC[1] * M_rot_dir[0][1];
    SO[1] = SO[0] * clamp(T * boosts[1], -T, M_PI_F / 2.0f - T);
    SO[0] = fmax(SO[0] * boosts[0], 0.0f);

    JC[0] = fmax(SO[0] * M_rot_inv[0][0] + SO[1] * M_rot_inv[0][1], 0.0f);
    JC[1] = fmax(SO[0] * M_rot_inv[1][0] + SO[1] * M_rot_inv[1][1], 0.0f);

    const float out_max_sat_h = vk_lookup_gamut(gamut_lut, h);
    const float sat = (JC[0] > 0.0f) ? vk_soft_clip(JC[1] / JC[0], 0.8f * out_max_sat_h, out_max_sat_h)
                                     : out_max_sat_h;
    const float max_C_at_sat = JC[0] * sat;
    const float max_J_at_sat = (sat > 0.0f) ? JC[1] / sat : JC[0];
    JC[0] = (JC[0] + max_J_at_sat) / 2.0f;
    JC[1] = (JC[1] + max_C_at_sat) / 2.0f;

    const float cos_H = cos(h);
    const float sin_H = sin(h);

    const float d0 = 1.6295499532821566e-11f;
    const float d = -0.56f;
    float Iz = JC[0] + d0;
    Iz /= (1.0f + d - d * Iz);
    Iz = fmax(Iz, 0.0f);

    const float4 AI[3] = { {  1.0f,  0.1386050432715393f,  0.0580473161561189f, 0.0f },
                           {  1.0f, -0.1386050432715393f, -0.0580473161561189f, 0.0f },
                           {  1.0f, -0.0960192420263190f, -0.8118918960560390f, 0.0f } };

    const float4 IzAzBz = { Iz, JC[1] * cos_H, JC[1] * sin_H, 0.0f };
    LMS.x = dot(AI[0], IzAzBz);
    LMS.y = dot(AI[1], IzAzBz);
    LMS.z = dot(AI[2], IzAzBz);

    float max_C = JC[1];
    if(LMS.x < 0.0f)
      max_C = fmin(-Iz / (AI[0].y * cos_H + AI[0].z * sin_H), max_C);
    if(LMS.y < 0.0f)
      max_C = fmin(-Iz / (AI[1].y * cos_H + AI[1].z * sin_H), max_C);
    if(LMS.z < 0.0f)
      max_C = fmin(-Iz / (AI[2].y * cos_H + AI[2].z * sin_H), max_C);

    Jab.x = JC[0];
    Jab.y = max_C * cos_H;
    Jab.z = max_C * sin_H;

    XYZ_D65 = vk_JzAzBz_2_XYZ(Jab);
  }
  else
  {
    float4 xyY = vk_D65_XYZ_to_xyY(XYZ_D65);
    float4 JCH = vk_xyY_to_dt_UCS_JCH(xyY, L_white);
    float4 HCB = vk_dt_UCS_JCH_to_HCB(JCH);

    const float radius = vk_dt_fast_hypot(HCB.y, HCB.z);
    const float sin_T = (radius > 0.0f) ? HCB.y / radius : 0.0f;
    const float cos_T = (radius > 0.0f) ? HCB.z / radius : 0.0f;
    const float M_rot_inv[2][2] = { { cos_T,  sin_T }, { -sin_T, cos_T } };

    const float P = fmax(FLT_MIN, HCB.y);
    const float W = sin_T * HCB.y + cos_T * HCB.z;

    float a = fmax(1.0f + params->saturation_global + dot(opacities, saturation), 0.0f);
    const float b = fmax(1.0f + params->brilliance_global + dot(opacities, brilliance), 0.0f);

    const float max_a = vk_dt_fast_hypot(P, W) / P;
    a = vk_soft_clip(a, 0.5f * max_a, max_a);

    const float P_prime = (a - 1.0f) * P;
    const float W_prime = sqrt(P * P * (1.0f - a * a) + W * W) * b;

    HCB.y = fmax(M_rot_inv[0][0] * P_prime + M_rot_inv[0][1] * W_prime, 0.0f);
    HCB.z = fmax(M_rot_inv[1][0] * P_prime + M_rot_inv[1][1] * W_prime, 0.0f);

    JCH = vk_dt_UCS_HCB_to_JCH(HCB);

    const float max_colorfulness = vk_lookup_gamut(gamut_lut, JCH.z); // this is M^2
    const float max_chroma = 15.932993652962535f * pow(JCH.x * L_white, 0.6523997524738018f)
                             * pow(max_colorfulness, 0.6007557017508491f) / L_white;
    const float4 JCH_gamut_boundary = { JCH.x, max_chroma, JCH.z, 0.0f };
    const float4 HSB_gamut_boundary = vk_dt_UCS_JCH_to_HSB(JCH_gamut_boundary);

    float4 HSB = { HCB.x, (HCB.z > 0.0f) ? HCB.y / HCB.z : 0.0f, HCB.z, 0.0f };
    HSB.y = vk_soft_clip(HSB.y, 0.8f * HSB_gamut_boundary.y, HSB_gamut_boundary.y);

    JCH = vk_dt_UCS_HSB_to_JCH(HSB);
    xyY = vk_dt_UCS_JCH_to_xyY(JCH, L_white);
    XYZ_D65 = vk_xyY_to_XYZ(xyY);
  }

  // Project back to D50 pipeline RGB
  RGB = _cb_matrix_product(XYZ_D65, matrix_out);

  if(params->mask_display)
  {
    float4 color;
    if(x % params->checker_1 < x % params->checker_2)
    {
      if(y % params->checker_1 < y % params->checker_2) color = _cb_ld4(params->checker_color_2);
      else                                              color = _cb_ld4(params->checker_color_1);
    }
    else
    {
      if(y % params->checker_1 < y % params->checker_2) color = _cb_ld4(params->checker_color_1);
      else                                              color = _cb_ld4(params->checker_color_2);
    }
    // opacities are laid out {alpha(shadows), gamma(midtones), beta(highlights), 0}
    float opacity = (params->mask_type == 0) ? opacities.x
                  : (params->mask_type == 1) ? opacities.y
                  : (params->mask_type == 2) ? opacities.z
                  :                            opacities.w;
    const float opacity_comp = 1.0f - opacity;
    RGB = opacity_comp * color + opacity * fmax(RGB, 0.0f);
    RGB.w = 1.0f;
  }
  else
  {
    RGB = fmax(RGB, 0.0f);
    RGB.w = pix_in.w;
  }

  out[idx] = RGB;
}
