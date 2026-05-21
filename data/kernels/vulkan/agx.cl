// Vulkan port of agx.cl::kernel_agx.
//
// AgX-inspired tone mapper. The math is line-for-line equivalent to
// the OpenCL kernel — only the bindings change: image2d_t in/out
// become storage-buffer float4*, and the by-value
// dt_iop_agx_tone_mapping_params_t kernel arg becomes a storage-buffer
// binding because its size (124 B) exceeds the 128 B push-constant
// budget once width/height/base_working_same_profile are added.
//
// Bindings (7 storage buffers):
//   0: in       (float4 RGBA)
//   1: out      (float4 RGBA)
//   2: params   (dt_iop_agx_tone_mapping_params_t — 27 floats + 4 ints)
//   3: m_pb     (9-float 3x3, packed via pack_3xSSE_to_3x3)
//   4: m_br     (same)
//   5: m_rp     (same)
//   6: m_rxyz   (same)
//
// Push constants: 12 B (width, height, base_working_same_profile).

#include "dt_vulkan_common.h"

typedef struct dt_iop_agx_tone_mapping_params_t
{
  float min_ev;
  float max_ev;
  float range_in_ev;
  float curve_gamma;
  float pivot_x;
  float pivot_y;
  float target_black;
  float toe_power;
  float toe_transition_x;
  float toe_transition_y;
  float toe_scale;
  int   need_convex_toe;
  float toe_fallback_coefficient;
  float toe_fallback_power;
  float slope;
  float intercept;
  float target_white;
  float shoulder_power;
  float shoulder_transition_x;
  float shoulder_transition_y;
  float shoulder_scale;
  int   need_concave_shoulder;
  float shoulder_fallback_coefficient;
  float shoulder_fallback_power;
  float look_offset;
  float look_slope;
  float look_power;
  float look_saturation;
  float look_original_hue_mix_ratio;
  int   look_tuned;
  int   restore_hue;
} dt_iop_agx_tone_mapping_params_t;

#define _AGX_EPS 1e-6f

static inline float4 _vk_mat3x3_apply(const float4 v, global const float *m)
{
  const float R = m[0] * v.x + m[1] * v.y + m[2] * v.z;
  const float G = m[3] * v.x + m[4] * v.y + m[5] * v.z;
  const float B = m[6] * v.x + m[7] * v.y + m[8] * v.z;
  return (float4)(R, G, B, v.w);
}

static inline void _agx_compress_into_gamut(float4 *pixel)
{
  const float luminance_coeffs[3] = { 0.2658180370250449f,
                                      0.59846986045365f,
                                      0.1357121025213052f };
  const float input_y = pixel->x * luminance_coeffs[0]
                      + pixel->y * luminance_coeffs[1]
                      + pixel->z * luminance_coeffs[2];
  const float max_rgb = fmax(pixel->x, fmax(pixel->y, pixel->z));

  float4 opponent_rgb = max_rgb - (*pixel);
  const float opponent_y = opponent_rgb.x * luminance_coeffs[0]
                         + opponent_rgb.y * luminance_coeffs[1]
                         + opponent_rgb.z * luminance_coeffs[2];
  const float max_opponent = fmax(opponent_rgb.x, fmax(opponent_rgb.y, opponent_rgb.z));
  const float y_compensate_negative = max_opponent - opponent_y + input_y;

  const float min_rgb = fmin(pixel->x, fmin(pixel->y, pixel->z));
  const float offset = fmax(-min_rgb, 0.0f);
  float4 rgb_offset = (*pixel) + (float4)(offset, offset, offset, 0.0f);

  const float max_of_rgb_offset = fmax(rgb_offset.x, fmax(rgb_offset.y, rgb_offset.z));
  float4 opponent_rgb_offset = max_of_rgb_offset - rgb_offset;

  const float max_inverse_rgb_offset = fmax(opponent_rgb_offset.x,
                                            fmax(opponent_rgb_offset.y,
                                                 opponent_rgb_offset.z));
  const float y_inverse_rgb_offset = opponent_rgb_offset.x * luminance_coeffs[0]
                                   + opponent_rgb_offset.y * luminance_coeffs[1]
                                   + opponent_rgb_offset.z * luminance_coeffs[2];
  float y_new = rgb_offset.x * luminance_coeffs[0]
              + rgb_offset.y * luminance_coeffs[1]
              + rgb_offset.z * luminance_coeffs[2];
  y_new = max_inverse_rgb_offset - y_inverse_rgb_offset + y_new;

  const float luminance_ratio =
    (y_new > y_compensate_negative && y_new > _AGX_EPS)
      ? y_compensate_negative / y_new
      : 1.0f;
  *pixel = luminance_ratio * rgb_offset;
}

static inline float _agx_log_encode(const float x, const float range_in_ev, const float min_ev)
{
  const float x_relative = fmax(_AGX_EPS, x / 0.18f);
  const float mapped = (log2(fmax(x_relative, 0.0f)) - min_ev) / range_in_ev;
  return clamp(mapped, 0.0f, 1.0f);
}

static inline float _agx_sigmoid_one(const float x, const float power)
{
  return x / pow(1.0f + pow(x, power), 1.0f / power);
}

static inline float _agx_scaled_sigmoid(const float x, const float scale,
                                        const float slope, const float power,
                                        const float tx, const float ty)
{
  return scale * _agx_sigmoid_one(slope * (x - tx) / scale, power) + ty;
}

static inline float _agx_fallback_toe(const float x,
                                      global const dt_iop_agx_tone_mapping_params_t *p)
{
  return x < 0.0f
    ? p->target_black
    : p->target_black + fmax(0.0f, p->toe_fallback_coefficient
                                  * pow(x, p->toe_fallback_power));
}

static inline float _agx_fallback_shoulder(const float x,
                                           global const dt_iop_agx_tone_mapping_params_t *p)
{
  return x >= 1.0f
    ? p->target_white
    : p->target_white - fmax(0.0f, p->shoulder_fallback_coefficient
                                   * pow(1.0f - x, p->shoulder_fallback_power));
}

static inline float _agx_curve(const float x,
                               global const dt_iop_agx_tone_mapping_params_t *p)
{
  float result;
  if(x < p->toe_transition_x)
  {
    result = p->need_convex_toe
      ? _agx_fallback_toe(x, p)
      : _agx_scaled_sigmoid(x, p->toe_scale, p->slope, p->toe_power,
                            p->toe_transition_x, p->toe_transition_y);
  }
  else if(x <= p->shoulder_transition_x)
  {
    result = p->slope * x + p->intercept;
  }
  else
  {
    result = p->need_concave_shoulder
      ? _agx_fallback_shoulder(x, p)
      : _agx_scaled_sigmoid(x, p->shoulder_scale, p->slope, p->shoulder_power,
                            p->shoulder_transition_x, p->shoulder_transition_y);
  }
  return clamp(result, p->target_black, p->target_white);
}

static inline float _agx_slope_offset(const float x, const float slope, const float offset)
{
  const float m = slope / (1.0f + offset);
  const float b = offset * m;
  return m * x + b;
}

static inline void _agx_look(float4 *pixel,
                             global const dt_iop_agx_tone_mapping_params_t *p,
                             global const float *m_rxyz)
{
  const float slope  = p->look_slope;
  const float offset = p->look_offset;
  const float power  = p->look_power;
  const float sat    = p->look_saturation;

  float4 t;
  t.x = _agx_slope_offset(pixel->x, slope, offset);
  t.y = _agx_slope_offset(pixel->y, slope, offset);
  t.z = _agx_slope_offset(pixel->z, slope, offset);

  pixel->x = t.x > 0.0f ? pow(t.x, power) : t.x;
  pixel->y = t.y > 0.0f ? pow(t.y, power) : t.y;
  pixel->z = t.z > 0.0f ? pow(t.z, power) : t.z;

  const float4 xyz = _vk_mat3x3_apply(*pixel, m_rxyz);
  const float luma = xyz.y;
  pixel->x = luma + sat * (pixel->x - luma);
  pixel->y = luma + sat * (pixel->y - luma);
  pixel->z = luma + sat * (pixel->z - luma);
}

static inline float _agx_lerp_hue(const float h0, const float h1, const float mix)
{
  const float d = h1 - h0 - rint(h1 - h0);
  const float h = (1.0f - mix) * d + h0;
  return h - floor(h);
}

static inline void _agx_tone_mapping(float4 *rgb_io,
                                     global const dt_iop_agx_tone_mapping_params_t *p,
                                     global const float *m_rxyz)
{
  float4 hsv = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
  if(p->restore_hue) hsv = vk_RGB_to_HSV(*rgb_io);
  const float h_before = hsv.x;

  float4 t;
  t.x = _agx_curve(_agx_log_encode(rgb_io->x, p->range_in_ev, p->min_ev), p);
  t.y = _agx_curve(_agx_log_encode(rgb_io->y, p->range_in_ev, p->min_ev), p);
  t.z = _agx_curve(_agx_log_encode(rgb_io->z, p->range_in_ev, p->min_ev), p);
  t.w = rgb_io->w;

  if(p->look_tuned) _agx_look(&t, p, m_rxyz);

  t.x = pow(fmax(0.0f, t.x), p->curve_gamma);
  t.y = pow(fmax(0.0f, t.y), p->curve_gamma);
  t.z = pow(fmax(0.0f, t.z), p->curve_gamma);

  if(p->restore_hue)
  {
    hsv = vk_RGB_to_HSV(t);
    hsv.x = _agx_lerp_hue(h_before, hsv.x, p->look_original_hue_mix_ratio);
    *rgb_io = vk_HSV_to_RGB(hsv);
  }
  else
  {
    *rgb_io = t;
  }
}

kernel void kernel_agx(global const float4 *in,
                       global       float4 *out,
                       global const dt_iop_agx_tone_mapping_params_t *params,
                       global const float  *m_pb,
                       global const float  *m_br,
                       global const float  *m_rp,
                       global const float  *m_rxyz,
                       const int width,
                       const int height,
                       const int base_working_same_profile)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 i = in[idx];
  // Sanitize input range and replace NaNs with 0 (kernel-input safety).
  i = select(clamp(i, -1e6f, 1e6f), (float4)(0.0f), isnan(i));

  float4 base_rgb = base_working_same_profile
                  ? i
                  : _vk_mat3x3_apply(i, m_pb);

  _agx_compress_into_gamut(&base_rgb);

  float4 rendering = _vk_mat3x3_apply(base_rgb, m_br);
  _agx_tone_mapping(&rendering, params, m_rxyz);

  float4 o = _vk_mat3x3_apply(rendering, m_rp);
  o.w = i.w;
  out[idx] = o;
}
