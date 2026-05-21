// Vulkan port of sigmoid.cl::sigmoid_loglogistic_rgb_ratio.
//
// Norm-preserving sigmoid tone mapper: applies a loglogistic curve to
// the per-pixel luma, then scales the RGB triplet uniformly, then
// compresses chroma via a hyperbolic gamut roll-off. No matrices —
// works in pipe space directly (the per_channel kernel handles the
// pipe→rendering primary remap).
//
// Bindings (2 storage buffers): in, out  (RGBA float, pipe space)
// Push constants: 32 B (2 ints + 6 floats).

#include "dt_vulkan_common.h"

static inline float4 _vk_desat_neg(const float4 i)
{
  const float avg = fmax((i.x + i.y + i.z) / 3.0f, 0.0f);
  const float mn  = fmin(fmin(i.x, i.y), i.z);
  const float sf  = (mn < 0.0f) ? -avg / (mn - avg) : 1.0f;
  return avg + sf * (i - avg);
}

static inline float _vk_sigmoid_scalar(const float v, const float magnitude,
                                       const float paper_exp, const float film_fog,
                                       const float film_power, const float paper_power)
{
  const float c = fmax(v, 0.0f);
  const float film  = pow(film_fog + c, film_power);
  const float paper = magnitude * pow(film / (paper_exp + film), paper_power);
  return isnan(paper) ? magnitude : paper;
}

kernel void sigmoid_loglogistic_rgb_ratio(global const float4 *in,
                                          global       float4 *out,
                                          const int    width,
                                          const int    height,
                                          const float  white_target,
                                          const float  black_target,
                                          const float  paper_exp,
                                          const float  film_fog,
                                          const float  contrast_power,
                                          const float  skew_power)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 i = in[idx];
  const float alpha = i.w;

  i = _vk_desat_neg(i);

  const float luma = (i.x + i.y + i.z) / 3.0f;
  const float mapped = _vk_sigmoid_scalar(luma, white_target,
                                          paper_exp, film_fog,
                                          contrast_power, skew_power);

  if(luma > 1e-9f)
  {
    const float s = mapped / luma;
    i = s * i;
  }
  else
  {
    i = (float4)mapped;
  }

  const float pmin = fmin(fmin(i.x, i.y), i.z);
  const float pmax = fmax(fmax(i.x, i.y), i.z);

  const float eps = 1e-6f;
  const float dw  = (white_target - mapped) / (pmax - mapped + eps);
  const float db  = (black_target - mapped) / (pmin - mapped - eps);
  const float d   = fmin(dw, db);
  const float c   = (mapped - pmin) / (mapped + eps);

  const float adj = 1.0f / (c * d + eps);
  const float hc  = 2.0f * c / (1.0f - c * c + eps) * adj;
  const float hz  = sqrt(hc * hc + 1.0f);
  const float cf  = hc / (1.0f + hz) * d;

  i = mapped + cf * (i - mapped);
  i.w = alpha;
  out[idx] = i;
}
