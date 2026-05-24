// Vulkan port of colorharmonizer.cl::colorharmonizer_map.
//
// First of two passes: per-pixel forward RGB→XYZ_D65→xyY→UCS_JCH,
// compute hue shift toward the nearest harmony node (Gaussian-weighted)
// and saturation correction, write the correction map plus the cached
// JCH so the apply kernel can skip the forward conversion.
//
// Bindings (6 storage buffers):
//   0: in              (float4 RGBA, work-profile)
//   1: p_out           (float4 per pixel — hue_shift, sat_delta, 0, 0)
//                       padded to float4 so the existing 4-channel
//                       dt_gaussian_blur_vk can smooth it before the
//                       apply kernel runs (the VK Gaussian helper
//                       only supports the 4-channel path today)
//   2: jch_out         (float4 per pixel — J, chroma, normalized hue, alpha)
//   3: matrix_in       (9 floats, pack_3xSSE_to_3x3 of work_profile→XYZ_D65)
//   4: nodes           (COLORHARMONIZER_MAX_NODES floats, hue positions)
//   5: node_saturation (COLORHARMONIZER_MAX_NODES floats, per-node sat scale)
//
// Push constants: 20 B (2 ints + 1 int + 2 floats).

#include "dt_vulkan_common.h"

#ifndef M_PI_F
#define M_PI_F   3.14159265358979323846f
#endif
#ifndef DT_2PI_F
#define DT_2PI_F 6.28318530717958647692f
#endif

static inline float4 _mat3x3_apply(const float4 v, global const float *m)
{
  const float R = m[0] * v.x + m[1] * v.y + m[2] * v.z;
  const float G = m[3] * v.x + m[4] * v.y + m[5] * v.z;
  const float B = m[6] * v.x + m[7] * v.y + m[8] * v.z;
  return (float4)(R, G, B, v.w);
}

static inline float _ch_weighted_hue_shift(const float px_hue,
                                           global const float *nodes,
                                           const int num_nodes,
                                           const float zone_width,
                                           int *out_winning_idx,
                                           float *out_max_weight)
{
  if(num_nodes <= 0)
  {
    *out_winning_idx = 0;
    *out_max_weight  = 0.0f;
    return 0.0f;
  }
  const float sigma = zone_width * 0.5f / (float)num_nodes;
  const float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);

  float max_w        = 0.0f;
  int   winning_idx  = 0;
  float diff_winning = 0.0f;

  for(int i = 0; i < num_nodes; i++)
  {
    float d = fabs(px_hue - nodes[i]);
    if(d > 0.5f) d = 1.0f - d;

    const float w = exp(-d * d * inv_2sigma2);
    float diff = nodes[i] - px_hue;
    if(diff > 0.5f)       diff -= 1.0f;
    else if(diff < -0.5f) diff += 1.0f;

    if(w > max_w)
    {
      max_w        = w;
      winning_idx  = i;
      diff_winning = diff;
    }
  }

  *out_winning_idx = winning_idx;
  *out_max_weight  = max_w;
  return diff_winning * max_w;
}

kernel void colorharmonizer_map(global const float4 *in,
                                global       float4 *p_out,
                                global       float4 *jch_out,
                                global const float  *matrix_in,
                                global const float  *nodes,
                                global const float  *node_saturation,
                                const int   width,
                                const int   height,
                                const int   num_nodes,
                                const float zone_width,
                                const float L_white)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int    idx    = idx2d(x, y, width);
  const float4 pix_in = in[idx];
  const float4 XYZ_D65 = _mat3x3_apply(fmax(0.0f, pix_in), matrix_in);
  const float4 xyY = vk_D65_XYZ_to_xyY(XYZ_D65);
  const float4 JCH = vk_xyY_to_dt_UCS_JCH(xyY, L_white);

  const float hue = (JCH.z + M_PI_F) / DT_2PI_F;
  jch_out[idx] = (float4)(JCH.x, JCH.y, hue, pix_in.w);

  int   winning_idx = 0;
  float max_weight  = 0.0f;
  const float hue_shift = _ch_weighted_hue_shift(hue, nodes, num_nodes, zone_width,
                                                 &winning_idx, &max_weight);
  const float sd = (node_saturation[winning_idx] - 1.0f) * max_weight;
  p_out[idx] = (float4)(hue_shift, sd, 0.0f, 0.0f);
}
