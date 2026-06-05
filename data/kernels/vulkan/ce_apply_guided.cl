// Vulkan port of colorequal.cl :: apply_guided.
// Apply the guided coefficients to produce the final saturation and
// brightness corrections, weighted by satweight and the edge-gradient
// (scharr). 8 bindings (uv float2, saturation float, scharr float,
// a float4, b float2, corrections float2, b_corrections float,
// weights float), PC: sat_shift, bright_shift (floats), width, height = 16 B.
#include "dt_vulkan_common.h"

#define VK_CE_SATSIZE 4096.0f

static inline float vk_ce_get_satweight(const float sat, global const float *weights)
{
  const float isat = VK_CE_SATSIZE * (1.0f + clamp(sat, -1.0f, 1.0f - (1.0f / VK_CE_SATSIZE)));
  const float base = floor(isat);
  const int i = (int)base;
  return weights[i] + (isat - base) * (weights[i+1] - weights[i]);
}

kernel void ce_apply_guided(global const float2 *uv,
                            global const float *saturation,
                            global const float *scharr,
                            global const float4 *a,
                            global const float2 *b,
                            global float2 *corrections,
                            global float *b_corrections,
                            global const float *weights,
                            const float sat_shift,
                            const float bright_shift,
                            const int width,
                            const int height)
{
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if(col >= width || row >= height) return;
  const int k = row * width + col;

  const float2 CV = (float2)(a[k].x * uv[k].x + a[k].y * uv[k].y + b[k].x,
                             a[k].z * uv[k].x + a[k].w * uv[k].y + b[k].y);

  corrections[k].y = 1.0f + (CV.x - 1.0f) * vk_ce_get_satweight(saturation[k] - sat_shift, weights);
  const float gradient_weight = 1.0f - clipf(scharr[k]);
  b_corrections[k] = CV.y * gradient_weight * vk_ce_get_satweight(saturation[k] - bright_shift, weights);
}
