// Vulkan port of colorequal.cl :: apply_prefilter.
// Blends the guided-filter-smoothed UV with the original UV, weighted
// by the saturation-dependent satweight LUT. 5 bindings
// (uv float2, saturation float, a float4, b float2, weights float),
// PC: sat_shift (float), width, height = 12 B.
#include "dt_vulkan_common.h"

#define VK_CE_SATSIZE 4096.0f

static inline float vk_ce_get_satweight(const float sat, global const float *weights)
{
  const float isat = VK_CE_SATSIZE * (1.0f + clamp(sat, -1.0f, 1.0f - (1.0f / VK_CE_SATSIZE)));
  const float base = floor(isat);
  const int i = (int)base;
  return weights[i] + (isat - base) * (weights[i+1] - weights[i]);
}

kernel void ce_apply_prefilter(global float2 *uv,
                               global const float *saturation,
                               global const float4 *a,
                               global const float2 *b,
                               global const float *weights,
                               const float sat_shift,
                               const int width,
                               const int height)
{
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if(col >= width || row >= height) return;
  const int k = row * width + col;

  const float2 UV = uv[k];
  const float2 cv = (float2)(a[k].x * UV.x + a[k].y * UV.y + b[k].x,
                             a[k].z * UV.x + a[k].w * UV.y + b[k].y);
  const float satweight = vk_ce_get_satweight(saturation[k] - sat_shift, weights);
  uv[k].x = mix(UV.x, cv.x, satweight);
  uv[k].y = mix(UV.y, cv.y, satweight);
}
