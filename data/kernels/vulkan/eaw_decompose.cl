// Vulkan port of atrous.cl :: eaw_decompose.
//
// Edge-aware à-trous wavelet decompose: 5x5 stencil with sample
// spacing `mult = 1 << scale`, per-channel exponential weights based
// on Lab L (luma) and a,b (chroma) deltas. Writes the coarse (low-
// frequency) image plus the detail (= input − coarse) for this scale.
// Math byte-for-byte identical to atrous.cl; the OpenCL sampleri
// (CLAMP_TO_EDGE) becomes an explicit clamp on the source index.
//
// Bindings (4 storage buffers):
//   0: in       (float4, source for this scale)
//   1: coarse   (float4, low-pass output)
//   2: detail   (float4, high-frequency output = in - coarse)
//   3: filter   (25 floats, 5x5 separable B3-spline kernel, host-built)
// Push constants: 16 B (width, height, scale, sharpen).

#include "dt_vulkan_common.h"

static inline float4 vk_atrous_weight(const float4 c1, const float4 c2, const float sharpen)
{
  const float wc = exp(-((c1.y - c2.y) * (c1.y - c2.y) + (c1.z - c2.z) * (c1.z - c2.z)) * sharpen);
  const float wl = exp(-(c1.x - c2.x) * (c1.x - c2.x) * sharpen);
  return (float4)(wl, wc, wc, 1.0f);
}

kernel void eaw_decompose(global const float4 *in,
                          global       float4 *coarse,
                          global       float4 *detail,
                          global const float  *filter,
                          const int   width,
                          const int   height,
                          const int   scale,
                          const float sharpen)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const int mult = 1 << scale;
  const float4 pixel = in[idx];

  float4 sum = (float4)0.0f;
  float4 wgt = (float4)0.0f;
  for(int j = 0; j < 5; j++)
    for(int i = 0; i < 5; i++)
    {
      const int xx = clamp(mult * (i - 2) + x, 0, width  - 1);
      const int yy = clamp(mult * (j - 2) + y, 0, height - 1);
      const float4 px = in[idx2d(xx, yy, width)];
      const float4 w = filter[j * 5 + i] * vk_atrous_weight(pixel, px, sharpen);
      sum += w * px;
      wgt += w;
    }
  sum /= wgt;
  sum.w = pixel.w;

  detail[idx] = pixel - sum;
  coarse[idx] = sum;
}
