// Vulkan port of denoiseprofile.cl :: denoiseprofile_decompose.
//
// One à-trous decomposition pass at the given scale. For each pixel
// reads a 5x5 neighbourhood at the scale-dependent stride (1, 2, 4, …)
// and produces edge-weighted coarse (sum) and detail (input - sum)
// outputs.
//
// Binding layout (4 storage buffers):
//   0: in       (float4) — current scale's input
//   1: coarse   (float4) — output coarse layer
//   2: detail   (float4) — output detail layer
//   3: filter   (float)  — 25 entries: m[j]*m[i] separable kernel
// Push constants: 16 B (width, height, scale, inv_sigma2).

#include "dt_vulkan_common.h"

static inline float4 vk_dp_weight(float4 c1, float4 c2, float inv_sigma2)
{
  const float4 sqr = (c1 - c2) * (c1 - c2);
  const float dt = (sqr.x + sqr.y + sqr.z) * inv_sigma2;
  const float var = 0.02f;
  const float off2 = 9.0f;
  const float r = vk_fast_mexp2f(fmax(0.0f, dt * var - off2));
  return (float4)(r, r, r, r);
}

kernel void denoiseprofile_decompose(
    global const float4 *in_buf,
    global       float4 *coarse_buf,
    global       float4 *detail_buf,
    global const float  *filter,
    const int width, const int height,
    const int scale, const float inv_sigma2)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int mult = 1 << scale;
  const float4 pixel = read_clamped(in_buf, x, y, width, height);
  float4 sum = (float4)(0.0f);
  float4 wgt = (float4)(0.0f);

  for(int j = 0; j < 5; j++)
    for(int i = 0; i < 5; i++)
    {
      const int xx = x + mult * (i - 2);
      const int yy = y + mult * (j - 2);
      const int k  = j * 5 + i;
      const float4 px = read_clamped(in_buf, xx, yy, width, height);
      const float4 w = filter[k] * vk_dp_weight(pixel, px, inv_sigma2);
      sum += w * px;
      wgt += w;
    }

  sum = sum / wgt;
  sum.w = pixel.w;
  detail_buf[y * width + x] = pixel - sum;
  coarse_buf[y * width + x] = sum;
}
