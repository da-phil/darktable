// Vulkan port of colorequal.cl :: prepare_prefilter.
// Per-pixel 2x2 covariance inverse → guided-filter a/b coefficients.
// 4 bindings (uv float2, covariance float4, a float4, b float2),
// PC: eps (float), width, height = 12 B.
#include "dt_vulkan_common.h"

kernel void ce_prepare_prefilter(global const float2 *uv,
                                 global const float4 *covariance,
                                 global float4 *a,
                                 global float2 *b,
                                 const float eps,
                                 const int width,
                                 const int height)
{
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if(col >= width || row >= height) return;
  const int k = row * width + col;

  const float4 cov = covariance[k];
  const float4 sigma = (float4)(cov.x + eps, cov.y, cov.z, cov.w + eps);
  const float det = sigma.x * sigma.w - sigma.y * sigma.z;

  float4 ak;
  if(fabs(det) > 4.0f * FLT_EPSILON)
  {
    const float4 si = (float4)(sigma.w / det, -sigma.y / det, -sigma.z / det, sigma.x / det);
    ak.x = cov.x * si.x + cov.y * si.y;
    ak.y = cov.x * si.z + cov.y * si.w;
    ak.z = cov.z * si.x + cov.w * si.y;
    ak.w = cov.z * si.z + cov.w * si.w;
  }
  else
    ak = (float4)(0.0f);
  a[k] = ak;
  b[k].x = uv[k].x - ak.x * uv[k].x - ak.y * uv[k].y;
  b[k].y = uv[k].y - ak.z * uv[k].x - ak.w * uv[k].y;
}
