// Vulkan port of colorequal.cl :: final_guide.
// Second guided-filter solve: covariance inverse applied to the
// correlations → a/b coefficients for the correction guidance.
// 7 bindings (covariance float4, correlations float4, corrections
// float2, b_corrections float, uv float2, a float4, b float2),
// PC: eps (float), width, height = 12 B.
#include "dt_vulkan_common.h"

kernel void ce_final_guide(global const float4 *covariance,
                           global const float4 *correlations,
                           global const float2 *corrections,
                           global const float *b_corrections,
                           global const float2 *uv,
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
  const float4 cor = correlations[k];
  const float4 sigma = (float4)(cov.x + eps, cov.y, cov.z, cov.w + eps);
  const float det = sigma.x * sigma.w - sigma.y * sigma.z;

  float4 ak;
  if(fabs(det) > 4.0f * FLT_EPSILON)
  {
    const float4 si = (float4)(sigma.w / det, -sigma.y / det, -sigma.z / det, sigma.x / det);
    ak.x = cor.x * si.x + cor.y * si.y;
    ak.y = cor.x * si.z + cor.y * si.w;
    ak.z = cor.z * si.x + cor.w * si.y;
    ak.w = cor.z * si.z + cor.w * si.w;
  }
  else
    ak = (float4)(0.0f);
  a[k] = ak;
  b[k].x = corrections[k].y - ak.x * uv[k].x - ak.y * uv[k].y;
  b[k].y = b_corrections[k] - ak.z * uv[k].x - ak.w * uv[k].y;
}
