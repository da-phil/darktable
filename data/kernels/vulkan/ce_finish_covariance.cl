// Vulkan port of colorequal.cl :: finish_covariance.
// Subtract the outer product of the (now blurred) UV mean.
// 2 bindings (covariance float4, uv float2), 8 B PC.
#include "dt_vulkan_common.h"

kernel void ce_finish_covariance(global float4 *covariance,
                                 global const float2 *uv,
                                 const int width,
                                 const int height)
{
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if(col >= width || row >= height) return;
  const int k = row * width + col;
  covariance[k].x -= uv[k].x * uv[k].x;
  covariance[k].y -= uv[k].x * uv[k].y;
  covariance[k].z -= uv[k].x * uv[k].y;
  covariance[k].w -= uv[k].y * uv[k].y;
}
