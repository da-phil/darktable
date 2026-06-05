// Vulkan port of colorequal.cl :: finish_correlations.
// Subtract the blurred-mean outer products from both covariance and
// correlations. 5 bindings (corrections float2, b_corrections float,
// uv float2, correlations float4, covariance float4), 8 B PC.
#include "dt_vulkan_common.h"

kernel void ce_finish_correlations(global const float2 *corrections,
                                   global const float *b_corrections,
                                   global const float2 *uv,
                                   global float4 *correlations,
                                   global float4 *covariance,
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

  correlations[k].x -= uv[k].x * corrections[k].y;
  correlations[k].y -= uv[k].y * corrections[k].y;
  correlations[k].z -= uv[k].x * b_corrections[k];
  correlations[k].w -= uv[k].y * b_corrections[k];
}
