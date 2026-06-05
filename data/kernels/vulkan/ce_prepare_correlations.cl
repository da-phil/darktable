// Vulkan port of colorequal.cl :: prepare_correlations.
// correlations = UV ⊗ (saturation correction, brightness correction).
// 4 bindings (corrections float2, b_corrections float, uv float2,
// correlations float4), 8 B PC.
#include "dt_vulkan_common.h"

kernel void ce_prepare_correlations(global const float2 *corrections,
                                    global const float *b_corrections,
                                    global const float2 *uv,
                                    global float4 *correlations,
                                    const int width,
                                    const int height)
{
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if(col >= width || row >= height) return;
  const int k = row * width + col;
  correlations[k].x = uv[k].x * corrections[k].y;
  correlations[k].y = uv[k].y * corrections[k].y;
  correlations[k].z = uv[k].x * b_corrections[k];
  correlations[k].w = uv[k].y * b_corrections[k];
}
