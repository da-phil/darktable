// Vulkan port of blurs.cl :: restore_alpha.
//
// Copies RGB from the blurred buffer, alpha from the original. Used
// after dt_gaussian_blur_vk to restore the pipeline mask channel that
// the Gaussian helper smears along with the colour channels.
//
// Bindings (3 storage buffers):
//   0: original (float4 RGBA, alpha source)
//   1: blurred  (float4 RGBA, RGB source)
//   2: out      (float4 RGBA)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void restore_alpha(global const float4 *original,
                          global const float4 *blurred,
                          global       float4 *out,
                          const int width,
                          const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 px = blurred[idx];
  px.w = original[idx].w;
  out[idx] = px;
}
