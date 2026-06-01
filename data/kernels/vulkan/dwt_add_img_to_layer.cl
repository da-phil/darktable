// Vulkan port of dwt.cl :: dwt_add_img_to_layer.
//
// Per-pixel float4 add: layer[i] += img[i]. Used by the host helper to
// accumulate every visible detail scale into the reconstructed image
// and, when `merge_from_scale > 0`, to fold the merged detail layers
// into the final output.
//
// Binding layout (2 storage buffers):
//   0: img    (float4)  — read-only source
//   1: layer  (float4)  — destination, accumulated in place
// Push constants: 2 ints = 8 bytes.

#include "dt_vulkan_common.h"

kernel void dwt_add_img_to_layer(global const float4 *img,
                                 global       float4 *layer,
                                 const int width,
                                 const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int idx = y * width + x;
  layer[idx] += img[idx];
}
