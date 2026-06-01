// Vulkan port of dwt.cl :: dwt_subtract_layer.
//
// Per-pixel float4 subtract: bh[i] -= bl[i]. After each scale's
// hat-transform pair (row + col) produces the low-pass `bl`, this
// kernel turns the previous-scale buffer `bh` into the high-pass
// detail layer for that scale (high = orig − low).
//
// Note: the OpenCL kernel signature includes `const float lpass_mult`
// passed by the host as 1/16, but the body never references it. We
// drop the dead arg so the push-constant block stays at 8 bytes.
//
// Binding layout (2 storage buffers):
//   0: bl  (float4)  — low-pass result (read-only here)
//   1: bh  (float4)  — high-pass: written as bh -= bl
// Push constants: 2 ints = 8 bytes.

#include "dt_vulkan_common.h"

kernel void dwt_subtract_layer(global const float4 *bl,
                               global       float4 *bh,
                               const int width,
                               const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int idx = y * width + x;
  bh[idx] -= bl[idx];
}
