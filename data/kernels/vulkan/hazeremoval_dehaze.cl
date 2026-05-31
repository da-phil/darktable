// Vulkan port of hazeremoval.cl :: hazeremoval_dehaze.
//
// Final dehazing equation:
//   t   = max(trans_map(x,y), t_min)
//   out = (pixel - A0) / t + A0   (per channel; alpha passes through)
//
// The OpenCL kernel takes a float4 A0 with .w unused; the VK PC keeps
// just the three relevant scalars to stay tight.
//
// Bindings (3 storage buffers):
//   0: in         (float4 RGB)
//   1: trans_map  (float, refined transition map)
//   2: out        (float4 RGB)
// Push constants: 24 B (width, height, t_min, A0_x, A0_y, A0_z).

#include "dt_vulkan_common.h"

kernel void hazeremoval_dehaze(global const float4 *in,
                               global const float  *trans_map,
                               global       float4 *out,
                               const int   width,
                               const int   height,
                               const float t_min,
                               const float A0_x,
                               const float A0_y,
                               const float A0_z)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 pixel = in[idx];
  const float t = fmax(trans_map[idx], t_min);
  out[idx] = (float4)((pixel.x - A0_x) / t + A0_x,
                      (pixel.y - A0_y) / t + A0_y,
                      (pixel.z - A0_z) / t + A0_z,
                      pixel.w);
}
