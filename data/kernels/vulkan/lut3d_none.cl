// Vulkan port of lut3d.cl :: lut3d_none.
//
// Identity copy when no LUT is configured. The OpenCL kernel reads
// `in` as image2d_t with fixed integer coords — image-shortcut
// applies and the binding is a flat float4 storage buffer.
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 2 ints = 8 bytes (width, height).

#include "dt_vulkan_common.h"

kernel void lut3d_none(global const float4 *in,
                       global       float4 *out,
                       const int width,
                       const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;
  out[idx] = in[idx];
}
