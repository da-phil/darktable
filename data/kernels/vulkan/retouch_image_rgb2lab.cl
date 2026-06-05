// Vulkan port of retouch.cl :: retouch_image_rgb2lab.
//
// In-place sRGB -> Lab conversion of a float4 buffer. Used inside
// the bilateral blur form so the blur operates in Lab space.
//
// Binding layout (1 storage buffer):
//   0: in  (float4, in-place)
// Push constants: 8 B (width, height).

#include "dt_vulkan_common.h"

kernel void retouch_image_rgb2lab(global float4 *in_buf,
                                  const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;
  in_buf[idx] = vk_XYZ_to_Lab(vk_sRGB_to_XYZ(in_buf[idx]));
}
