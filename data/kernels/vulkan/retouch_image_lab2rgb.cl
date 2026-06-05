// Vulkan port of retouch.cl :: retouch_image_lab2rgb.
//
// In-place Lab -> sRGB conversion, the inverse of rgb2lab. Used to
// undo the Lab pre-conversion after the bilateral blur runs.

#include "dt_vulkan_common.h"

kernel void retouch_image_lab2rgb(global float4 *in_buf,
                                  const int width, const int height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;
  in_buf[idx] = vk_XYZ_to_sRGB(vk_Lab_to_XYZ(in_buf[idx]));
}
