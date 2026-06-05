// Vulkan port of retouch.cl :: retouch_copy_mask_to_alpha.
//
// Composites the scaled mask into the alpha channel of `in` at
// the mask's offset within the layer ROI. The OpenCL version
// passes the roi_in / roi_mask_scaled structs as device buffers;
// in VK we pass only the (x, y, width, height) ints we need via
// push-constants, no extra binding needed.
//
// Binding layout (2 storage buffers):
//   0: in           (float4, in-place; alpha is updated)
//   1: mask_scaled  (float, single channel)
// Push constants: 6 ints + 1 float = 28 B (roi_in_x, roi_in_y,
//   roi_in_width, roi_ms_x, roi_ms_y, roi_ms_width, roi_ms_height,
//   opacity).

#include "dt_vulkan_common.h"

kernel void retouch_copy_mask_to_alpha(
    global       float4 *in_buf,
    global const float  *mask_scaled,
    const int roi_in_x, const int roi_in_y, const int roi_in_width,
    const int roi_ms_x, const int roi_ms_y,
    const int roi_ms_width, const int roi_ms_height,
    const float opacity)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= roi_ms_width || y >= roi_ms_height) return;

  const int mi = y * roi_ms_width + x;
  const int di = (y + roi_ms_y - roi_in_y) * roi_in_width + (x + roi_ms_x - roi_in_x);
  const float f = clamp(mask_scaled[mi] * opacity, 0.f, 1.f);
  if(f > in_buf[di].w) in_buf[di].w = f;
}
