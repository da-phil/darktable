// Vulkan port of retouch.cl :: retouch_fill.
//
// Mix the layer pixel with a constant fill color, weighted by the
// scaled mask × opacity. Alpha is preserved.
//
// Binding layout (2 storage buffers):
//   0: in           (float4, in-place)
//   1: mask_scaled  (float)
// Push constants: 7 ints + 1 float + 3 floats = 44 B
//   (roi_in_x, roi_in_y, roi_in_width,
//    roi_ms_x, roi_ms_y, roi_ms_width, roi_ms_height,
//    opacity, color_x, color_y, color_z).

#include "dt_vulkan_common.h"

kernel void retouch_fill(
    global       float4 *in_buf,
    global const float  *mask_scaled,
    const int roi_in_x, const int roi_in_y, const int roi_in_width,
    const int roi_ms_x, const int roi_ms_y,
    const int roi_ms_width, const int roi_ms_height,
    const float opacity,
    const float color_x, const float color_y, const float color_z)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= roi_ms_width || y >= roi_ms_height) return;

  const int mi = y * roi_ms_width + x;
  const int di = (y + roi_ms_y - roi_in_y) * roi_in_width + (x + roi_ms_x - roi_in_x);
  const float f = clamp(mask_scaled[mi] * opacity, 0.f, 1.f);
  const float w = in_buf[di].w;
  float4 fill = (float4)(color_x, color_y, color_z, 0.f);
  float4 v = in_buf[di] * (1.f - f) + fill * f;
  v.w = w;
  in_buf[di] = v;
}
