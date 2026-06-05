// Vulkan port of retouch.cl :: retouch_copy_buffer_to_buffer_masked
// AND retouch_copy_image_to_buffer_masked. With image2d_t -> flat
// float4 buffer in VK both OpenCL kernels collapse to the same
// per-pixel mix, so one kernel covers both entry points.
//
// Binding layout (3 storage buffers):
//   0: buffer_src   (float4) — source pixels (was image2d_t for the
//                              image variant; identical in VK)
//   1: buffer_dest  (float4, in-place) — destination layer
//   2: mask_scaled  (float)             — per-mask-pixel weight
// Push constants: 6 ints + 1 float = 28 B (roi_dest_x, roi_dest_y,
//   roi_dest_width, roi_ms_x, roi_ms_y, roi_ms_width, roi_ms_height,
//   opacity).

#include "dt_vulkan_common.h"

kernel void retouch_copy_buffer_to_buffer_masked(
    global const float4 *src,
    global       float4 *dest,
    global const float  *mask_scaled,
    const int roi_dest_x, const int roi_dest_y, const int roi_dest_width,
    const int roi_ms_x, const int roi_ms_y,
    const int roi_ms_width, const int roi_ms_height,
    const float opacity)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= roi_ms_width || y >= roi_ms_height) return;

  const int di = (y + roi_ms_y - roi_dest_y) * roi_dest_width
               + (x + roi_ms_x - roi_dest_x);
  const int si = y * roi_ms_width + x;
  const int mi = si;

  const float f = clamp(mask_scaled[mi] * opacity, 0.f, 1.f);
  const float w = dest[di].w;

  float4 v = dest[di] * (1.f - f) + src[si] * f;
  v.w = w;
  dest[di] = v;
}
