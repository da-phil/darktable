// Vulkan port of basic.cl :: rawoverexposed_mark_solid.
//
// Same shape as mark_cfa but uses a single user-chosen solid_color
// for all clipped pixels (no per-CFA-slot lookup), so the colors[]
// array drops out of the binding and one float4 of PC is added.
//
// Binding layout (5 storage buffers): in, out, coord, raw, xtrans.

#include "dt_vulkan_common.h"

kernel void rawoverexposed_mark_solid(
    global const float4 *in_buf,
    global       float4 *out_buf,
    global const float  *coord_buf,
    global const uint   *raw_buf,
    global const uint   *xtrans_flat,
    const int width, const int height,
    const int raw_width, const int raw_height,
    const uint filters,
    const uint thr0, const uint thr1, const uint thr2, const uint thr3,
    const float scx, const float scy, const float scz, const float scw)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int piwidth = 2 * width;
  const int ci = y * piwidth + 2 * x;
  const int raw_x = (int)coord_buf[ci];
  const int raw_y = (int)coord_buf[ci + 1];

  if(raw_x < 0 || raw_y < 0 || raw_x >= raw_width || raw_y >= raw_height)
    return;

  const uint raw_pixel = raw_buf[raw_y * raw_width + raw_x];
  const int c = (filters == 9u)
                    ? vk_FCxtrans(raw_y, raw_x, xtrans_flat)
                    : vk_FC(raw_y, raw_x, filters);

  const uint thr = (c == 0) ? thr0 : ((c == 1) ? thr1 : ((c == 2) ? thr2 : thr3));
  if(raw_pixel < thr) return;

  float4 pixel = fmax((float4)(0.0f, 0.0f, 0.0f, 0.0f), in_buf[y * width + x]);
  pixel.xyz = (float3)(scx, scy, scz);
  out_buf[y * width + x] = pixel;
}
