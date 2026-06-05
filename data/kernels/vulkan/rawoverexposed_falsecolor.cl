// Vulkan port of basic.cl :: rawoverexposed_falsecolor.
//
// Zeroes the CFA-channel of clipped pixels (R, G, or B), producing
// the false-color visualisation. No solid_color or colors[] needed.
//
// Binding layout (5 storage buffers): in, out, coord, raw, xtrans.

#include "dt_vulkan_common.h"

kernel void rawoverexposed_falsecolor(
    global const float4 *in_buf,
    global       float4 *out_buf,
    global const float  *coord_buf,
    global const uint   *raw_buf,
    global const uint   *xtrans_flat,
    const int width, const int height,
    const int raw_width, const int raw_height,
    const uint filters,
    const uint thr0, const uint thr1, const uint thr2, const uint thr3)
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
  if(c == 2)      pixel.z = 0.0f;
  else if(c == 1) pixel.y = 0.0f;
  else if(c == 0) pixel.x = 0.0f;

  out_buf[y * width + x] = pixel;
}
