// Vulkan port of basic.cl :: rawoverexposed_mark_cfa.
//
// Marks raw-clipped pixels with their CFA color. Inputs:
//   - the post-pipe float4 image (in/out)
//   - the original raw uint16 buffer (uploaded as flat uint, one
//     uint per pixel; host pre-promotes uint16 -> uint32)
//   - a coordinate buffer (one float2 per output pixel: where the
//     pixel came from in the raw, post-distort-backtransform)
//   - xtrans 6x6 byte matrix (flat uint, 36 entries)
//
// Push constants (52 bytes) carry width/height, raw dims, filters,
// the 4 per-channel uint thresholds, and the 4 CFA float4 colors
// (one per Bayer/X-Trans color slot).
//
// Binding layout (5 storage buffers):
//   0: in    (float4)
//   1: out   (float4)
//   2: coord (float, 2 floats per pixel)
//   3: raw   (uint, one per raw pixel)
//   4: xtrans (uint, 36 entries; ignored when filters != 9)

#include "dt_vulkan_common.h"

kernel void rawoverexposed_mark_cfa(
    global const float4 *in_buf,
    global       float4 *out_buf,
    global const float  *coord_buf,
    global const uint   *raw_buf,
    global const uint   *xtrans_flat,
    const int width, const int height,
    const int raw_width, const int raw_height,
    const uint filters,
    const uint thr0, const uint thr1, const uint thr2, const uint thr3,
    const float c0x, const float c0y, const float c0z, const float c0w,
    const float c1x, const float c1y, const float c1z, const float c1w,
    const float c2x, const float c2y, const float c2z, const float c2w,
    const float c3x, const float c3y, const float c3z, const float c3w)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int piwidth = 2 * width;
  const int ci = y * piwidth + 2 * x;
  const int raw_x = (int)coord_buf[ci];
  const int raw_y = (int)coord_buf[ci + 1];

  // OOB / sub-threshold: leave dev_out untouched (host pre-copied
  // dev_in over it, matching the OpenCL early-return semantics).
  if(raw_x < 0 || raw_y < 0 || raw_x >= raw_width || raw_y >= raw_height)
    return;

  const uint raw_pixel = raw_buf[raw_y * raw_width + raw_x];
  const int c = (filters == 9u)
                    ? vk_FCxtrans(raw_y, raw_x, xtrans_flat)
                    : vk_FC(raw_y, raw_x, filters);

  const uint thr = (c == 0) ? thr0 : ((c == 1) ? thr1 : ((c == 2) ? thr2 : thr3));
  if(raw_pixel < thr) return;

  float4 pixel = fmax((float4)(0.0f, 0.0f, 0.0f, 0.0f), in_buf[y * width + x]);
  const int ci3 = c & 3;
  float4 cfa_color;
  if(ci3 == 0)      cfa_color = (float4)(c0x, c0y, c0z, c0w);
  else if(ci3 == 1) cfa_color = (float4)(c1x, c1y, c1z, c1w);
  else if(ci3 == 2) cfa_color = (float4)(c2x, c2y, c2z, c2w);
  else              cfa_color = (float4)(c3x, c3y, c3z, c3w);

  pixel.xyz = cfa_color.xyz;
  out_buf[y * width + x] = pixel;
}
