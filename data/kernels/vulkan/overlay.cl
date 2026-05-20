// Vulkan port of overlay.cl::overlay_blend.
//
// Alpha-blend a Cairo ARGB32 overlay onto an RGBA float image.
// The original CL kernel takes image2d_t in/out and a byte buffer
// for the overlay. The buffer-only HAL drops the image bindings;
// the ARGB byte buffer is read as a packed uint storage buffer
// (one uint per pixel — Cairo ARGB32 stride is always 4-byte aligned
// and x*4 is naturally word-aligned, so each pixel is exactly one
// uint and no cross-word shuffles are needed).
//
// Bindings (3 storage buffers):
//   0: in    (float4 RGBA)
//   1: argb  (uint, packed B|G|R|A little-endian — matches Cairo's
//             memory layout on little-endian hosts)
//   2: out   (float4 RGBA)
//
// Push constants: 16 bytes (2 ints + 1 float + 1 int).

#include "dt_vulkan_common.h"

kernel void overlay_blend(global const float4 *in,
                          global const uint   *overlay_argb,
                          global       float4 *out,
                          const int   width,
                          const int   height,
                          const float opacity,
                          const int   stride)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);
  const float4 i = in[idx];

  // One uint per pixel — stride is in bytes, divide by 4 for uint index.
  const int word_idx = (y * stride + x * 4) / 4;
  const uint v = overlay_argb[word_idx];
  const float b = (float)( v        & 0xffu) / 255.0f;
  const float g = (float)((v >>  8) & 0xffu) / 255.0f;
  const float r = (float)((v >> 16) & 0xffu) / 255.0f;
  const float a = (float)((v >> 24) & 0xffu) / 255.0f * opacity;

  float4 o;
  o.x = (1.0f - a) * i.x + opacity * r;
  o.y = (1.0f - a) * i.y + opacity * g;
  o.z = (1.0f - a) * i.z + opacity * b;
  o.w = i.w;
  out[idx] = o;
}
