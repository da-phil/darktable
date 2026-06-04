// Vulkan port of basic.cl :: highlights_4f_clip.
//
// Per-pixel highlight clip on the post-demosaic 4-channel float
// path. The OpenCL kernel reads `in` via `readpixel` with sampler-
// clamp integer coords → image-shortcut, translates to a flat float4
// storage buffer. Image is already demosaiced so this is just a
// simple per-channel clamp; nothing to recover.
//
// The Bayer / X-Trans 1f variants (highlights_1f_clip,
// highlights_1f_lch_bayer, highlights_1f_lch_xtrans, the LAPLACIAN
// chroma reconstruction's 8+ kernels, the OPPOSED / INPAINT /
// SEGMENTS modes) all need single-channel uint16 / float input
// buffers and the FC()/FCxtrans() pattern lookup that aren't wired
// into the Vulkan pipeline integration yet. commit_params gates
// piece->process_vk_ready off for those (the §10.2 predictive
// pattern — same shape as rawprepare's 1f gate).
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 2 ints + 1 float = 12 bytes (width, height, clip).

#include "dt_vulkan_common.h"

kernel void highlights_4f_clip(global const float4 *in,
                               global       float4 *out,
                               const int width,
                               const int height,
                               const float clip)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;
  float4 pixel = in[idx];
  pixel.x = fmin(clip, pixel.x);
  pixel.y = fmin(clip, pixel.y);
  pixel.z = fmin(clip, pixel.z);
  out[idx] = pixel;
}
