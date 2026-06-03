// Vulkan port of basic.cl :: rawprepare_4f.
//
// Per-pixel black-level subtraction + per-channel divisor for the
// 4-channel float pre-demosaic path. The OpenCL kernel reads `in`
// via `readpixel` (sampler-clamp integer coords) at (x + cx, y + cy)
// — pure image-shortcut, so the binding becomes a flat float4 buffer.
//
// The 1-channel Bayer / X-Trans rawprepare paths (uint16 / float
// single-channel) need single-channel input buffers and the FC()
// pattern lookup; commit_params gates piece->process_vk_ready off
// for those (the §10.2 predictive pattern) so this kernel only runs
// when the pipeline already has float4 input — typically HDR DNG /
// 16-bit TIFF / PNG / etc.
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 2 ints + 4 ints + 4 floats + 4 floats = 56 bytes
//   (width, height, cx, cy, rx, ry, black0..3, div0..3).
//   (rx, ry are unused by 4f but kept to mirror the OpenCL signature;
//    they're used by the 1f variants for the Bayer FC() lookup origin.)

#include "dt_vulkan_common.h"

kernel void rawprepare_4f(global const float4 *in,
                          global       float4 *out,
                          const int width,
                          const int height,
                          const int cx,
                          const int cy,
                          const int rx,
                          const int ry,
                          const int in_width,
                          const float b0, const float b1,
                          const float b2, const float b3,
                          const float d0, const float d1,
                          const float d2, const float d3)
{
  (void)rx; (void)ry;
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const float4 black = (float4)(b0, b1, b2, b3);
  const float4 div   = (float4)(d0, d1, d2, d3);

  float4 pixel = in[(y + cy) * in_width + (x + cx)];
  pixel.xyz = (pixel.xyz - black.xyz) / div.xyz;

  out[y * width + x] = pixel;
}
