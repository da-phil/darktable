// Vulkan port of denoiseprofile.cl :: denoiseprofile_synthesize.
//
// Per-pixel wavelet reconstruction: clip detail magnitudes below a
// per-channel threshold then boost-mix back into the coarse layer.
//
// Binding layout (3 storage buffers):
//   0: coarse  (float4)
//   1: detail  (float4)
//   2: out     (float4)
// Push constants: 4 + 4*4 + 4*4 = 40 B (width, height, thrs0..3, boost0..3).

#include "dt_vulkan_common.h"

kernel void denoiseprofile_synthesize(
    global const float4 *coarse_buf,
    global const float4 *detail_buf,
    global       float4 *out_buf,
    const int width, const int height,
    const float t0, const float t1, const float t2, const float t3,
    const float b0, const float b1, const float b2, const float b3)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;

  const float4 thrs  = (float4)(t0, t1, t2, t3);
  const float4 boost = (float4)(b0, b1, b2, b3);

  const float4 c = coarse_buf[idx];
  const float4 d = detail_buf[idx];
  const float4 absd = (float4)(fabs(d.x), fabs(d.y), fabs(d.z), fabs(d.w));
  const float4 abs_amt = fmax((float4)(0.0f), absd - thrs);
  const float4 signs = (float4)(d.x < 0.0f ? -1.0f : 1.0f,
                                d.y < 0.0f ? -1.0f : 1.0f,
                                d.z < 0.0f ? -1.0f : 1.0f,
                                d.w < 0.0f ? -1.0f : 1.0f);
  const float4 amount = signs * abs_amt;
  float4 sum = c + boost * amount;
  sum.w = c.w;
  out_buf[idx] = sum;
}
