// Vulkan port of denoiseprofile.cl :: denoiseprofile_precondition_v2.
//
// Generalised Anscombe transform that maps per-channel signal-
// dependent noise to unit variance. Per-pixel.
//   t = max(2 * pow(max(0, pixel/wb + b), 1 - p/2) / ((-p + 2) * sqrt(a)), 0)
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 4 + 4*4 + 4*4 + 4*4 + 4*4 = 68 B
//   (width, height, a0..3, p0..3, b0..3, wb0..3).

#include "dt_vulkan_common.h"

kernel void denoiseprofile_precondition_v2(
    global const float4 *in_buf,
    global       float4 *out_buf,
    const int width, const int height,
    const float a0, const float a1, const float a2, const float a3,
    const float p0, const float p1, const float p2, const float p3,
    const float b0, const float b1, const float b2, const float b3,
    const float wb0, const float wb1, const float wb2, const float wb3)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = y * width + x;

  const float4 a  = (float4)(a0, a1, a2, a3);
  const float4 p  = (float4)(p0, p1, p2, p3);
  const float4 b  = (float4)(b0, b1, b2, b3);
  const float4 wb = (float4)(wb0, wb1, wb2, wb3);

  float4 pixel = in_buf[idx];
  const float alpha = pixel.w;

  float4 inner = fmax((float4)(0.0f), pixel / wb + b);
  float4 raised;
  raised.x = pow(inner.x, 1.0f - p.x / 2.0f);
  raised.y = pow(inner.y, 1.0f - p.y / 2.0f);
  raised.z = pow(inner.z, 1.0f - p.z / 2.0f);
  raised.w = pow(inner.w, 1.0f - p.w / 2.0f);
  float4 sqrt_a;
  sqrt_a.x = sqrt(a.x); sqrt_a.y = sqrt(a.y);
  sqrt_a.z = sqrt(a.z); sqrt_a.w = sqrt(a.w);

  float4 t = fmax(2.0f * raised / ((-p + 2.0f) * sqrt_a), (float4)(0.0f));
  t.w = alpha;
  out_buf[idx] = t;
}
