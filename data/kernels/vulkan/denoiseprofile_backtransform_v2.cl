// Vulkan port of denoiseprofile.cl :: denoiseprofile_backtransform_v2.
//
// Inverse of the generalised Anscombe transform applied per-pixel.
// Restores signal-dependent noise scaling after denoising.
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 4 + 4*4 + 4*4 + 4*4 + 4 + 4*4 = 72 B
//   (width, height, a0..3, p0..3, b0..3, bias, wb0..3).

#include "dt_vulkan_common.h"

kernel void denoiseprofile_backtransform_v2(
    global const float4 *in_buf,
    global       float4 *out_buf,
    const int width, const int height,
    const float a0, const float a1, const float a2, const float a3,
    const float p0, const float p1, const float p2, const float p3,
    const float b0, const float b1, const float b2, const float b3,
    const float bias,
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

  float4 px = in_buf[idx];
  const float alpha = px.w;

  px = fmax((float4)(0.0f), px);
  const float4 delta = px * px + (float4)bias;
  float4 sqrt_a;
  sqrt_a.x = sqrt(a.x); sqrt_a.y = sqrt(a.y);
  sqrt_a.z = sqrt(a.z); sqrt_a.w = sqrt(a.w);
  const float4 denominator = 4.0f / (sqrt_a * (2.0f - p));
  float4 delta_clamped = fmax((float4)(0.0f), delta);
  float4 sqrt_delta;
  sqrt_delta.x = sqrt(delta_clamped.x); sqrt_delta.y = sqrt(delta_clamped.y);
  sqrt_delta.z = sqrt(delta_clamped.z); sqrt_delta.w = sqrt(delta_clamped.w);
  const float4 z1 = (px + sqrt_delta) / denominator;

  float4 raised;
  raised.x = pow(z1.x, 1.0f / (1.0f - p.x / 2.0f));
  raised.y = pow(z1.y, 1.0f / (1.0f - p.y / 2.0f));
  raised.z = pow(z1.z, 1.0f / (1.0f - p.z / 2.0f));
  raised.w = pow(z1.w, 1.0f / (1.0f - p.w / 2.0f));
  px = fmax(raised - b, (float4)(0.0f));
  px = px * wb;
  px.w = alpha;
  out_buf[idx] = px;
}
