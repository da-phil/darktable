// Vulkan port of sharpen.cl :: sharpen_mix.
//
// Unsharp mask with threshold: delta = orig.L - blurred.L; amount
// applies only where |delta| > thrs and is shaped by copysign so the
// soft-threshold preserves sign. Outermost rad-pixel border is left
// unchanged (matches the OpenCL skip condition).
//
// Bindings (3 storage buffers):
//   0: in_a  (float4 Lab, original)
//   1: in_b  (float4 Lab, blurred reference; only .x is read)
//   2: out   (float4 Lab)
// Push constants: 20 B (width, height, sharpen, threshold, rad).

#include "dt_vulkan_common.h"

kernel void sharpen_mix(global const float4 *in_a,
                        global const float4 *in_b,
                        global       float4 *out,
                        const int   width,
                        const int   height,
                        const float sharpen,
                        const float threshold,
                        const int   rad)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  float4 pixel = in_a[idx];
  if(x >= rad && y >= rad && x < width - rad && y < height - rad)
  {
    const float blurredx = in_b[idx].x;
    const float delta = pixel.x - blurredx;
    const float amount = sharpen * copysign(fmax(0.0f, fabs(delta) - threshold), delta);
    pixel.x = pixel.x + amount;
  }
  out[idx] = pixel;
}
