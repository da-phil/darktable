// Vulkan port of locallaplacian.cl :: process_curve.
//
// Applies the per-gamma S-curve to the padded monochrome input.
// Note the OpenCL kernel takes input as `image2d_t` and reads via
// `readpixel`, but only writes pixel.x — and the output here is a
// monochrome float buffer (the `processed[k][0]` allocation). So the
// Vulkan port reads a float buffer (matching pad_input's output)
// and writes a float buffer.
//
// Binding layout (2 storage buffers):
//   0: in   (float)
//   1: out  (float)
// Push constants: 2 ints + 5 floats = 28 bytes
//   (wd, ht, g, sigma, shadows, highlights, clarity).

#include "dt_vulkan_common.h"

static inline float ll_curve(const float x,
                             const float g, const float sigma,
                             const float shadows, const float highlights,
                             const float clarity)
{
  const float c = x - g;
  const float ssigma = c > 0.0f ? sigma : -sigma;
  const float shadhi = c > 0.0f ? shadows : highlights;
  float val;
  if(fabs(c) > 2.0f * sigma)
  {
    val = g + ssigma + shadhi * (c - ssigma);
  }
  else
  {
    const float t  = clamp(c / (2.0f * ssigma), 0.0f, 1.0f);
    const float t2 = t * t;
    const float mt = 1.0f - t;
    val = g + ssigma * 2.0f * mt * t + t2 * (ssigma + ssigma * shadhi);
  }
  // midtone local contrast (dt_fast_expf with exact exp; the
  // approximation isn't needed on modern GPUs).
  val += clarity * c * exp(-c * c / (2.0f * sigma * sigma / 3.0f));
  return val;
}

kernel void ll_process_curve(global const float *in,
                             global       float *out,
                             const int wd,
                             const int ht,
                             const float g,
                             const float sigma,
                             const float shadows,
                             const float highlights,
                             const float clarity)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= wd || y >= ht) return;

  const float v = in[y * wd + x];
  out[y * wd + x] = ll_curve(v, g, sigma, shadows, highlights, clarity);
}
