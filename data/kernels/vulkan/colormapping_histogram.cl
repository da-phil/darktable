// Vulkan port of extended.cl :: colormapping_histogram.
//
// Image-shortcut port: the OpenCL kernel reads in/out as image2d_t but
// only ever uses (int2)(x, y) sampler-clamp coords with `sampleri` —
// no actual filtering — so the in/out bindings translate to flat
// float4 storage buffers exactly the same way as overlay / sigmoid /
// agx have already done.
//
// Computes the equalised dL = 0.5 * ((L*(1-eq) + ihist[hist[L*HISTN/100]]*eq) - L) + 50
// then clamps to [0, 100] and writes into out.x; out.yzw = 0.
//
// Binding layout (4 storage buffers):
//   0: in            (float4)  — Lab input
//   1: out           (float4)  — dL written into out.x
//   2: target_hist   (int*)    — HISTN entries
//   3: source_ihist  (float*)  — HISTN entries
// Push constants: 2 ints + 1 float = 12 bytes (width, height, equalization).
// HISTN = 1<<11 = 2048 (matches extended.cl's `#define HISTN`).

#include "dt_vulkan_common.h"

#define HISTN (1 << 11)

kernel void colormapping_histogram(global const float4 *in,
                                   global       float4 *out,
                                   global const int    *target_hist,
                                   global const float  *source_ihist,
                                   const int width,
                                   const int height,
                                   const float equalization)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int idx = y * width + x;
  const float L = in[idx].x;

  const int bin = (int)clamp((float)HISTN * L / 100.0f,
                             0.0f, (float)(HISTN - 1));
  float dL = 0.5f * ((L * (1.0f - equalization)
                      + source_ihist[target_hist[bin]] * equalization) - L)
             + 50.0f;
  dL = clamp(dL, 0.0f, 100.0f);

  out[idx] = (float4)(dL, 0.0f, 0.0f, 0.0f);
}

#undef HISTN
