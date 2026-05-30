// Vulkan port of blurs.cl :: convolve_sparse.
//
// Sparse 2D convolution: only the n_entries non-negligible kernel
// entries are applied. Each entry is a (dx, dy) offset + a weight.
// Matches the OpenCL kernel byte-for-byte (CLAMP_TO_EDGE on the
// shifted coords; alpha taken from the centre pixel).
//
// Bindings (5 storage buffers):
//   0: in        (float4 RGBA)
//   1: offsets_x (n_entries ints)
//   2: offsets_y (n_entries ints)
//   3: values    (n_entries floats)
//   4: out       (float4 RGBA)
// Push constants: 12 B (width, height, n_entries).

#include "dt_vulkan_common.h"

kernel void convolve_sparse(global const float4 *in,
                            global const int    *offsets_x,
                            global const int    *offsets_y,
                            global const float  *values,
                            global       float4 *out,
                            const int width,
                            const int height,
                            const int n_entries)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int idx = idx2d(x, y, width);

  const float4 pix_in = in[idx];
  float4 acc = (float4)0.0f;
  for(int i = 0; i < n_entries; i++)
  {
    const int jj = clamp(x + offsets_x[i], 0, width - 1);
    const int ii = clamp(y + offsets_y[i], 0, height - 1);
    acc += values[i] * in[idx2d(jj, ii, width)];
  }
  acc.w = pix_in.w;
  out[idx] = acc;
}
