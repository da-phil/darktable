// Vulkan port of basic.cl :: interpolation_resample.
//
// Generic non-1:1 resample (up/down-scale) with a precomputed
// separable filter plan. The OpenCL version processes the image
// column-wise using workgroup-local memory + a recursive reduction
// for the vertical convolution. The Vulkan port instead does the
// full separable convolution as a single per-output-pixel gather:
//   out[x,y] = Σ_iy vkernel[iy] · (Σ_ix hkernel[ix] · in[vindex[iy],
//                                                          hindex[ix]])
// which is mathematically identical (separable kernel = outer
// product) and needs no local memory or barriers. For typical tap
// counts (lanczos3 = 6×6 = 36 MACs/pixel) the arithmetic cost is
// modest and the simpler dispatch shape is a better fit for the
// one-shot HAL.
//
// The plan tables are built host-side by `_prepare_resampling_plan`
// (shared with the OpenCL path) and uploaded as flat storage buffers.
// `*meta[x*3 + {0,1,2}]` gives the (length, kernel, index) base
// offsets for output column/row x; the index arrays hold pre-clamped
// input coordinates so no border handling is needed in the kernel.
//
// Binding layout (10 storage buffers):
//   0: in       (float4, in_width × in_height)
//   1: out      (float4, width × height)
//   2: hmeta    (int, width × 3)
//   3: vmeta    (int, height × 3)
//   4: hlength  (int, width)
//   5: vlength  (int, height)
//   6: hindex   (int, packed)
//   7: vindex   (int, packed)
//   8: hkernel  (float, packed)
//   9: vkernel  (float, packed)
// Push constants: 4 ints = 16 bytes (width, height, in_width, in_height).

#include "dt_vulkan_common.h"

kernel void interpolation_resample(global const float4 *in,
                                   global       float4 *out,
                                   global const int   *hmeta,
                                   global const int   *vmeta,
                                   global const int   *hlength,
                                   global const int   *vlength,
                                   global const int   *hindex,
                                   global const int   *vindex,
                                   global const float *hkernel,
                                   global const float *vkernel,
                                   const int width,
                                   const int height,
                                   const int in_width,
                                   const int in_height)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  (void)in_height;

  const int hlidx = hmeta[x * 3];
  const int hkidx = hmeta[x * 3 + 1];
  const int hiidx = hmeta[x * 3 + 2];
  const int vlidx = vmeta[y * 3];
  const int vkidx = vmeta[y * 3 + 1];
  const int viidx = vmeta[y * 3 + 2];

  const int hl = hlength[hlidx];
  const int vl = vlength[vlidx];

  float4 acc = (float4)(0.0f);
  for(int iy = 0; iy < vl; iy++)
  {
    const int yy = vindex[viidx + iy];
    float4 hsum = (float4)(0.0f);
    for(int ix = 0; ix < hl; ix++)
    {
      const int xx = hindex[hiidx + ix];
      hsum += in[yy * in_width + xx] * hkernel[hkidx + ix];
    }
    acc += hsum * vkernel[vkidx + iy];
  }
  out[y * width + x] = acc;
}
