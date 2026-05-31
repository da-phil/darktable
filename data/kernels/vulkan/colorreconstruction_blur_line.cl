// Vulkan port of colorreconstruction.cl :: colorreconstruction_blur_line.
//
// One axis of the 3D B3-spline blur over the bilateral grid. Each
// work item walks one "line" of length size3, computing a symmetric
// 5-tap (w0=6/16, w1=4/16, w2=1/16) blur with the OpenCL's
// drop-missing-samples boundary handling (no edge mirroring).
//
// Dispatched 2D over (size1, size2); the line traversal happens in
// strides of `offset3` float4 elements along the inner axis. Three
// passes are issued by the host (one per axis) with different
// stride / size triples.
//
// Bindings (2 storage buffers):
//   0: ibuf (float4)
//   1: obuf (float4)
// Push constants: 24 B (offset1, offset2, offset3, size1, size2, size3).

#include "dt_vulkan_common.h"

kernel void colorreconstruction_blur_line(global const float4 *ibuf,
                                          global       float4 *obuf,
                                          const int offset1,
                                          const int offset2,
                                          const int offset3,
                                          const int size1,
                                          const int size2,
                                          const int size3)
{
  const int k = get_global_id(0);
  const int j = get_global_id(1);
  if(k >= size1 || j >= size2) return;

  const float w0 = 6.0f / 16.0f;
  const float w1 = 4.0f / 16.0f;
  const float w2 = 1.0f / 16.0f;
  int index = k * offset1 + j * offset2;

  // i = 0: no left neighbours.
  float4 tmp1 = ibuf[index];
  float4 outv = ibuf[index] * w0
              + ibuf[index + offset3] * w1
              + ibuf[index + 2 * offset3] * w2;
  obuf[index] = outv;
  index += offset3;

  // i = 1: tmp1 is the (i-1) left neighbour; (i-2) dropped.
  float4 tmp2 = ibuf[index];
  outv = ibuf[index] * w0
       + (ibuf[index + offset3] + tmp1) * w1
       + ibuf[index + 2 * offset3] * w2;
  obuf[index] = outv;
  index += offset3;

  // Steady state.
  for(int i = 2; i < size3 - 2; i++)
  {
    const float4 tmp3 = ibuf[index];
    outv = ibuf[index] * w0
         + (ibuf[index + offset3]   + tmp2) * w1
         + (ibuf[index + 2 * offset3] + tmp1) * w2;
    obuf[index] = outv;
    index += offset3;
    tmp1 = tmp2;
    tmp2 = tmp3;
  }

  // i = size3 - 2: (i+2) dropped.
  const float4 tmp3 = ibuf[index];
  outv = ibuf[index] * w0
       + (ibuf[index + offset3] + tmp2) * w1
       + tmp1 * w2;
  obuf[index] = outv;
  index += offset3;

  // i = size3 - 1: (i+1) and (i+2) dropped.
  outv = ibuf[index] * w0 + tmp3 * w1 + tmp2 * w2;
  obuf[index] = outv;
}
