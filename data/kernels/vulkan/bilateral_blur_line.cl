// Vulkan port of bilateral.cl :: kernel blur_line.
//
// Separable 5-tap (1-4-6-4-1 binomial) blur along one axis of the
// 3-D bilateral grid. Each work-item processes one full line:
// dispatched (size1, size2, 1) globally.
//
// The host dispatches this kernel twice (X then Y blur), passing
// different (offset1, offset2, offset3, size1, size2, size3)
// configurations to walk the grid in the right axis order.
//
// Binding layout (2 storage buffers):
//   0: ibuf  — source grid (float)
//   1: obuf  — destination grid (float)
// Push constants: 6 ints = 24 bytes.

#include "dt_vulkan_common.h"

kernel void bilateral_blur_line(global const float *ibuf,
                                global       float *obuf,
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

  // 1-4-6-4-1 / 16. Pre-divided coefficients.
  const float w0 = 6.0f / 16.0f;
  const float w1 = 4.0f / 16.0f;
  const float w2 = 1.0f / 16.0f;

  int index = k * offset1 + j * offset2;

  float tmp1 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * ibuf[index + offset3] + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  float tmp2 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp1) + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  for(int i = 2; i < size3 - 2; i++)
  {
    const float tmp3 = ibuf[index];
    obuf[index] = ibuf[index] * w0
                + w1 * (ibuf[index + offset3]     + tmp2)
                + w2 * (ibuf[index + 2 * offset3] + tmp1);
    index += offset3;
    tmp1 = tmp2;
    tmp2 = tmp3;
  }
  const float tmp3 = ibuf[index];
  obuf[index] = ibuf[index] * w0 + w1 * (ibuf[index + offset3] + tmp2) + w2 * tmp1;
  index += offset3;
  obuf[index] = ibuf[index] * w0 + w1 * tmp3 + w2 * tmp2;
}
