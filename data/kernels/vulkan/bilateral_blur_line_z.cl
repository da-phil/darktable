// Vulkan port of bilateral.cl :: kernel blur_line_z.
//
// Signed (1-0-(-1)-style) 5-tap blur along the Z (luminance) axis
// of the bilateral grid. Identical structure to bilateral_blur_line
// but with the derivative-shaped coefficient set (w0=0, w1=4/16,
// w2=2/16) — see the OpenCL kernel header for the original maths.
//
// Binding layout (2 storage buffers):
//   0: ibuf  — source grid (float)
//   1: obuf  — destination grid (float)
// Push constants: 6 ints = 24 bytes.

#include "dt_vulkan_common.h"

kernel void bilateral_blur_line_z(global const float *ibuf,
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

  const float w1 = 4.0f / 16.0f;
  const float w2 = 2.0f / 16.0f;

  int index = k * offset1 + j * offset2;

  float tmp1 = ibuf[index];
  obuf[index] = w1 * ibuf[index + offset3] + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  float tmp2 = ibuf[index];
  obuf[index] = w1 * (ibuf[index + offset3] - tmp1) + w2 * ibuf[index + 2 * offset3];
  index += offset3;
  for(int i = 2; i < size3 - 2; i++)
  {
    const float tmp3 = ibuf[index];
    obuf[index] = w1 * (ibuf[index + offset3]     - tmp2)
                + w2 * (ibuf[index + 2 * offset3] - tmp1);
    index += offset3;
    tmp1 = tmp2;
    tmp2 = tmp3;
  }
  const float tmp3 = ibuf[index];
  obuf[index] = w1 * (ibuf[index + offset3] - tmp2) - w2 * tmp1;
  index += offset3;
  obuf[index] = -w1 * tmp3 - w2 * tmp2;
}
