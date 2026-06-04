// Vulkan port of src/iop/censorize.c :: pixelate pass.
//
// Each work-item handles one *big-pixel block* (2*pixel_radius
// square). Samples 5 fixed positions inside the block from the input,
// averages them, and writes that solid colour to every output pixel
// inside the block. Mirrors the CPU loop in censorize.c::process
// byte-for-byte (same 5-sample bounding-box + centre pattern).
//
// Image-shortcut port: the OpenCL build doesn't have this kernel —
// the OpenCL `process_cl` was wrapped in `#if FALSE` and unimplemented.
// This is the first GPU acceleration for censorize's pixelate.
//
// Binding layout (2 storage buffers):
//   0: in   (float4)
//   1: out  (float4)
// Push constants: 3 ints = 12 bytes (width, height, pixel_radius).

#include "dt_vulkan_common.h"

kernel void censorize_pixelate(global const float4 *in,
                               global       float4 *out,
                               const int width,
                               const int height,
                               const int pixel_radius)
{
  // One work-item per big-pixel block (i, j).
  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int pr2 = 2 * pixel_radius;
  const int pixels_x = width  / pr2;
  const int pixels_y = height / pr2;
  if(i > pixels_x || j > pixels_y) return;

  // Top-left, centre, bottom-right of the big pixel (CLAMP-to-edge).
  const int tlx = clamp(pr2 * i,                  0, width  - 1);
  const int tly = clamp(pr2 * j,                  0, height - 1);
  const int ccx = clamp(tlx + pixel_radius,       0, width  - 1);
  const int ccy = clamp(tly + pixel_radius,       0, height - 1);
  const int brx = clamp(ccx + pixel_radius,       0, width  - 1);
  const int bry = clamp(ccy + pixel_radius,       0, height - 1);

  const int2 samples[5] = {
    (int2)(tlx, tly), (int2)(brx, tly), (int2)(ccx, ccy),
    (int2)(tlx, bry), (int2)(brx, bry),
  };
  float4 rgb = (float4)(0.0f);
  for(int k = 0; k < 5; k++)
    rgb += in[samples[k].y * width + samples[k].x] / 5.0f;

  // Paint every output pixel in [tlx, brx) × [tly, bry) with rgb.
  for(int jj = tly; jj < bry; jj++)
    for(int ii = tlx; ii < brx; ii++)
      out[jj * width + ii] = rgb;
}
