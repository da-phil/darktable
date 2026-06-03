// Vulkan port of basic.cl :: ashift_bilinear.
//
// Perspective backtransform with manual 4-tap bilinear sampling.
// The OpenCL kernel uses `read_imagef(in, samplerf, (rx+0.5, ry+0.5))`
// for hardware bilinear (CLK_ADDRESS_CLAMP — out-of-bounds returns 0).
// The Vulkan port does the 4-tap blend in-kernel:
//   ix = floor(rx); iy = floor(ry);
//   fx = rx - ix;   fy = ry - iy;
//   out = (1-fx)(1-fy)·in[ix,iy] + fx(1-fy)·in[ix+1,iy]
//       + (1-fx)fy ·in[ix,iy+1] + fx·fy   ·in[ix+1,iy+1]
// with each corner zeroed if its integer coord is out of [0, iwidth/iheight).
// The outer "valid centre" check mirrors process_cl byte-for-byte:
// when the floor of (rx, ry) is itself out of range, the entire pixel
// is set to 0 (transparent black).
//
// Binding layout (3 storage buffers):
//   0: in        (float4)
//   1: out       (float4)
//   2: homograph (float, 9 entries — 3×3 perspective matrix)
// Push constants: 6 ints + 4 floats = 40 bytes
//   (width, height, iwidth, iheight, roi_in_x, roi_in_y,
//    in_scale, out_scale, clip_x, clip_y).

#include "dt_vulkan_common.h"

kernel void ashift_bilinear(global const float4 *in,
                            global       float4 *out,
                            global const float  *homograph,
                            const int width,
                            const int height,
                            const int iwidth,
                            const int iheight,
                            const int roi_in_x,
                            const int roi_in_y,
                            const int roi_out_x,
                            const int roi_out_y,
                            const float in_scale,
                            const float out_scale,
                            const float clip_x,
                            const float clip_y)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  // Build the output-image-space coordinate, then homography → input.
  float pout0 = ((float)(roi_out_x + x) + clip_x) / out_scale;
  float pout1 = ((float)(roi_out_y + y) + clip_y) / out_scale;
  // pout2 = 1
  float pin[3];
  for(int i = 0; i < 3; i++)
  {
    pin[i] = homograph[3 * i + 0] * pout0
           + homograph[3 * i + 1] * pout1
           + homograph[3 * i + 2];
  }
  pin[0] = (pin[0] / pin[2]) * in_scale - (float)roi_in_x;
  pin[1] = (pin[1] / pin[2]) * in_scale - (float)roi_in_y;

  const float rx = pin[0];
  const float ry = pin[1];
  const int tx = (int)rx;
  const int ty = (int)ry;

  float4 pixel = (float4)(0.0f);
  if(tx >= 0 && ty >= 0 && tx < iwidth && ty < iheight)
  {
    // Hardware-bilinear equivalent: 4-tap with CLAMP-to-zero borders.
    const int ix0 = (int)floor(rx);
    const int iy0 = (int)floor(ry);
    const float fx = rx - (float)ix0;
    const float fy = ry - (float)iy0;
    const int ix1 = ix0 + 1;
    const int iy1 = iy0 + 1;

    const float4 c00 = (ix0 >= 0 && ix0 < iwidth && iy0 >= 0 && iy0 < iheight)
                       ? in[iy0 * iwidth + ix0] : (float4)(0.0f);
    const float4 c10 = (ix1 >= 0 && ix1 < iwidth && iy0 >= 0 && iy0 < iheight)
                       ? in[iy0 * iwidth + ix1] : (float4)(0.0f);
    const float4 c01 = (ix0 >= 0 && ix0 < iwidth && iy1 >= 0 && iy1 < iheight)
                       ? in[iy1 * iwidth + ix0] : (float4)(0.0f);
    const float4 c11 = (ix1 >= 0 && ix1 < iwidth && iy1 >= 0 && iy1 < iheight)
                       ? in[iy1 * iwidth + ix1] : (float4)(0.0f);
    pixel = (1.0f - fx) * (1.0f - fy) * c00
          + (       fx) * (1.0f - fy) * c10
          + (1.0f - fx) * (       fy) * c01
          + (       fx) * (       fy) * c11;
  }
  out[y * width + x] = pixel;
}
