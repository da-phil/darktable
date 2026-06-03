// Vulkan port of basic.cl :: ashift_lanczos2.
//
// Same scaffold as ashift_bicubic but with a 4×4 Lanczos-2 kernel
// (kwidth=2, vk_interpolation_lanczos(2, t)).

#include "dt_vulkan_common.h"

kernel void ashift_lanczos2(global const float4 *in,
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

  const int kwidth = 2;

  float pout0 = ((float)(roi_out_x + x) + clip_x) / out_scale;
  float pout1 = ((float)(roi_out_y + y) + clip_y) / out_scale;
  float pin[3];
  for(int i = 0; i < 3; i++)
    pin[i] = homograph[3*i+0] * pout0
           + homograph[3*i+1] * pout1
           + homograph[3*i+2];
  pin[0] = (pin[0] / pin[2]) * in_scale - (float)roi_in_x;
  pin[1] = (pin[1] / pin[2]) * in_scale - (float)roi_in_y;

  const float rx = pin[0];
  const float ry = pin[1];
  const int tx = (int)rx;
  const int ty = (int)ry;

  float4 pixel = (float4)(0.0f);
  float weight = 0.0f;
  for(int jj = 1 - kwidth; jj <= kwidth; jj++)
    for(int ii = 1 - kwidth; ii <= kwidth; ii++)
    {
      const int i = tx + ii;
      const int j = ty + jj;
      const float wx = vk_interpolation_lanczos(2.0f, (float)i - rx);
      const float wy = vk_interpolation_lanczos(2.0f, (float)j - ry);
      const float w = wx * wy;
      const int mi = vk_clip_mirror(i, iwidth  - 1);
      const int mj = vk_clip_mirror(j, iheight - 1);
      pixel += in[mj * iwidth + mi] * (float4)(w);
      weight += w;
    }
  pixel = (tx >= 0 && ty >= 0 && tx < iwidth && ty < iheight)
          ? pixel / weight : (float4)(0.0f);
  out[y * width + x] = pixel;
}
