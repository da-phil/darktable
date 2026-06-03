// Vulkan port of basic.cl :: clip_rotate_lanczos3.
//
// Same homography backtransform as clip_rotate_bilinear, with a 4×4
// Lanczos-3 reconstruction (kwidth = 3). The OpenCL kernel uses
// `Areadpixel` (samplerA = nearest + ADDRESS_NONE) at integer coords;
// the Vulkan twin reads flat buffers with `vk_clip_mirror` edge
// handling.

#include "dt_vulkan_common.h"

static inline void vk_clip_backtransform(float *px, float *py,
                                         const float m0, const float m1,
                                         const float m2, const float m3,
                                         const float kx, const float ky)
{
  float qx = *px, qy = *py;
  qy /= 1.0f + qx * kx;
  qx /= 1.0f + qy * ky;
  *px = qx * m0 + qy * m1;
  *py = qx * m2 + qy * m3;
}

static inline void vk_clip_keystone_backtransform(float *px, float *py,
                                                  const float ksx, const float ksy,
                                                  const float kax, const float kay,
                                                  const float ma0, const float ma1,
                                                  const float ma2, const float ma3,
                                                  const float mb0, const float mb1)
{
  const float xx = *px - ksx;
  const float yy = *py - ksy;
  const float div = (ma2 * xx - ma0 * yy) * mb1
                  + (ma1 * yy - ma3 * xx) * mb0
                  + ma0 * ma3 - ma1 * ma2;
  *px =  (ma3 * xx - ma1 * yy) / div + kax;
  *py = -(ma2 * xx - ma0 * yy) / div + kay;
}

kernel void clip_rotate_lanczos3(
    global const float4 *in,
    global       float4 *out,
    global const float  *_unused,
    const int width, const int height, const int in_width, const int in_height,
    const int roi_in_x, const int roi_in_y, const int flip,
    const float roi_out_x, const float roi_out_y,
    const float scale_in,  const float scale_out,
    const float t_x,  const float t_y, const float k_x, const float k_y,
    const float m0, const float m1, const float m2, const float m3,
    const float ks_x, const float ks_y, const float ks_z, const float ks_w,
    const float ka_x, const float ka_y,
    const float ma0, const float ma1, const float ma2, const float ma3,
    const float mb0, const float mb1)
{
  (void)_unused;
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;

  const int kwidth = 3;

  float px = roi_out_x + (float)x + 0.5f;
  float py = roi_out_y + (float)y + 0.5f;
  px -= flip ? t_y * scale_out : t_x * scale_out;
  py -= flip ? t_x * scale_out : t_y * scale_out;
  px /= scale_out;
  py /= scale_out;
  vk_clip_backtransform(&px, &py, m0, m1, m2, m3, k_x, k_y);
  px *= scale_in;
  py *= scale_in;
  px += t_x * scale_in;
  py += t_y * scale_in;
  if(ks_z > 0.0f)
    vk_clip_keystone_backtransform(&px, &py, ks_x, ks_y, ka_x, ka_y,
                                   ma0, ma1, ma2, ma3, mb0, mb1);
  px -= (float)roi_in_x + 0.5f;
  py -= (float)roi_in_y + 0.5f;

  const int tx = (int)px;
  const int ty = (int)py;

  float4 pixel = (float4)(0.0f);
  float weight = 0.0f;
  for(int jj = 1 - kwidth; jj <= kwidth; jj++)
    for(int ii = 1 - kwidth; ii <= kwidth; ii++)
    {
      const int i = tx + ii;
      const int j = ty + jj;
      const float wx = vk_interpolation_lanczos(3.0f, (float)i - px);
      const float wy = vk_interpolation_lanczos(3.0f, (float)j - py);
      const float w = wx * wy;
      const int mi = vk_clip_mirror(i, in_width  - 1);
      const int mj = vk_clip_mirror(j, in_height - 1);
      pixel += in[mj * in_width + mi] * (float4)(w);
      weight += w;
    }
  pixel = (tx >= 0 && ty >= 0 && tx < in_width && ty < in_height)
          ? pixel / weight : (float4)(0.0f);
  out[y * width + x] = pixel;
}
