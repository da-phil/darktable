// Vulkan port of basic.cl :: clip_rotate_bilinear.
//
// Crop + rotate + optional keystone backtransform with manual 4-tap
// bilinear sampling (same hardware-bilinear-to-4-tap rewrite as
// ashift_bilinear). The OpenCL kernel uses pixel-centre coords
// (po + 0.5) with `samplerf` (CLAMP-to-zero border); the Vulkan
// twin reads at floor(po) with the four-corner weighted blend, also
// zeroing each out-of-bounds corner.
//
// Coordinate flow (mirrors process_cl byte-for-byte):
//   pi    = (roi_out + x + 0.5, roi_out + y + 0.5)
//   pi   -= (flip ? t.yx : t.xy) * scale_out
//   pi   /= scale_out
//   po    = mat · (pi after keystone rotational backtransform via k)
//   po   *= scale_in
//   po   += t * scale_in
//   (if k_space.z > 0) po = keystone_backtransform(po, k_space, ka, ma, mb)
//   po   -= (roi_in + 0.5)
//
// 3 storage bindings (in, out, optional unused) + 124 B PC.

#include "dt_vulkan_common.h"

// Same body as basic.cl::backtransform.
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

// Same body as basic.cl::keystone_backtransform.
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

kernel void clip_rotate_bilinear(
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

  const int ii = (int)px;
  const int jj = (int)py;

  float4 pixel = (float4)(0.0f);
  if(ii >= 0 && jj >= 0 && ii < in_width && jj < in_height)
  {
    const int ix0 = (int)floor(px);
    const int iy0 = (int)floor(py);
    const float fx = px - (float)ix0;
    const float fy = py - (float)iy0;
    const int ix1 = ix0 + 1;
    const int iy1 = iy0 + 1;
    const float4 c00 = (ix0 >= 0 && ix0 < in_width && iy0 >= 0 && iy0 < in_height)
                       ? in[iy0 * in_width + ix0] : (float4)(0.0f);
    const float4 c10 = (ix1 >= 0 && ix1 < in_width && iy0 >= 0 && iy0 < in_height)
                       ? in[iy0 * in_width + ix1] : (float4)(0.0f);
    const float4 c01 = (ix0 >= 0 && ix0 < in_width && iy1 >= 0 && iy1 < in_height)
                       ? in[iy1 * in_width + ix0] : (float4)(0.0f);
    const float4 c11 = (ix1 >= 0 && ix1 < in_width && iy1 >= 0 && iy1 < in_height)
                       ? in[iy1 * in_width + ix1] : (float4)(0.0f);
    pixel = (1.0f - fx) * (1.0f - fy) * c00
          + (       fx) * (1.0f - fy) * c10
          + (1.0f - fx) * (       fy) * c01
          + (       fx) * (       fy) * c11;
  }
  out[y * width + x] = pixel;
}
