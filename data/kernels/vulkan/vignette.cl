/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan port of extended.cl::vignette — radial darkening/brightening
    with optional dithering and saturation adjustment.

    The OpenCL build uses a small inline TEA cipher + triangular-PDF
    helper for dither noise; we inline both directly into this file
    since they're only used by this one kernel.

    Bindings:
      0 = input  float4 buffer
      1 = output float4 buffer
    Push constants: width, height, scale[2], roi_center_scaled[2],
                    expt[2], dscale, fscale, brightness, saturation,
                    dither, unbound. 14 scalars = 56 bytes.
*/

#include "dt_vulkan_common.h"

#define VK_TEA_ROUNDS 8

static inline void vk_encrypt_tea(unsigned int *v0p, unsigned int *v1p)
{
  const unsigned int key0 = 0xa341316c;
  const unsigned int key1 = 0xc8013ea4;
  const unsigned int key2 = 0xad90777d;
  const unsigned int key3 = 0x7e95761e;
  const unsigned int delta = 0x9e3779b9;
  unsigned int v0 = *v0p, v1 = *v1p;
  unsigned int sum = 0;
  for(int i = 0; i < VK_TEA_ROUNDS; i++)
  {
    sum += delta;
    v0 += ((v1 << 4) + key0) ^ (v1 + sum) ^ ((v1 >> 5) + key1);
    v1 += ((v0 << 4) + key2) ^ (v0 + sum) ^ ((v0 >> 5) + key3);
  }
  *v0p = v0;
  *v1p = v1;
}

static inline float vk_tpdf(unsigned int urandom)
{
  const float frandom = (float)urandom / (float)0xFFFFFFFFu;
  return (frandom < 0.5f) ? (sqrt(2.0f * frandom) - 1.0f)
                          : (1.0f - sqrt(2.0f * (1.0f - frandom)));
}

kernel void vignette(global const float4 *in,
                     global float4 *out,
                     const int width,
                     const int height,
                     const float scale_x,
                     const float scale_y,
                     const float roi_center_x,
                     const float roi_center_y,
                     const float expt_x,
                     const float expt_y,
                     const float dscale,
                     const float fscale,
                     const float brightness,
                     const float saturation,
                     const float dither,
                     const int unbound)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  if(x >= width || y >= height) return;
  const int i = idx2d(x, y, width);

  unsigned int t0 = (unsigned int)(y * width + x);
  unsigned int t1 = 0;
  vk_encrypt_tea(&t0, &t1);

  const float pvx = fabs((float)x * scale_x - roi_center_x);
  const float pvy = fabs((float)y * scale_y - roi_center_y);
  const float cplen = pow(pow(pvx, expt_x) + pow(pvy, expt_x), expt_y);

  float weight = 0.0f;
  float dith = 0.0f;
  if(cplen >= dscale)
  {
    weight = (cplen - dscale) / fscale;
    dith = (weight <= 1.0f && weight >= 0.0f) ? dither * vk_tpdf(t0) : 0.0f;
    weight = (weight >= 1.0f) ? 1.0f
           : ((weight <= 0.0f) ? 0.0f
                               : 0.5f - cos(M_PI_F * weight) / 2.0f);
  }

  float4 pixel = in[i];
  if(weight > 0.0f)
  {
    const float falloff = (brightness < 0.0f)
                            ? 1.0f + (weight * brightness)
                            : weight * brightness;
    if(brightness < 0.0f)
    {
      pixel.x = pixel.x * falloff + dith;
      pixel.y = pixel.y * falloff + dith;
      pixel.z = pixel.z * falloff + dith;
    }
    else
    {
      pixel.x = pixel.x + falloff + dith;
      pixel.y = pixel.y + falloff + dith;
      pixel.z = pixel.z + falloff + dith;
    }
    if(!unbound)
    {
      pixel.x = clamp(pixel.x, 0.0f, 1.0f);
      pixel.y = clamp(pixel.y, 0.0f, 1.0f);
      pixel.z = clamp(pixel.z, 0.0f, 1.0f);
    }
    const float mv = (pixel.x + pixel.y + pixel.z) / 3.0f;
    const float wss = weight * saturation;
    pixel.x = pixel.x - (mv - pixel.x) * wss;
    pixel.y = pixel.y - (mv - pixel.y) * wss;
    pixel.z = pixel.z - (mv - pixel.z) * wss;
    if(!unbound)
    {
      pixel.x = clamp(pixel.x, 0.0f, 1.0f);
      pixel.y = clamp(pixel.y, 0.0f, 1.0f);
      pixel.z = clamp(pixel.z, 0.0f, 1.0f);
    }
  }
  out[i] = pixel;
}
