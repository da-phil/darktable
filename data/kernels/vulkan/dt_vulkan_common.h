/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Shared helpers for Vulkan-targeted IOP kernels.

    Convention: every kernel here operates on flat RGBA float storage
    buffers (the pixelpipe stages cl_mem / float* inputs through
    dt_vulkan_write_to_device before dispatch; see
    src/develop/pixelpipe_hb.c). The standard signature is:

        kernel void NAME(global const float4 *in,
                         global float4 *out,
                         const int width,
                         const int height,
                         ... scalar params ...);

    Push constants carry width / height plus the module-specific
    scalars. Storage buffers carry input and output (and constant
    tables when needed, on subsequent bindings).
*/

#pragma once

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

// Branchless clamp helpers.
static inline float clampf(const float x, const float lo, const float hi)
{
  return fmin(fmax(x, lo), hi);
}

// Match the clipf helper used by data/kernels/common.h so kernel
// ports read 1:1 against their OpenCL counterparts.
static inline float clipf(const float a)
{
  return clamp(a, 0.0f, 1.0f);
}

static inline float4 clampf4(const float4 v, const float lo, const float hi)
{
  return (float4)(clampf(v.x, lo, hi),
                  clampf(v.y, lo, hi),
                  clampf(v.z, lo, hi),
                  v.w);
}

// Linear-index helper.
static inline int idx2d(const int x, const int y, const int width)
{
  return y * width + x;
}

// Read with edge-clamped coords (saves writing the clamp+idx pair on
// every kernel that does spatial neighbourhoods).
static inline float4 read_clamped(global const float4 *buf,
                                  const int x, const int y,
                                  const int width, const int height)
{
  const int cx = clamp(x, 0, width  - 1);
  const int cy = clamp(y, 0, height - 1);
  return buf[cy * width + cx];
}

// Apply a 3x3 matrix to a float4 (alpha preserved).
static inline float4 matmul3(const float4 v, constant const float *m)
{
  // m is 9 floats laid out row-major: r0c0 r0c1 r0c2 / r1c0 ...
  float4 r;
  r.x = m[0] * v.x + m[1] * v.y + m[2] * v.z;
  r.y = m[3] * v.x + m[4] * v.y + m[5] * v.z;
  r.z = m[6] * v.x + m[7] * v.y + m[8] * v.z;
  r.w = v.w;
  return r;
}

// Same, but for a row-major 3x4 padded matrix (the format darktable
// stores ICC transform matrices in).
static inline float4 matmul3_padded(const float4 v, constant const float *m)
{
  float4 r;
  r.x = m[0]  * v.x + m[1]  * v.y + m[2]  * v.z;
  r.y = m[4]  * v.x + m[5]  * v.y + m[6]  * v.z;
  r.z = m[8]  * v.x + m[9]  * v.y + m[10] * v.z;
  r.w = v.w;
  return r;
}
