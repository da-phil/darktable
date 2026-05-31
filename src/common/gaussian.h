/*
    This file is part of darktable,
    Copyright (C) 2012-2020 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "common/opencl.h"
#include <assert.h>
#include <math.h>

typedef enum dt_gaussian_order_t
{
  DT_IOP_GAUSSIAN_ZERO = 0, // $DESCRIPTION: "order 0"
  DT_IOP_GAUSSIAN_ONE = 1,  // $DESCRIPTION: "order 1"
  DT_IOP_GAUSSIAN_TWO = 2   // $DESCRIPTION: "order 2"
} dt_gaussian_order_t;


typedef struct dt_gaussian_t
{
  int width, height, channels;
  float sigma;
  int order;
  float *max;
  float *min;
  float *buf;
} dt_gaussian_t;

dt_gaussian_t *dt_gaussian_init(const int width, const int height, const int channels, const float *max,
                                const float *min, const float sigma, const int order);

size_t dt_gaussian_memory_use(const int width, const int height, const int channels);
#ifdef HAVE_OPENCL
size_t dt_gaussian_memory_use_cl(const int width, const int height, const int channels);
#endif

size_t dt_gaussian_singlebuffer_size(const int width, const int height, const int channels);

void dt_gaussian_blur(dt_gaussian_t *g, const float *const in, float *const out);

void dt_gaussian_blur_4c(dt_gaussian_t *g, const float *const in, float *const out);

void dt_gaussian_free(dt_gaussian_t *g);
void dt_gaussian_fast_blur(float *in, float *out, const int width, const int height, const float sigma, const float min, const float max, const int channels);

// Convenience in-place Gaussian blur for IOP processing buffers.
// Uses unbounded signal range; works for arbitrary channel counts (1, 2, or 4).
static inline void dt_gaussian_mean_blur(float *const buf,
                                         const int width,
                                         const int height,
                                         const int ch,
                                         const float sigma)
{
  const float range = 1.0e9f;
  const dt_aligned_pixel_t max = { range, range, range, range };
  const dt_aligned_pixel_t min = { -range, -range, -range, -range };
  dt_gaussian_t *g = dt_gaussian_init(width, height, ch, max, min, sigma, DT_IOP_GAUSSIAN_ZERO);
  if(!g) return;
  if(ch == 4)
    dt_gaussian_blur_4c(g, buf, buf);
  else
    dt_gaussian_blur(g, buf, buf);
  dt_gaussian_free(g);
}

#ifdef HAVE_OPENCL
typedef struct dt_gaussian_cl_global_t
{
  int kernel_gaussian_column_4c, kernel_gaussian_transpose_4c;
  int kernel_gaussian_column_2c, kernel_gaussian_transpose_2c;
  int kernel_gaussian_column_1c, kernel_gaussian_transpose_1c;
  int kernel_gaussian_9x9;
} dt_gaussian_cl_global_t;


typedef struct dt_gaussian_cl_t
{
  dt_gaussian_cl_global_t *global;
  int devid;
  int width, height, channels;
  int blocksize;
  size_t bwidth, bheight;
  float sigma;
  int order;
  float *min;
  float *max;
  cl_mem dev_temp1;
  cl_mem dev_temp2;
} dt_gaussian_cl_t;

dt_gaussian_cl_global_t *dt_gaussian_init_cl_global(void);

void dt_gaussian_free_cl_global(dt_gaussian_cl_global_t *g);

dt_gaussian_cl_t *dt_gaussian_init_cl(const int devid, const int width, const int height, const int channels,
                                      const float *max, const float *min, const float sigma, const int order);

cl_int dt_gaussian_blur_cl(dt_gaussian_cl_t *g, cl_mem dev_in, cl_mem dev_out);
cl_int dt_gaussian_blur_cl_buffer(dt_gaussian_cl_t *g, cl_mem dev_in, cl_mem dev_out);
cl_int dt_gaussian_fast_blur_cl_buffer(const int devid, cl_mem dev_in, cl_mem dev_out, const int width, const int height, const float sigma, const int ch, const float min, const float max);

void dt_gaussian_free_cl(dt_gaussian_cl_t *g);

// OpenCL counterpart of dt_gaussian_mean_blur for GPU buffers.
static inline int dt_gaussian_mean_blur_cl(const int devid,
                                           cl_mem buf,
                                           const int width,
                                           const int height,
                                           const int ch,
                                           const float sigma)
{
  const float range = 1.0e9f;
  const dt_aligned_pixel_t max = { range, range, range, range };
  const dt_aligned_pixel_t min = { -range, -range, -range, -range };
  dt_gaussian_cl_t *g = dt_gaussian_init_cl(devid, width, height, ch, max, min, sigma,
                                            DT_IOP_GAUSSIAN_ZERO);
  if(!g) return DT_OPENCL_PROCESS_CL;
  const cl_int err = dt_gaussian_blur_cl_buffer(g, buf, buf);
  dt_gaussian_free_cl(g);
  return err;
}
#endif

// Vulkan recursive Gaussian. Mirrors the OpenCL surface above but
// operates on flat storage buffers and runs row-then-column instead
// of column+transpose (the transpose would need workgroup-local
// memory plumbing we don't have yet, and the IIR cost dominates the
// per-pixel walk regardless of dispatch shape).
//
// Three channel paths are wired up: 1-channel (float), 2-channel
// (float2) and 4-channel (float4). The channel count is passed at
// init time and selects the matching kernel pair internally.
#include "common/vulkan.h"

typedef struct dt_gaussian_vk_t
{
  int width, height;
  int channels;          // 1, 2 or 4
  float sigma;
  int order;
  float max[4], min[4];
  dt_vk_mem_t *dev_temp1;
  dt_vk_mem_t *dev_temp2;
} dt_gaussian_vk_t;

#ifdef HAVE_VULKAN
/** Allocate a Gaussian context for a width × height buffer of
 *  `channels` floats per pixel (1, 2 or 4). `max` / `min` are the
 *  per-channel clamp envelopes (read first `channels` entries).
 *  Returns NULL if Vulkan isn't running, the kernel set isn't loaded,
 *  or scratch allocation fails. */
dt_gaussian_vk_t *dt_gaussian_init_vk(int width, int height, int channels,
                                      const float *max, const float *min,
                                      float sigma, int order);

/** Blur dev_in (binding 0, `channels`-floats-per-pixel, sized
 *  width*height) into dev_out using the two internal scratch buffers.
 *  dev_in and dev_out may alias. Returns 0 on success or -1 if
 *  Vulkan isn't running / the kernels aren't loaded / dispatch failed. */
int dt_gaussian_blur_vk(dt_gaussian_vk_t *g,
                        dt_vk_mem_t *dev_in,
                        dt_vk_mem_t *dev_out);

void dt_gaussian_free_vk(dt_gaussian_vk_t *g);

/** One-shot in-place blur convenience matching dt_gaussian_mean_blur_cl.
 *  Allocates a context, runs the two-axis blur with dev_buf as both
 *  source and destination, and frees the context. Returns 0 on success
 *  or -1 on any failure. */
static inline int dt_gaussian_mean_blur_vk(int devid,
                                           dt_vk_mem_t *dev_buf,
                                           int width, int height, int channels,
                                           float sigma)
{
  (void)devid;
  const float range = 1.0e9f;
  const dt_aligned_pixel_t mx = { range, range, range, range };
  const dt_aligned_pixel_t mn = { -range, -range, -range, -range };
  dt_gaussian_vk_t *g = dt_gaussian_init_vk(width, height, channels, mx, mn, sigma,
                                            DT_IOP_GAUSSIAN_ZERO);
  if(!g) return -1;
  const int err = dt_gaussian_blur_vk(g, dev_buf, dev_buf);
  dt_gaussian_free_vk(g);
  return err;
}
#else
static inline dt_gaussian_vk_t *dt_gaussian_init_vk(int w, int h, int ch, const float *mx,
                                                    const float *mn, float s, int o)
{ (void)w; (void)h; (void)ch; (void)mx; (void)mn; (void)s; (void)o; return NULL; }
static inline int dt_gaussian_blur_vk(dt_gaussian_vk_t *g, dt_vk_mem_t *i, dt_vk_mem_t *o)
{ (void)g; (void)i; (void)o; return -1; }
static inline void dt_gaussian_free_vk(dt_gaussian_vk_t *g) { (void)g; }
static inline int dt_gaussian_mean_blur_vk(int devid, dt_vk_mem_t *b, int w, int h, int ch, float s)
{ (void)devid; (void)b; (void)w; (void)h; (void)ch; (void)s; return -1; }
#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

