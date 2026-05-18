/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan equivalent of bilateralcl.{c,h}: 3-D bilateral grid
    splat / blur / slice on the Vulkan compute backend. See
    dev-doc/gpu_acceleration_clspv_vulkan.md §5.13 for the design.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License.
*/

#pragma once

#include "common/vulkan.h"

#ifdef HAVE_VULKAN

#include <stddef.h>

typedef struct dt_bilateral_vk_t
{
  int width, height;
  int size_x, size_y, size_z;
  float sigma_s, sigma_r;
  dt_vk_mem_t *dev_grid;
  dt_vk_mem_t *dev_grid_tmp;
} dt_bilateral_vk_t;

/** Allocate and zero a 3-D bilateral grid sized to (width, height,
 *  sigma_s, sigma_r). Returns NULL if Vulkan isn't running, the
 *  bilateral kernels failed to load, or allocation failed. */
dt_bilateral_vk_t *dt_bilateral_init_vk(int width, int height,
                                        float sigma_s, float sigma_r);

/** Splat `dev_in` (float4 Lab pixels) into the grid via 8-cell
 *  trilinear distribution. Returns 0 on success, -1 on dispatch
 *  failure. */
int dt_bilateral_splat_vk(dt_bilateral_vk_t *b, dt_vk_mem_t *dev_in);

/** Run the three separable 5-tap blurs (X, Y, Z) over the grid.
 *  Returns 0 on success. */
int dt_bilateral_blur_vk(dt_bilateral_vk_t *b);

/** Sample the grid at each pixel's (x, y, L_in) and write the
 *  result to `dev_out` (.x = L_in + norm*Ldiff, rest copied from
 *  dev_in). Returns 0 on success. */
int dt_bilateral_slice_vk(dt_bilateral_vk_t *b,
                          dt_vk_mem_t *dev_in, dt_vk_mem_t *dev_out,
                          float detail);

/** Same as dt_bilateral_slice_vk but adds the grid delta to a
 *  separately-bound `dev_target` buffer's L channel (rather than to
 *  `dev_in`'s L). Used by retouch / shadhi-style consumers that
 *  produce the L correction relative to a different base. */
int dt_bilateral_slice_to_output_vk(dt_bilateral_vk_t *b,
                                    dt_vk_mem_t *dev_in,
                                    dt_vk_mem_t *dev_target,
                                    dt_vk_mem_t *dev_out,
                                    float detail);

/** Free the grid buffers (safe on NULL). */
void dt_bilateral_free_vk(dt_bilateral_vk_t *b);

#else // HAVE_VULKAN

typedef struct dt_bilateral_vk_t dt_bilateral_vk_t;
static inline dt_bilateral_vk_t *dt_bilateral_init_vk(int w, int h, float s, float r)
{ (void)w; (void)h; (void)s; (void)r; return NULL; }
static inline int dt_bilateral_splat_vk(dt_bilateral_vk_t *b, dt_vk_mem_t *i)
{ (void)b; (void)i; return -1; }
static inline int dt_bilateral_blur_vk(dt_bilateral_vk_t *b) { (void)b; return -1; }
static inline int dt_bilateral_slice_vk(dt_bilateral_vk_t *b, dt_vk_mem_t *i,
                                        dt_vk_mem_t *o, float d)
{ (void)b; (void)i; (void)o; (void)d; return -1; }
static inline int dt_bilateral_slice_to_output_vk(dt_bilateral_vk_t *b, dt_vk_mem_t *i,
                                                  dt_vk_mem_t *t, dt_vk_mem_t *o, float d)
{ (void)b; (void)i; (void)t; (void)o; (void)d; return -1; }
static inline void dt_bilateral_free_vk(dt_bilateral_vk_t *b) { (void)b; }

#endif // HAVE_VULKAN
