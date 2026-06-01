/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan equivalent of locallaplaciancl.{c,h}: local Laplacian
    pyramid for HDR-style local tone mapping on the Vulkan compute
    backend. See dev-doc/gpu_acceleration_clspv_vulkan.md §5.17.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License.
*/
#pragma once

#include "common/vulkan.h"

#ifdef HAVE_VULKAN

#include <stddef.h>

#define DT_LL_VK_MAX_LEVELS 30
#define DT_LL_VK_NUM_GAMMA   6

typedef struct dt_local_laplacian_vk_t
{
  int devid;

  int width, height;
  int num_levels;
  float sigma, highlights, shadows, clarity;
  int max_supp;
  int bwidth, bheight;
  int lwidth[DT_LL_VK_MAX_LEVELS];
  int lheight[DT_LL_VK_MAX_LEVELS];

  // pyramid of padded monochrome input buffer
  dt_vk_mem_t *dev_padded[DT_LL_VK_MAX_LEVELS];
  // pyramid of padded output buffer, monochrome, too
  dt_vk_mem_t *dev_output[DT_LL_VK_MAX_LEVELS];
  // one pyramid per gamma value
  dt_vk_mem_t *dev_processed[DT_LL_VK_NUM_GAMMA][DT_LL_VK_MAX_LEVELS];
} dt_local_laplacian_vk_t;

dt_local_laplacian_vk_t *dt_local_laplacian_init_vk(int devid,
                                                    int width, int height,
                                                    float sigma,
                                                    float shadows,
                                                    float highlights,
                                                    float clarity);

void dt_local_laplacian_free_vk(dt_local_laplacian_vk_t *g);

/** Apply the local-laplacian filter: `input` and `output` are float4
 *  Lab/RGB buffers sized width × height. Only the L channel is
 *  processed; chroma + alpha are copied. Returns 0 on success or -1
 *  on dispatch failure. */
int dt_local_laplacian_vk(dt_local_laplacian_vk_t *g,
                          dt_vk_mem_t *input, dt_vk_mem_t *output);

#else

typedef struct dt_local_laplacian_vk_t dt_local_laplacian_vk_t;
static inline dt_local_laplacian_vk_t *dt_local_laplacian_init_vk(
  int devid, int w, int h, float s, float sh, float hi, float c)
{ (void)devid; (void)w; (void)h; (void)s; (void)sh; (void)hi; (void)c; return NULL; }
static inline void dt_local_laplacian_free_vk(dt_local_laplacian_vk_t *g) { (void)g; }
static inline int dt_local_laplacian_vk(dt_local_laplacian_vk_t *g,
                                         dt_vk_mem_t *i, dt_vk_mem_t *o)
{ (void)g; (void)i; (void)o; return -1; }

#endif
