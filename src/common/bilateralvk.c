/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan equivalent of bilateralcl.c — 3-D bilateral grid splat /
    blur / slice on the Vulkan compute backend. See
    dev-doc/gpu_acceleration_clspv_vulkan.md §5.13 for the design.
*/

#ifdef HAVE_VULKAN

#include "common/bilateral.h"
#include "common/bilateralvk.h"
#include "common/darktable.h"

#include <stdlib.h>
#include <string.h>

// Kernel slots loaded lazily on first dt_bilateral_init_vk call,
// same shape as the dt_gaussian_*_vk / colorspaces helper modules.
// One slot per .spv file — separating each entry into its own .spv
// keeps the glslang-fallback path working (glslang's -e flag emits
// a single entry per module; clspv emits all together but the host
// loader doesn't care about the bundling).
static dt_vk_module_kernel_t _vk_bilat_zero          = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_bilat_splat         = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_bilat_blur          = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_bilat_blur_z        = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_bilat_slice         = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_bilat_slice_out     = DT_VK_MODULE_KERNEL_INIT;
static gboolean              _vk_bilat_loaded        = FALSE;

// Push-constant structs — match the .cl/.comp twins byte-for-byte.
typedef struct
{
  int width;
  int height;
} pc_zero_t;

typedef struct
{
  int   width;
  int   height;
  int   size_x;
  int   size_y;
  int   size_z;
  float sigma_s;
  float sigma_r;
} pc_splat_t;

typedef struct
{
  int offset1;
  int offset2;
  int offset3;
  int size1;
  int size2;
  int size3;
} pc_blur_t;

typedef struct
{
  int   width;
  int   height;
  int   size_x;
  int   size_y;
  int   size_z;
  float sigma_s;
  float sigma_r;
  float detail;
} pc_slice_t;

static void _vk_bilat_ensure_kernels(void)
{
  if(_vk_bilat_loaded) return;
  if(!dt_vulkan_running()) return;
  dt_vulkan_module_kernel_load(&_vk_bilat_zero,      "bilateral_zero",
                               "bilateral_zero",      1, sizeof(pc_zero_t),  16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_bilat_splat,     "bilateral_splat",
                               "bilateral_splat",     2, sizeof(pc_splat_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_bilat_blur,      "bilateral_blur_line",
                               "bilateral_blur_line", 2, sizeof(pc_blur_t),  16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_bilat_blur_z,    "bilateral_blur_line_z",
                               "bilateral_blur_line_z", 2, sizeof(pc_blur_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_bilat_slice,     "bilateral_slice",
                               "bilateral_slice",     3, sizeof(pc_slice_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_bilat_slice_out, "bilateral_slice_to_output",
                               "bilateral_slice_to_output", 4, sizeof(pc_slice_t), 16, 16, 1);
  _vk_bilat_loaded = TRUE;
}

static gboolean _all_kernels_ready(void)
{
  return _vk_bilat_zero.kernel      >= 0
      && _vk_bilat_splat.kernel     >= 0
      && _vk_bilat_blur.kernel      >= 0
      && _vk_bilat_blur_z.kernel    >= 0
      && _vk_bilat_slice.kernel     >= 0
      && _vk_bilat_slice_out.kernel >= 0;
}

dt_bilateral_vk_t *dt_bilateral_init_vk(int width, int height,
                                        float sigma_s, float sigma_r)
{
  if(!dt_vulkan_running()) return NULL;
  _vk_bilat_ensure_kernels();
  if(!_all_kernels_ready()) return NULL;

  dt_bilateral_vk_t *b = calloc(1, sizeof(*b));
  if(!b) return NULL;
  b->width = width;
  b->height = height;

  // Reuse the CL helper for grid sizing — same maths, same clamps,
  // same L_range=100.0f convention so VK and CL produce bit-equal
  // grids for matching sigma inputs.
  dt_bilateral_t b2;
  dt_bilateral_grid_size(&b2, width, height, 100.0f, sigma_s, sigma_r);
  b->size_x  = b2.size_x;
  b->size_y  = b2.size_y;
  b->size_z  = b2.size_z;
  b->sigma_s = b2.sigma_s;
  b->sigma_r = b2.sigma_r;

  const size_t grid_bytes = sizeof(float) * b->size_x * b->size_y * b->size_z;
  b->dev_grid     = dt_vulkan_alloc_buffer(0, grid_bytes);
  b->dev_grid_tmp = dt_vulkan_alloc_buffer(0, grid_bytes);
  if(!b->dev_grid || !b->dev_grid_tmp)
  {
    dt_bilateral_free_vk(b);
    return NULL;
  }

  // Zero the grid via the zero kernel — dispatched as a 2D grid
  // sized (size_x, size_y * size_z), one work-item per cell.
  const pc_zero_t pc = { .width = b->size_x, .height = b->size_y * b->size_z };
  dt_vk_mem_t *bufs[] = { b->dev_grid };
  if(dt_vulkan_dispatch_n(&_vk_bilat_zero, bufs, 1,
                          pc.width, pc.height, &pc, sizeof(pc)) != 0)
  {
    dt_bilateral_free_vk(b);
    return NULL;
  }
  return b;
}

int dt_bilateral_splat_vk(dt_bilateral_vk_t *b, dt_vk_mem_t *dev_in)
{
  if(!b || !dev_in) return -1;
  const pc_splat_t pc = {
    .width = b->width, .height = b->height,
    .size_x = b->size_x, .size_y = b->size_y, .size_z = b->size_z,
    .sigma_s = b->sigma_s, .sigma_r = b->sigma_r,
  };
  dt_vk_mem_t *bufs[] = { dev_in, b->dev_grid };
  return dt_vulkan_dispatch_n(&_vk_bilat_splat, bufs, 2,
                              b->width, b->height, &pc, sizeof(pc));
}

int dt_bilateral_blur_vk(dt_bilateral_vk_t *b)
{
  if(!b) return -1;
  const size_t grid_bytes = sizeof(float) * b->size_x * b->size_y * b->size_z;

  // Copy dev_grid -> dev_grid_tmp so the first X-blur reads from
  // dev_grid_tmp and writes back into dev_grid (mirroring the CL
  // helper's ping-pong sequence).
  if(dt_vulkan_copy_device_to_device(0, b->dev_grid_tmp, b->dev_grid, grid_bytes) != 0)
    return -1;

  // X-blur: dev_grid_tmp -> dev_grid
  {
    const pc_blur_t pc = {
      .offset1 = b->size_x * b->size_y,
      .offset2 = b->size_x,
      .offset3 = 1,
      .size1 = b->size_z, .size2 = b->size_y, .size3 = b->size_x,
    };
    dt_vk_mem_t *bufs[] = { b->dev_grid_tmp, b->dev_grid };
    if(dt_vulkan_dispatch_n(&_vk_bilat_blur, bufs, 2,
                            pc.size1, pc.size2, &pc, sizeof(pc)) != 0)
      return -1;
  }
  // Y-blur: dev_grid -> dev_grid_tmp
  {
    const pc_blur_t pc = {
      .offset1 = b->size_x * b->size_y,
      .offset2 = 1,
      .offset3 = b->size_x,
      .size1 = b->size_z, .size2 = b->size_x, .size3 = b->size_y,
    };
    dt_vk_mem_t *bufs[] = { b->dev_grid, b->dev_grid_tmp };
    if(dt_vulkan_dispatch_n(&_vk_bilat_blur, bufs, 2,
                            pc.size1, pc.size2, &pc, sizeof(pc)) != 0)
      return -1;
  }
  // Z-blur (derivative shape): dev_grid_tmp -> dev_grid
  {
    const pc_blur_t pc = {
      .offset1 = 1,
      .offset2 = b->size_x,
      .offset3 = b->size_x * b->size_y,
      .size1 = b->size_x, .size2 = b->size_y, .size3 = b->size_z,
    };
    dt_vk_mem_t *bufs[] = { b->dev_grid_tmp, b->dev_grid };
    if(dt_vulkan_dispatch_n(&_vk_bilat_blur_z, bufs, 2,
                            pc.size1, pc.size2, &pc, sizeof(pc)) != 0)
      return -1;
  }
  return 0;
}

int dt_bilateral_slice_vk(dt_bilateral_vk_t *b,
                          dt_vk_mem_t *dev_in, dt_vk_mem_t *dev_out,
                          float detail)
{
  if(!b || !dev_in || !dev_out) return -1;
  const pc_slice_t pc = {
    .width = b->width, .height = b->height,
    .size_x = b->size_x, .size_y = b->size_y, .size_z = b->size_z,
    .sigma_s = b->sigma_s, .sigma_r = b->sigma_r, .detail = detail,
  };
  dt_vk_mem_t *bufs[] = { dev_in, dev_out, b->dev_grid };
  return dt_vulkan_dispatch_n(&_vk_bilat_slice, bufs, 3,
                              b->width, b->height, &pc, sizeof(pc));
}

int dt_bilateral_slice_to_output_vk(dt_bilateral_vk_t *b,
                                    dt_vk_mem_t *dev_in,
                                    dt_vk_mem_t *dev_target,
                                    dt_vk_mem_t *dev_out,
                                    float detail)
{
  if(!b || !dev_in || !dev_target || !dev_out) return -1;
  const pc_slice_t pc = {
    .width = b->width, .height = b->height,
    .size_x = b->size_x, .size_y = b->size_y, .size_z = b->size_z,
    .sigma_s = b->sigma_s, .sigma_r = b->sigma_r, .detail = detail,
  };
  dt_vk_mem_t *bufs[] = { dev_in, dev_target, dev_out, b->dev_grid };
  return dt_vulkan_dispatch_n(&_vk_bilat_slice_out, bufs, 4,
                              b->width, b->height, &pc, sizeof(pc));
}

void dt_bilateral_free_vk(dt_bilateral_vk_t *b)
{
  if(!b) return;
  if(b->dev_grid)     dt_vulkan_free_buffer(0, b->dev_grid);
  if(b->dev_grid_tmp) dt_vulkan_free_buffer(0, b->dev_grid_tmp);
  free(b);
}

#endif // HAVE_VULKAN

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
