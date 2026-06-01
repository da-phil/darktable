/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    Vulkan equivalent of locallaplaciancl.c — local Laplacian pyramid
    on the Vulkan compute backend. See dev-doc/gpu_acceleration_clspv_vulkan.md
    §5.17 for the design.
*/

#ifdef HAVE_VULKAN

#include "common/darktable.h"
#include "common/locallaplacianvk.h"

#include <stdlib.h>
#include <string.h>

// Kernel slots loaded lazily on first dt_local_laplacian_init_vk —
// same shape as bilateralvk / dwt.
static dt_vk_module_kernel_t _vk_ll_pad    = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_ll_reduce = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_ll_curve  = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_ll_asm    = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t _vk_ll_back   = DT_VK_MODULE_KERNEL_INIT;
static gboolean              _vk_ll_loaded = FALSE;

typedef struct { int wd; int ht; int max_supp; int wd2; int ht2; } pc_pad_t;
typedef struct { int wd; int ht; int fine_w; int fine_h; }         pc_reduce_t;
typedef struct
{
  int   wd; int ht;
  float g; float sigma; float shadows; float highlights; float clarity;
} pc_curve_t;
typedef struct { int pw; int ph; int cw; int ch; }                 pc_asm_t;
typedef struct { int max_supp; int wd; int ht; int padded_w; }     pc_back_t;

static void _vk_ll_ensure_kernels(void)
{
  if(_vk_ll_loaded) return;
  if(!dt_vulkan_running()) return;
  dt_vulkan_module_kernel_load(&_vk_ll_pad,    "ll_pad_input",
                               "ll_pad_input",        2, sizeof(pc_pad_t),    16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_ll_reduce, "ll_gauss_reduce",
                               "ll_gauss_reduce",     2, sizeof(pc_reduce_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_ll_curve,  "ll_process_curve",
                               "ll_process_curve",    2, sizeof(pc_curve_t),  16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_ll_asm,    "ll_laplacian_assemble",
                               "ll_laplacian_assemble", 15, sizeof(pc_asm_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&_vk_ll_back,   "ll_write_back",
                               "ll_write_back",       3, sizeof(pc_back_t),   16, 16, 1);
  _vk_ll_loaded = TRUE;
}

static gboolean _vk_ll_ready(void)
{
  return _vk_ll_pad.kernel    >= 0
      && _vk_ll_reduce.kernel >= 0
      && _vk_ll_curve.kernel  >= 0
      && _vk_ll_asm.kernel    >= 0
      && _vk_ll_back.kernel   >= 0;
}

// downsample-by-2 with ceiling: matches the `dl` helper in locallaplaciancl.c
static inline int _dl(int size, int level)
{
  for(int l = 0; l < level; l++) size = (size - 1) / 2 + 1;
  return size;
}

dt_local_laplacian_vk_t *dt_local_laplacian_init_vk(int devid,
                                                    int width, int height,
                                                    float sigma,
                                                    float shadows,
                                                    float highlights,
                                                    float clarity)
{
  if(!dt_vulkan_running()) return NULL;
  _vk_ll_ensure_kernels();
  if(!_vk_ll_ready()) return NULL;

  dt_local_laplacian_vk_t *g = calloc(1, sizeof(*g));
  if(!g) return NULL;
  g->devid = devid;
  g->width = width;
  g->height = height;
  g->sigma = sigma;
  g->shadows = shadows;
  g->highlights = highlights;
  g->clarity = clarity;

  // Number of pyramid levels: log2 of the smaller dimension, clamped.
  int s = (width < height) ? width : height;
  int nl = 0;
  while(s > 1 && nl < DT_LL_VK_MAX_LEVELS) { s >>= 1; nl++; }
  g->num_levels = nl;
  g->max_supp = 1 << (g->num_levels - 1);
  g->bwidth  = width  + 2 * g->max_supp;
  g->bheight = height + 2 * g->max_supp;

  for(int l = 0; l < g->num_levels; l++)
  {
    g->lwidth[l]  = _dl(g->bwidth, l);
    g->lheight[l] = _dl(g->bheight, l);
    const size_t bytes = (size_t)g->lwidth[l] * g->lheight[l] * sizeof(float);
    g->dev_padded[l] = dt_vulkan_alloc_buffer(devid, bytes);
    g->dev_output[l] = dt_vulkan_alloc_buffer(devid, bytes);
    if(!g->dev_padded[l] || !g->dev_output[l]) goto fail;
    for(int k = 0; k < DT_LL_VK_NUM_GAMMA; k++)
    {
      g->dev_processed[k][l] = dt_vulkan_alloc_buffer(devid, bytes);
      if(!g->dev_processed[k][l]) goto fail;
    }
  }
  return g;

fail:
  dt_local_laplacian_free_vk(g);
  return NULL;
}

void dt_local_laplacian_free_vk(dt_local_laplacian_vk_t *g)
{
  if(!g) return;
  for(int l = 0; l < DT_LL_VK_MAX_LEVELS; l++)
  {
    if(g->dev_padded[l]) dt_vulkan_free_buffer(g->devid, g->dev_padded[l]);
    if(g->dev_output[l]) dt_vulkan_free_buffer(g->devid, g->dev_output[l]);
    for(int k = 0; k < DT_LL_VK_NUM_GAMMA; k++)
      if(g->dev_processed[k][l])
        dt_vulkan_free_buffer(g->devid, g->dev_processed[k][l]);
  }
  free(g);
}

int dt_local_laplacian_vk(dt_local_laplacian_vk_t *g,
                          dt_vk_mem_t *input, dt_vk_mem_t *output)
{
  if(!g || g->bwidth <= 1 || g->bheight <= 1) return -1;

  // pad_input: build the padded level-0 monochrome buffer.
  {
    const pc_pad_t pc = {
      .wd = g->width, .ht = g->height,
      .max_supp = g->max_supp,
      .wd2 = g->bwidth, .ht2 = g->bheight,
    };
    dt_vk_mem_t *bufs[] = { input, g->dev_padded[0] };
    if(dt_vulkan_dispatch_n(&_vk_ll_pad, bufs, 2,
                            g->bwidth, g->bheight, &pc, sizeof(pc)) != 0)
      return -1;
  }

  // gauss pyramid of padded input. The coarsest level writes
  // directly to dev_output[L-1] to match the OpenCL helper.
  for(int l = 1; l < g->num_levels; l++)
  {
    const int wd = g->lwidth[l], ht = g->lheight[l];
    const int fw = g->lwidth[l - 1], fh = g->lheight[l - 1];
    dt_vk_mem_t *dst = (l == g->num_levels - 1)
                         ? g->dev_output[l] : g->dev_padded[l];
    const pc_reduce_t pc = { .wd = wd, .ht = ht, .fine_w = fw, .fine_h = fh };
    dt_vk_mem_t *bufs[] = { g->dev_padded[l - 1], dst };
    if(dt_vulkan_dispatch_n(&_vk_ll_reduce, bufs, 2, wd, ht, &pc, sizeof(pc)) != 0)
      return -1;
  }

  // For each of the num_gamma curve values: run process_curve at
  // level 0, then build a gaussian pyramid in dev_processed[k].
  for(int k = 0; k < DT_LL_VK_NUM_GAMMA; k++)
  {
    const float gv = ((float)k + 0.5f) / (float)DT_LL_VK_NUM_GAMMA;
    {
      const pc_curve_t pc = {
        .wd = g->bwidth, .ht = g->bheight,
        .g = gv, .sigma = g->sigma,
        .shadows = g->shadows, .highlights = g->highlights, .clarity = g->clarity,
      };
      dt_vk_mem_t *bufs[] = { g->dev_padded[0], g->dev_processed[k][0] };
      if(dt_vulkan_dispatch_n(&_vk_ll_curve, bufs, 2,
                              g->bwidth, g->bheight, &pc, sizeof(pc)) != 0)
        return -1;
    }
    for(int l = 1; l < g->num_levels; l++)
    {
      const int wd = g->lwidth[l], ht = g->lheight[l];
      const int fw = g->lwidth[l - 1], fh = g->lheight[l - 1];
      const pc_reduce_t pc = { .wd = wd, .ht = ht, .fine_w = fw, .fine_h = fh };
      dt_vk_mem_t *bufs[] = { g->dev_processed[k][l - 1], g->dev_processed[k][l] };
      if(dt_vulkan_dispatch_n(&_vk_ll_reduce, bufs, 2, wd, ht, &pc, sizeof(pc)) != 0)
        return -1;
    }
  }

  // Assemble output pyramid coarse to fine.
  for(int l = g->num_levels - 2; l >= 0; l--)
  {
    const int pw = g->lwidth[l],     ph = g->lheight[l];
    const int cw = g->lwidth[l + 1], ch = g->lheight[l + 1];
    const pc_asm_t pc = { .pw = pw, .ph = ph, .cw = cw, .ch = ch };
    dt_vk_mem_t *bufs[] = {
      g->dev_padded[l], g->dev_output[l + 1], g->dev_output[l],
      g->dev_processed[0][l], g->dev_processed[0][l + 1],
      g->dev_processed[1][l], g->dev_processed[1][l + 1],
      g->dev_processed[2][l], g->dev_processed[2][l + 1],
      g->dev_processed[3][l], g->dev_processed[3][l + 1],
      g->dev_processed[4][l], g->dev_processed[4][l + 1],
      g->dev_processed[5][l], g->dev_processed[5][l + 1],
    };
    if(dt_vulkan_dispatch_n(&_vk_ll_asm, bufs, 15, pw, ph, &pc, sizeof(pc)) != 0)
      return -1;
  }

  // write_back: input + processed L → output (chroma passed through).
  {
    const pc_back_t pc = {
      .max_supp = g->max_supp, .wd = g->width, .ht = g->height,
      .padded_w = g->bwidth,
    };
    dt_vk_mem_t *bufs[] = { input, g->dev_output[0], output };
    if(dt_vulkan_dispatch_n(&_vk_ll_back, bufs, 3,
                            g->width, g->height, &pc, sizeof(pc)) != 0)
      return -1;
  }
  return 0;
}

#endif // HAVE_VULKAN
