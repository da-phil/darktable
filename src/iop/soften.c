/*
    This file is part of darktable,
    Copyright (C) 2011-2026 darktable developers.

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

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bauhaus/bauhaus.h"
#include "common/box_filters.h"
#include "common/colorspaces.h"
#include "common/imagebuf.h"
#include "common/math.h"
#include "common/opencl.h"
#include "common/vulkan.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "develop/tiling.h"
#include "dtgtk/resetlabel.h"
#include "gui/accelerators.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"
#include <gtk/gtk.h>
#include <inttypes.h>

DT_MODULE_INTROSPECTION(1, dt_iop_soften_params_t)

typedef struct dt_iop_soften_params_t
{
  float size;       // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0
  float saturation; // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 100.0
  float brightness; // $MIN: -2.0 $MAX: 2.0 $DEFAULT: 0.33
  float amount;     // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0 $DESCRIPTION: "mix"
} dt_iop_soften_params_t;

typedef struct dt_iop_soften_gui_data_t
{
  GtkWidget *size, *saturation, *brightness, *amount;
} dt_iop_soften_gui_data_t;

typedef struct dt_iop_soften_data_t
{
  float size;
  float saturation;
  float brightness;
  float amount;
} dt_iop_soften_data_t;

typedef struct dt_iop_soften_global_data_t
{
  int kernel_soften_overexposed;
  int kernel_soften_hblur;
  int kernel_soften_vblur;
  int kernel_soften_mix;
  dt_vk_module_kernel_t vk_overexposed;
  dt_vk_module_kernel_t vk_hblur;
  dt_vk_module_kernel_t vk_vblur;
  dt_vk_module_kernel_t vk_mix;
} dt_iop_soften_global_data_t;


const char *name()
{
  return _("soften");
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING | IOP_FLAGS_ALLOW_TILING;
}

int default_group()
{
  return IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("create a softened image using the Orton effect"),
                                      _("creative"),
                                      _("linear, RGB, display-referred"),
                                      _("linear, RGB"),
                                      _("linear, RGB, display-referred"));
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const dt_iop_soften_data_t *const d = (const dt_iop_soften_data_t *const)piece->data;

  if(!dt_iop_have_required_input_format(4 /*we need full-color pixels*/,
                                        self, piece->colors,
                                         ivoid, ovoid, roi_in, roi_out))
    return; // image has been copied through to output and module's
            // trouble flag has been updated

  const float brightness = 1.0 / exp2f(-d->brightness);
  const float saturation = d->saturation / 100.0;

  const float *const restrict in = (const float *const)ivoid;
  float *const restrict out = (float *const)ovoid;

  const size_t npixels = (size_t)roi_out->width * roi_out->height;
  /* create overexpose image and then blur */
  DT_OMP_FOR()
  for(size_t k = 0; k < 4 * npixels; k += 4)
  {
    float h, s, l;
    rgb2hsl(&in[k], &h, &s, &l);
    s *= saturation;
    l *= brightness;
    hsl2rgb(&out[k], h, CLIP(s), CLIP(l));
  }

  const float w = piece->iwidth * piece->iscale;
  const float h = piece->iheight * piece->iscale;
  const int mrad = dt_fast_hypotf(w, h) * 0.01f;
  const int rad = mrad * (fmin(100.0, d->size + 1.0f) / 100.0);
  const int radius = MIN(mrad, ceilf(rad * roi_in->scale / piece->iscale));

  dt_box_mean(out, roi_out->height, roi_out->width, 4, radius, BOX_ITERATIONS);

  const float amt = d->amount / 100.0f;
  dt_iop_image_linear_blend(out, amt, in, roi_out->width, roi_out->height, 4);
}


#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  dt_iop_soften_data_t *d = piece->data;
  dt_iop_soften_global_data_t *gd = self->global_data;

  cl_mem dev_tmp = NULL;
  cl_mem dev_m = NULL;

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  const float brightness = 1.0f / exp2f(-d->brightness);
  const float saturation = d->saturation / 100.0f;
  const float amount = d->amount / 100.0f;

  const float w = piece->iwidth * piece->iscale;
  const float h = piece->iheight * piece->iscale;
  const int mrad = dt_fast_hypotf(w, h) * 0.01f;

  const int rad = mrad * (fmin(100.0f, d->size + 1.0f) / 100.0f);
  const int radius = MIN(mrad, ceilf(rad * roi_in->scale / piece->iscale));

  /* sigma-radius correlation to match opencl
   * vs. non-opencl. identified by numerical experiments but
   * unproven. ask me if you need details. ulrich */
  const float sigma = sqrtf((radius * (radius + 1) * BOX_ITERATIONS + 2) / 3.0f);
  const int wdh = ceilf(3.0f * sigma);
  const int wd = 2 * wdh + 1;
  const size_t mat_size = sizeof(float) * wd;
  float *mat = malloc(mat_size);
  float *m = mat + wdh;
  float weight = 0.0f;

  // init gaussian kernel
  for(int l = -wdh; l <= wdh; l++)
    weight += m[l] = expf(-(l * l) / (2.f * sigma * sigma));
  for(int l = -wdh; l <= wdh; l++)
    m[l] /= weight;

  // for(int l=-wdh; l<=wdh; l++) printf("%.6f ", (double)m[l]);
  // printf("\n");

  int hblocksize;
  dt_opencl_local_buffer_t hlocopt
    = (dt_opencl_local_buffer_t){ .xoffset = 2 * wdh,
                                  .xfactor = 1,
                                  .yoffset = 0,
                                  .yfactor = 1,
                                  .cellsize = 4 * sizeof(float),
                                  .overhead = 0,
                                  .sizex = 1 << 16,
                                  .sizey = 1 };

  if(dt_opencl_local_buffer_opt(devid, gd->kernel_soften_hblur, &hlocopt) == CL_SUCCESS)
    hblocksize = hlocopt.sizex;
  else
    hblocksize = 1;

  int vblocksize;
  dt_opencl_local_buffer_t vlocopt
    = (dt_opencl_local_buffer_t){ .xoffset = 1,
                                  .xfactor = 1,
                                  .yoffset = 2 * wdh,
                                  .yfactor = 1,
                                  .cellsize = 4 * sizeof(float),
                                  .overhead = 0,
                                  .sizex = 1,
                                  .sizey = 1 << 16 };

  if(dt_opencl_local_buffer_opt(devid, gd->kernel_soften_vblur, &vlocopt) == CL_SUCCESS)
    vblocksize = vlocopt.sizey;
  else
    vblocksize = 1;


  const size_t bwidth = ROUNDUP(width, hblocksize);
  const size_t bheight = ROUNDUP(height, vblocksize);

  size_t sizes[2];
  size_t local[2];

  cl_int err = DT_OPENCL_SYSMEM_ALLOCATION;
  dev_tmp = dt_opencl_alloc_device(devid, width, height, sizeof(float) * 4);
  if(dev_tmp == NULL) goto error;

  dev_m = dt_opencl_copy_host_to_device_constant(devid, mat_size, mat);
  if(dev_m == NULL) goto error;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_soften_overexposed, width, height,
    CLARG(dev_in), CLARG(dev_tmp),
    CLARG(width), CLARG(height), CLARG(saturation), CLARG(brightness));
  if(err != CL_SUCCESS) goto error;

  if(rad != 0)
  {
    /* horizontal blur */
    sizes[0] = bwidth;
    sizes[1] = ROUNDUPDHT(height, devid);
    local[0] = hblocksize;
    local[1] = 1;
    err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_soften_hblur, sizes, local,
                              CLARG(dev_tmp), CLARG(dev_out), CLARG(dev_m),
                              CLARG(wdh), CLARG(width), CLARG(height), CLARG(hblocksize),
                              CLLOCAL((hblocksize + 2 * wdh) * 4 * sizeof(float)));
    if(err != CL_SUCCESS) goto error;


    /* vertical blur */
    sizes[0] = ROUNDUPDWD(width, devid);
    sizes[1] = bheight;
    local[0] = 1;
    local[1] = vblocksize;
    err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_soften_vblur, sizes, local,
      CLARG(dev_out), CLARG(dev_tmp), CLARG(dev_m),
      CLARG(wdh), CLARG(width), CLARG(height), CLARG(vblocksize), CLLOCAL((vblocksize + 2 * wdh) * 4 * sizeof(float)));
    if(err != CL_SUCCESS) goto error;
  }

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_soften_mix, width, height,
    CLARG(dev_in), CLARG(dev_tmp), CLARG(dev_out),
    CLARG(width), CLARG(height), CLARG(amount));

error:
  dt_opencl_release_mem_object(dev_m);
  dt_opencl_release_mem_object(dev_tmp);
  free(mat);
  return err;
}
#endif

// Push constants for the Vulkan kernels (defined unconditionally so
// init_global can size them even on HAVE_VULKAN=0 builds).
typedef struct vk_soften_overexposed_pc_t
{
  int   width;
  int   height;
  float saturation;
  float brightness;
} vk_soften_overexposed_pc_t;

typedef struct vk_soften_blur_pc_t
{
  int width;
  int height;
  int rad;
} vk_soften_blur_pc_t;

typedef struct vk_soften_mix_pc_t
{
  int   width;
  int   height;
  float amount;
} vk_soften_mix_pc_t;

#ifdef HAVE_VULKAN
// Vulkan twin of process_cl. Math is bit-equal: same HSL boost (via
// the dt_vulkan_common HSL helpers), same explicit Gaussian
// convolution with the host-computed normalised kernel, same
// CLAMP_TO_EDGE boundary via int clamp on the source index. The
// OpenCL kernel tiles the blur via workgroup-local memory; the
// Vulkan twin reads from the global storage buffer directly — the
// L1-cache pattern differs but the per-pixel sum is identical.
int process_vk(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               dt_vk_mem_t *dev_in, dt_vk_mem_t *dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  const dt_iop_soften_data_t *const d = piece->data;
  const dt_iop_soften_global_data_t *const gd = self->global_data;

  if(piece->colors != 4) return -1;

  const int devid  = piece->pipe->devid;
  const int width  = roi_in->width;
  const int height = roi_in->height;
  const size_t img_bytes = (size_t)width * height * 4 * sizeof(float);

  // Match the OpenCL host-side preparation byte-for-byte.
  const float brightness = 1.0f / exp2f(-d->brightness);
  const float saturation = d->saturation / 100.0f;
  const float amount     = d->amount / 100.0f;

  const float wf = piece->iwidth * piece->iscale;
  const float hf = piece->iheight * piece->iscale;
  const int mrad = dt_fast_hypotf(wf, hf) * 0.01f;
  const int rad_base = mrad * (fmin(100.0f, d->size + 1.0f) / 100.0f);
  const int radius = MIN(mrad, (int)ceilf(rad_base * roi_in->scale / piece->iscale));

  const float sigma = sqrtf((radius * (radius + 1) * BOX_ITERATIONS + 2) / 3.0f);
  const int wdh = ceilf(3.0f * sigma);
  const int mlen = 2 * wdh + 1;
  const size_t mat_size = sizeof(float) * mlen;

  float *mat = malloc(mat_size);
  if(!mat) return -1;
  float *m = mat + wdh;
  float weight = 0.0f;
  for(int l = -wdh; l <= wdh; l++) weight += m[l] = expf(-(l * l) / (2.f * sigma * sigma));
  for(int l = -wdh; l <= wdh; l++) m[l] /= weight;

  int rc = -1;
  dt_vk_mem_t *dev_tmp = dt_vulkan_alloc_buffer(devid, img_bytes);
  dt_vk_mem_t *dev_m   = dt_vulkan_alloc_buffer(devid, mat_size);
  if(!dev_tmp || !dev_m) goto cleanup;
  if(dt_vulkan_write_to_device(devid, dev_m, mat, mat_size) != 0) goto cleanup;

  // Step 1: HSL boost into dev_tmp.
  {
    const vk_soften_overexposed_pc_t pc = {
      .width = width, .height = height,
      .saturation = saturation, .brightness = brightness,
    };
    dt_vk_mem_t *bufs[] = { dev_in, dev_tmp };
    if(dt_vulkan_dispatch_n(&gd->vk_overexposed, bufs, 2,
                            width, height, &pc, sizeof(pc)) != 0)
      goto cleanup;
  }

  // Step 2: horizontal + vertical Gaussian (skipped when rad == 0,
  // matching the OpenCL fast path).
  if(rad_base != 0)
  {
    const vk_soften_blur_pc_t pc = { .width = width, .height = height, .rad = wdh };
    dt_vk_mem_t *bufs_h[] = { dev_tmp, dev_out, dev_m };
    if(dt_vulkan_dispatch_n(&gd->vk_hblur, bufs_h, 3,
                            width, height, &pc, sizeof(pc)) != 0)
      goto cleanup;
    dt_vk_mem_t *bufs_v[] = { dev_out, dev_tmp, dev_m };
    if(dt_vulkan_dispatch_n(&gd->vk_vblur, bufs_v, 3,
                            width, height, &pc, sizeof(pc)) != 0)
      goto cleanup;
  }

  // Step 3: blend the (blurred) overexposed reference back over the
  // original at `amount` and write to dev_out.
  {
    const vk_soften_mix_pc_t pc = { .width = width, .height = height, .amount = amount };
    dt_vk_mem_t *bufs[] = { dev_in, dev_tmp, dev_out };
    if(dt_vulkan_dispatch_n(&gd->vk_mix, bufs, 3,
                            width, height, &pc, sizeof(pc)) != 0)
      goto cleanup;
  }

  rc = 0;

cleanup:
  if(dev_tmp) dt_vulkan_free_buffer(devid, dev_tmp);
  if(dev_m)   dt_vulkan_free_buffer(devid, dev_m);
  free(mat);
  return rc;
}
#endif // HAVE_VULKAN

void tiling_callback(dt_iop_module_t *self,
                     dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  dt_iop_soften_data_t *d = piece->data;

  const float w = piece->iwidth * piece->iscale;
  const float h = piece->iheight * piece->iscale;
  const int mrad = dt_fast_hypotf(w, h) * 0.01f;

  const int rad = mrad * (fmin(100.0f, d->size + 1.0f) / 100.0f);
  const int radius = MIN(mrad, ceilf(rad * roi_in->scale / piece->iscale));

  /* sigma-radius correlation to match opencl
   * vs. non-opencl. identified by numerical experiments but
   * unproven. ask me if you need details. ulrich */
  const float sigma = sqrtf((radius * (radius + 1) * BOX_ITERATIONS + 2) / 3.0f);
  const int wdh = ceilf(3.0f * sigma);

  tiling->factor = 2.1f; // in + out + small slice for box_mean
  tiling->factor_cl = 3.0f; // in + out + tmp
  tiling->maxbuf = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = wdh;
  tiling->align = 1;
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 9; // soften.cl, from programs.conf
  dt_iop_soften_global_data_t *gd = malloc(sizeof(dt_iop_soften_global_data_t));
  self->data = gd;
  gd->kernel_soften_overexposed = dt_opencl_create_kernel(program, "soften_overexposed");
  gd->kernel_soften_hblur = dt_opencl_create_kernel(program, "soften_hblur");
  gd->kernel_soften_vblur = dt_opencl_create_kernel(program, "soften_vblur");
  gd->kernel_soften_mix = dt_opencl_create_kernel(program, "soften_mix");

  dt_vulkan_module_kernel_load(&gd->vk_overexposed, "soften_overexposed", "soften_overexposed",
                               2, sizeof(vk_soften_overexposed_pc_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&gd->vk_hblur, "soften_hblur", "soften_hblur",
                               3, sizeof(vk_soften_blur_pc_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&gd->vk_vblur, "soften_vblur", "soften_vblur",
                               3, sizeof(vk_soften_blur_pc_t), 16, 16, 1);
  dt_vulkan_module_kernel_load(&gd->vk_mix, "soften_mix", "soften_mix",
                               3, sizeof(vk_soften_mix_pc_t), 16, 16, 1);
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_soften_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_soften_overexposed);
  dt_opencl_free_kernel(gd->kernel_soften_hblur);
  dt_opencl_free_kernel(gd->kernel_soften_vblur);
  dt_opencl_free_kernel(gd->kernel_soften_mix);
  dt_vulkan_module_kernel_unload(&gd->vk_overexposed);
  dt_vulkan_module_kernel_unload(&gd->vk_hblur);
  dt_vulkan_module_kernel_unload(&gd->vk_vblur);
  dt_vulkan_module_kernel_unload(&gd->vk_mix);
  free(self->data);
  self->data = NULL;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_soften_params_t *p = (dt_iop_soften_params_t *)p1;
  dt_iop_soften_data_t *d = piece->data;

  d->size = p->size;
  d->saturation = p->saturation;
  d->brightness = p->brightness;
  d->amount = p->amount;
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_soften_data_t));
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_soften_gui_data_t *g = IOP_GUI_ALLOC(soften);

  g->size = dt_bauhaus_slider_from_params(self, N_("size"));
  dt_bauhaus_slider_set_format(g->size, "%");
  gtk_widget_set_tooltip_text(g->size, _("the size of blur"));

  g->saturation = dt_bauhaus_slider_from_params(self, N_("saturation"));
  dt_bauhaus_slider_set_format(g->saturation, "%");
  gtk_widget_set_tooltip_text(g->saturation, _("the saturation of blur"));

  g->brightness = dt_bauhaus_slider_from_params(self, N_("brightness"));
  dt_bauhaus_slider_set_format(g->brightness, _(" EV"));
  gtk_widget_set_tooltip_text(g->brightness, _("the brightness of blur"));

  g->amount = dt_bauhaus_slider_from_params(self, "amount");
  dt_bauhaus_slider_set_format(g->amount, "%");
  gtk_widget_set_tooltip_text(g->amount, _("the mix of effect"));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
