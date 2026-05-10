/*
    This file is part of darktable,
    Copyright (C) 2009-2026 darktable developers.

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

// Toolkit-free business side of the exposure IOP. All math lives
// here:
//   - image processing (process, process_cl, commit_params)
//   - EXIF bias lookups
//   - deflicker histogram + EV computation
//   - spot picker correction (xyz/lab math + bias compensation)
//   - param mutations + black/white cascade + history
//   - color-space transforms used by the GUI's swatches and slider
//   - proxy getters registered on darktable.develop->proxy.exposure
//
// The GUI side (src/gui/iop_exposure_gui.c, compiled into the same
// exposure module .so) is a passive view: it builds widgets, reads
// dt_iop_exposure_gui_state_t for display and dispatches user input
// through the business API declared in iop/exposure.h.

#include "iop/exposure.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "common/colorspaces_inline_conversions.h"
#include "common/histogram.h"
#include "common/image_cache.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "common/mipmap_cache.h"
#include "common/opencl.h"
#include "control/conf.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/pixelpipe.h"
#include "gui/iop_exposure_gui.h"
#include "gui/presets.h"
#include "iop/iop_api.h"

#define exposure2white(x) exp2f(-(x))
#define white2exposure(x) -dt_log2f(fmaxf(1e-20f, x))

DT_MODULE_INTROSPECTION(7, dt_iop_exposure_params_t)

typedef enum dt_iop_exposure_mode_t
{
  EXPOSURE_MODE_MANUAL,   // $DESCRIPTION: "manual"
  EXPOSURE_MODE_DEFLICKER // $DESCRIPTION: "automatic"
} dt_iop_exposure_mode_t;

// uint16_t pixel can have any value in range [0, 65535], thus, there is
// 65536 possible values.
#define DEFLICKER_BINS_COUNT (UINT16_MAX + 1)

typedef struct dt_iop_exposure_params_t
{
  dt_iop_exposure_mode_t mode;      // $DEFAULT: EXPOSURE_MODE_MANUAL
  float black;                      // $MIN: -1.0 $MAX: 1.0 $DEFAULT: 0.0 $DESCRIPTION: "black level correction"
  float exposure;                   // $MIN: -18.0 $MAX: 18.0 $DEFAULT: 0.0
  float deflicker_percentile;       // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0 $DESCRIPTION: "percentile"
  float deflicker_target_level;     // $MIN: -18.0 $MAX: 18.0 $DEFAULT: -4.0 $DESCRIPTION: "target level"
  gboolean compensate_exposure_bias;// $DEFAULT: FALSE $DESCRIPTION: "compensate exposure bias"
  gboolean compensate_hilite_pres;  // $DEFAULT: TRUE $DESCRIPTION: "compensate highlight preservation"
} dt_iop_exposure_params_t;

typedef struct dt_iop_exposure_data_t
{
  dt_iop_exposure_params_t params;
  int deflicker;
  float black;
  float scale;
} dt_iop_exposure_data_t;

typedef struct dt_iop_exposure_global_data_t
{
  int kernel_exposure;
} dt_iop_exposure_global_data_t;

#define EXPOSURE_CORRECTION_UNDEFINED (-FLT_MAX)

const char *name()
{
  return _("exposure");
}

const char** description(dt_iop_module_t *self)
{
  return dt_iop_set_description
    (self,
     _("redo the exposure of the shot as if you were still in-camera\n"
       "using a color-safe brightening similar to increasing ISO setting"),
     _("corrective and creative"),
     _("linear, RGB, scene-referred"),
     _("linear, RGB"),
     _("linear, RGB, scene-referred"));
}

int default_group()
{
  return IOP_GROUP_BASIC | IOP_GROUP_TECHNICAL;
}

int flags()
{
  return IOP_FLAGS_ALLOW_TILING | IOP_FLAGS_SUPPORTS_BLENDING;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  typedef struct dt_iop_exposure_params_v6_t
  {
    dt_iop_exposure_mode_t mode;
    float black;
    float exposure;
    float deflicker_percentile;
    float deflicker_target_level;
    gboolean compensate_exposure_bias;
  } dt_iop_exposure_params_v6_t;

  typedef struct dt_iop_exposure_params_v7_t
  {
    dt_iop_exposure_mode_t mode;
    float black;
    float exposure;
    float deflicker_percentile;
    float deflicker_target_level;
    gboolean compensate_exposure_bias;
    gboolean compensate_hilite_pres;
  } dt_iop_exposure_params_v7_t;

  if(old_version == 2)
  {
    typedef struct dt_iop_exposure_params_v2_t
    {
      float black, exposure, gain;
    } dt_iop_exposure_params_v2_t;

    const dt_iop_exposure_params_v2_t *o = (dt_iop_exposure_params_v2_t *)old_params;
    dt_iop_exposure_params_v6_t *n = malloc(sizeof(dt_iop_exposure_params_v6_t));

    n->mode = EXPOSURE_MODE_MANUAL;
    n->black = o->black;
    n->exposure = o->exposure;
    n->compensate_exposure_bias = FALSE;
    n->deflicker_percentile = 50.0f;
    n->deflicker_target_level = -4.0f;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_exposure_params_v6_t);
    *new_version = 6;
    return 0;
  }
  if(old_version == 3)
  {
    typedef struct dt_iop_exposure_params_v3_t
    {
      float black, exposure;
      gboolean deflicker;
      float deflicker_percentile, deflicker_target_level;
    } dt_iop_exposure_params_v3_t;

    const dt_iop_exposure_params_v3_t *o = (dt_iop_exposure_params_v3_t *)old_params;
    dt_iop_exposure_params_v6_t *n = malloc(sizeof(dt_iop_exposure_params_v6_t));

    n->mode = o->deflicker ? EXPOSURE_MODE_DEFLICKER : EXPOSURE_MODE_MANUAL;
    n->black = o->black;
    n->exposure = o->exposure;
    n->deflicker_percentile = o->deflicker_percentile;
    n->deflicker_target_level = o->deflicker_target_level;
    n->compensate_exposure_bias = FALSE;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_exposure_params_v6_t);
    *new_version = 6;
    return 0;
  }
  if(old_version == 4)
  {
    typedef enum dt_iop_exposure_deflicker_histogram_source_t {
      DEFLICKER_HISTOGRAM_SOURCE_THUMBNAIL,
      DEFLICKER_HISTOGRAM_SOURCE_SOURCEFILE
    } dt_iop_exposure_deflicker_histogram_source_t;

    typedef struct dt_iop_exposure_params_v4_t
    {
      dt_iop_exposure_mode_t mode;
      float black;
      float exposure;
      float deflicker_percentile, deflicker_target_level;
      dt_iop_exposure_deflicker_histogram_source_t deflicker_histogram_source;
    } dt_iop_exposure_params_v4_t;

    const dt_iop_exposure_params_v4_t *o = (dt_iop_exposure_params_v4_t *)old_params;
    dt_iop_exposure_params_v6_t *n = malloc(sizeof(dt_iop_exposure_params_v6_t));

    n->mode = o->mode;
    n->black = o->black;
    n->exposure = o->exposure;
    n->deflicker_percentile = o->deflicker_percentile;
    n->deflicker_target_level = o->deflicker_target_level;
    // deflicker_histogram_source is dropped. this does change output,
    // but deflicker still was not publicly released at that point
    n->compensate_exposure_bias = FALSE;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_exposure_params_v6_t);
    *new_version = 6;
    return 0;
  }
  if(old_version == 5)
  {
    typedef struct dt_iop_exposure_params_v5_t
    {
      dt_iop_exposure_mode_t mode;
      float black;
      float exposure;
      float deflicker_percentile, deflicker_target_level;
    } dt_iop_exposure_params_v5_t;

    const dt_iop_exposure_params_v5_t *o = (dt_iop_exposure_params_v5_t *)old_params;
    dt_iop_exposure_params_v6_t *n = malloc(sizeof(dt_iop_exposure_params_v6_t));

    n->mode = o->mode;
    n->black = o->black;
    n->exposure = o->exposure;
    n->deflicker_percentile = o->deflicker_percentile;
    n->deflicker_target_level = o->deflicker_target_level;
    n->compensate_exposure_bias = FALSE;

    *new_params = n;
    *new_params_size = sizeof(dt_iop_exposure_params_v6_t);
    *new_version = 6;
    return 0;
  }
  if(old_version == 6)
  {
    const dt_iop_exposure_params_v6_t *o = (dt_iop_exposure_params_v6_t *)old_params;
    dt_iop_exposure_params_v7_t *n = malloc(sizeof(dt_iop_exposure_params_v7_t));

    n->mode = o->mode;
    n->black = o->black;
    n->exposure = o->exposure;
    n->deflicker_percentile = o->deflicker_percentile;
    n->deflicker_target_level = o->deflicker_target_level;
    n->compensate_exposure_bias = o->compensate_exposure_bias;
    n->compensate_hilite_pres = FALSE;	// module did not compensate h.p. before version 7

    *new_params = n;
    *new_params_size = sizeof(dt_iop_exposure_params_v7_t);
    *new_version = 7;
    return 0;
  }
  return 1;
}

void init_presets(dt_iop_module_so_t *self)
{
  self->pref_based_presets = TRUE;

  dt_gui_presets_add_generic
    (_("magic lantern defaults"), self->op,
     self->version(),
     &(dt_iop_exposure_params_t){.mode = EXPOSURE_MODE_DEFLICKER,
                                 .black = 0.0f,
                                 .exposure = 0.0f,
                                 .deflicker_percentile = 50.0f,
                                 .deflicker_target_level = -4.0f,
                                 .compensate_exposure_bias = FALSE,
                                 .compensate_hilite_pres = FALSE },
     sizeof(dt_iop_exposure_params_t), TRUE, DEVELOP_BLEND_CS_RGB_DISPLAY);

  const gboolean is_scene_referred = dt_is_scene_referred();

  if(is_scene_referred)
  {
    // For scene-referred workflow, since filmic doesn't brighten as base curve does,
    // we need an initial exposure boost. This preset has the same value as what is
    // auto-applied (see reload_default below) for scene-referred workflow.
    dt_gui_presets_add_generic
      (_("scene-referred default"), self->op, self->version(),
       NULL, 0,
       TRUE, DEVELOP_BLEND_CS_RGB_SCENE);

    dt_gui_presets_update_format(BUILTIN_PRESET("scene-referred default"), self->op,
                                 self->version(), FOR_RAW);

    dt_gui_presets_update_autoapply(BUILTIN_PRESET("scene-referred default"),
                                    self->op, self->version(), TRUE);
  }
}

static float _get_exposure_bias(const dt_iop_module_t *self)
{
  float bias = 0.0f;

  // just check that pointers exist and are initialized
  if(self->dev && self->dev->image_storage.exif_exposure_bias)
    bias = self->dev->image_storage.exif_exposure_bias;

  // sanity checks, don't trust exif tags too much
  if(bias != DT_EXIF_TAG_UNINITIALIZED)
    return CLAMP(bias, -5.0f, 5.0f);
  else
    return 0.0f;
}

static float _get_highlight_bias(const dt_iop_module_t *self)
{
  float bias = 0.0f;

  // Nikon: Exif.Nikon3.Colorspace==4  --> +2 EV
  // Fuji:  Exif.Fujifilm.DevelopmentDynamicRange
  //             100 --> no comp
  //             200 --> +1 EV
  //             400 --> +2 EV

  if(self->dev && self->dev->image_storage.exif_highlight_preservation > 0.0f)
    bias = self->dev->image_storage.exif_highlight_preservation;

  // sanity checks, don't trust exif tags too much
  if(bias != DT_EXIF_TAG_UNINITIALIZED)
    return CLAMP(bias, -1.0f, 4.0f);
  else
    return 0.0f;
}

static void _prepare_deflicker_histogram(dt_iop_module_t *self,
                                         uint32_t **histogram,
                                         dt_dev_histogram_stats_t *histogram_stats)
{
  const dt_image_t *img = dt_image_cache_get(self->dev->image_storage.id, 'r');
  dt_image_t image = *img;
  dt_image_cache_read_release(img);

  if(!img || image.buf_dsc.channels != 1 || image.buf_dsc.datatype != TYPE_UINT16) return;

  dt_mipmap_buffer_t buf;
  dt_mipmap_cache_get(&buf, self->dev->image_storage.id, DT_MIPMAP_FULL, DT_MIPMAP_BLOCKING, 'r');
  if(!buf.buf)
  {
    dt_control_log(_("failed to get raw buffer from image `%s'"), image.filename);
    dt_mipmap_cache_release(&buf);
    return;
  }

  dt_dev_histogram_collection_params_t histogram_params = { 0 };

  dt_histogram_roi_t histogram_roi = {.width = image.width,
                                      .height = image.height,

                                      // FIXME: get those from rawprepare IOP somehow !!!
                                      .crop_x = image.crop_x,
                                      .crop_y = image.crop_y,
                                      .crop_right = image.crop_right,
                                      .crop_bottom = image.crop_bottom };

  histogram_params.roi = &histogram_roi;
  histogram_params.bins_count = DEFLICKER_BINS_COUNT;

  dt_histogram_helper(&histogram_params, histogram_stats, IOP_CS_RAW, IOP_CS_NONE,
                      buf.buf, histogram, NULL, FALSE, NULL);

  dt_mipmap_cache_release(&buf);
}

gboolean dt_iop_exposure_is_deflicker_eligible(const dt_iop_module_t *self)
{
  return dt_image_is_raw(&self->dev->image_storage)
      && self->dev->image_storage.buf_dsc.channels == 1
      && self->dev->image_storage.buf_dsc.datatype == TYPE_UINT16;
}

gboolean dt_iop_exposure_force_manual_mode_if_ineligible(dt_iop_module_t *self)
{
  dt_iop_exposure_params_t *p = self->params;
  if(p->mode == EXPOSURE_MODE_DEFLICKER && !dt_iop_exposure_is_deflicker_eligible(self))
  {
    p->mode = EXPOSURE_MODE_MANUAL;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
    return TRUE;
  }
  return FALSE;
}

void reload_defaults(dt_iop_module_t *self)
{
  dt_iop_exposure_params_t *d = self->default_params;

  const gboolean scene_raw =
     dt_image_is_rawprepare_supported(&self->dev->image_storage)
     && dt_is_scene_referred();

  d->mode = EXPOSURE_MODE_MANUAL;

  if(scene_raw && self->multi_priority == 0)
  {
    const gboolean mono = dt_image_is_monochrome(&self->dev->image_storage);
    d->exposure = mono ? 0.0f : 0.7f;
    d->black =    mono ? 0.0f : -0.000244140625f;
    d->compensate_exposure_bias = TRUE;
  }
  else
  {
    d->exposure = 0.0f;
    d->black = 0.0f;
    d->compensate_exposure_bias = FALSE;
  }
  // the new default is to compensate for highlight preservation mode,
  // but ONLY if we're the first instance (to avoid multiple application)
  d->compensate_hilite_pres = dt_iop_is_first_instance(self->dev->iop, self);

  // Image just (re)loaded: invalidate the cached deflicker histogram
  // and refresh the EXIF biases the GUI displays. The GUI never
  // recomputes these — it only reads them from gui_state.
  dt_iop_exposure_gui_state_t *gs = dt_iop_exposure_gui_state(self);
  if(gs)
  {
    dt_free_align(gs->deflicker_histogram);
    gs->deflicker_histogram = NULL;
    gs->deflicker_computed_exposure = EXPOSURE_CORRECTION_UNDEFINED;
    gs->exposure_bias = _get_exposure_bias(self);
    gs->highlight_bias = _get_highlight_bias(self);

    // Drop any stale spot-picker derived state
    gs->spot_RGB[0] = gs->spot_RGB[1] = gs->spot_RGB[2] = gs->spot_RGB[3] = 0.f;
    gs->spot_origin_lightness = 0.f;
    gs->spot_target_lightness = dt_conf_get_float("darkroom/modules/exposure/lightness");
  }
}

/* input: 0 - 65535 (valid range: from black level to white level) */
/* output: -16 ... 0 */
static double _raw_to_ev(const uint32_t raw,
                         const uint32_t black_level,
                         const uint32_t white_level)
{
  const uint32_t raw_max = white_level - black_level;

  // we are working on data without black clipping,
  // so we can get values which are lower than the black level !!!
  const int64_t raw_val = MAX((int64_t)raw - (int64_t)black_level, 1);

  const double raw_ev = -log2(raw_max) + log2(raw_val);

  return raw_ev;
}

static void _compute_correction(dt_iop_module_t *self,
                                dt_iop_exposure_params_t *p,
                                dt_dev_pixelpipe_t *pipe,
                                const uint32_t *const histogram,
                                const dt_dev_histogram_stats_t *const histogram_stats,
                                float *correction)
{
  *correction = EXPOSURE_CORRECTION_UNDEFINED;

  if(histogram == NULL) return;

  const double thr
      = CLAMP(((double)histogram_stats->pixels * (double)p->deflicker_percentile
               / (double)100.0), 0.0, (double)histogram_stats->pixels);

  size_t n = 0;
  uint32_t raw = 0;

  for(size_t i = 0; i < histogram_stats->bins_count; i++)
  {
    n += histogram[i];

    if((double)n >= thr)
    {
      raw = i;
      break;
    }
  }

  const double ev
      = _raw_to_ev(raw, (uint32_t)pipe->dsc.rawprepare.raw_black_level,
                   pipe->dsc.rawprepare.raw_white_point);

  *correction = p->deflicker_target_level - ev;
}

static void _process_common_setup(dt_iop_module_t *self,
                                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_exposure_gui_state_t *gs = dt_iop_exposure_gui_state(self);
  dt_iop_exposure_data_t *d = piece->data;

  d->black = d->params.black;
  float exposure = d->params.exposure;

  if(d->deflicker)
  {
    if(gs)
    {
      // histogram is precomputed and cached
      _compute_correction(self, &d->params, piece->pipe,
                          gs->deflicker_histogram, &gs->deflicker_histogram_stats,
                          &exposure);
    }
    else
    {
      uint32_t *histogram = NULL;
      dt_dev_histogram_stats_t histogram_stats;
      _prepare_deflicker_histogram(self, &histogram, &histogram_stats);
      _compute_correction(self, &d->params, piece->pipe, histogram,
                          &histogram_stats, &exposure);
      dt_free_align(histogram);
    }

    // second, show computed correction in UI.
    if(gs && (piece->pipe->type & DT_DEV_PIXELPIPE_PREVIEW))
    {
      dt_iop_gui_enter_critical_section(self);
      gs->deflicker_computed_exposure = exposure;
      dt_iop_gui_leave_critical_section(self);

      dt_iop_exposure_gui_schedule_show_computed(self);
    }
  }

  const float white = exposure2white(exposure);
  d->scale = 1.0 / (white - d->black);
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  dt_iop_exposure_data_t *d = piece->data;
  dt_iop_exposure_global_data_t *gd = self->global_data;

  _process_common_setup(self, piece);

  cl_int err = DT_OPENCL_DEFAULT_ERROR;
  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_exposure, width, height,
                                         CLARG(dev_in), CLARG(dev_out),
                                         CLARG(width), CLARG(height),
                                         CLARG((d->black)), CLARG((d->scale)));
  if(err != CL_SUCCESS) goto error;
  for(int k = 0; k < 3; k++) piece->pipe->dsc.processed_maximum[k] *= d->scale;

error:
  return err;
}
#endif

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const i,
             void *const o,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const dt_iop_exposure_data_t *const d = piece->data;

  _process_common_setup(self, piece);

  const int ch = piece->colors;

  const float *const restrict in = (float*)i;
  float *const restrict out = (float*)o;
  const float black = d->black;
  const float scale = d->scale;
  const size_t npixels = (size_t)roi_out->width * roi_out->height;
  DT_OMP_FOR_SIMD(aligned(in, out : 64))
  for(size_t k = 0; k < ch * npixels; k++)
  {
    out[k] = (in[k] - black) * scale;
  }
  for(int k = 0; k < 3; k++)
    piece->pipe->dsc.processed_maximum[k] *= d->scale;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  dt_iop_exposure_params_t *p = (dt_iop_exposure_params_t *)p1;
  dt_iop_exposure_data_t *d = piece->data;

  d->params.black = p->black;
  d->params.exposure = p->exposure;
  d->params.deflicker_percentile = p->deflicker_percentile;
  d->params.deflicker_target_level = p->deflicker_target_level;

  // Apply EXIF compensations on top of the user exposure correction.
  if(p->compensate_exposure_bias)
    d->params.exposure -= _get_exposure_bias(self);
  if(p->compensate_hilite_pres)
    d->params.exposure += _get_highlight_bias(self);

  d->deflicker = (p->mode == EXPOSURE_MODE_DEFLICKER
                  && dt_iop_exposure_is_deflicker_eligible(self))
                 ? 1 : 0;

  // Push derived display values into the shared gui state. The GUI
  // never recomputes these; it just reads them. We also lazily build
  // the deflicker histogram here when the user has just switched to
  // DEFLICKER mode (or the image was reloaded).
  dt_iop_exposure_gui_state_t *gs = dt_iop_exposure_gui_state(self);
  if(gs)
  {
    gs->effective_exposure = d->params.exposure;
    gs->exposure_bias = _get_exposure_bias(self);
    gs->highlight_bias = _get_highlight_bias(self);

    if(d->deflicker && !gs->deflicker_histogram)
    {
      _prepare_deflicker_histogram(self,
                                   &gs->deflicker_histogram,
                                   &gs->deflicker_histogram_stats);
    }
  }
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1,sizeof(dt_iop_exposure_data_t));
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 2; // from programs.conf: basic.cl
  dt_iop_exposure_global_data_t *gd = calloc(1,sizeof(dt_iop_exposure_global_data_t));
  self->data = gd;
  gd->kernel_exposure = dt_opencl_create_kernel(program, "exposure");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  dt_iop_exposure_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_exposure);
  free(self->data);
  self->data = NULL;
}

/* --------------------------------------------------------------------- *
 * Business API exposed in iop/exposure.h, called from the GUI side
 * (src/gui/iop_exposure_gui.c).
 * --------------------------------------------------------------------- */

gboolean dt_iop_exposure_set_white(dt_iop_module_t *self,
                                   const float white)
{
  dt_iop_exposure_params_t *p = self->params;
  const float exposure = white2exposure(white);
  if(p->exposure == exposure) return FALSE;
  p->exposure = exposure;
  if(p->black >= white)
    dt_iop_exposure_set_black(self, white - 0.01f);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  return TRUE;
}

gboolean dt_iop_exposure_set_black(dt_iop_module_t *self,
                                   const float black)
{
  dt_iop_exposure_params_t *p = self->params;
  if(p->black == black) return FALSE;
  p->black = black;
  if(p->black >= exposure2white(p->exposure))
    dt_iop_exposure_set_white(self, p->black + 0.01f);
  dt_dev_add_history_item(darktable.develop, self, TRUE);
  return TRUE;
}

gboolean dt_iop_exposure_clamp_black_below_white(dt_iop_module_t *self)
{
  dt_iop_exposure_params_t *p = self->params;
  const float white = exposure2white(p->exposure);
  if(p->black >= white)
  {
    p->black = white - 0.01f;
    dt_dev_add_history_item(darktable.develop, self, TRUE);
    return TRUE;
  }
  return FALSE;
}

gboolean dt_iop_exposure_clamp_white_above_black(dt_iop_module_t *self)
{
  dt_iop_exposure_params_t *p = self->params;
  if(p->black >= exposure2white(p->exposure))
  {
    p->exposure = white2exposure(p->black + 0.01f);
    dt_dev_add_history_item(darktable.develop, self, TRUE);
    return TRUE;
  }
  return FALSE;
}

void dt_iop_exposure_save_spot_lightness(const float lightness)
{
  dt_conf_set_float("darkroom/modules/exposure/lightness", lightness);
}

void dt_iop_exposure_lightness_to_srgb(const float lightness, float rgb[3])
{
  const dt_aligned_pixel_t Lch = { lightness, 0.f, 0.f, 0.f };
  dt_aligned_pixel_t Lab = { 0.f };
  dt_aligned_pixel_t XYZ = { 0.f };
  dt_aligned_pixel_t RGB = { 0.f };
  dt_LCH_2_Lab(Lch, Lab);
  dt_Lab_to_XYZ(Lab, XYZ);
  dt_XYZ_to_sRGB(XYZ, RGB);
  rgb[0] = RGB[0];
  rgb[1] = RGB[1];
  rgb[2] = RGB[2];
}

void dt_iop_exposure_compute_lightness_stops(const int n_stops,
                                             const float min,
                                             const float max,
                                             float (*stops_rgb)[3])
{
  const float range = max - min;
  for(int i = 0; i < n_stops; i++)
  {
    const float stop = (float)i / (float)(n_stops - 1);
    const float lightness = min + stop * range;
    dt_iop_exposure_lightness_to_srgb(lightness, stops_rgb[i]);
  }
}

float dt_iop_exposure_proxy_get_exposure(dt_iop_module_t *self)
{
  dt_iop_exposure_params_t *p = self->params;
  return (p->mode == EXPOSURE_MODE_DEFLICKER)
    ? p->deflicker_target_level
    : p->exposure;
}

float dt_iop_exposure_proxy_get_black(dt_iop_module_t *self)
{
  return ((dt_iop_exposure_params_t *)self->params)->black;
}

float dt_iop_exposure_proxy_get_effective_exposure(dt_iop_module_t *self)
{
  const dt_iop_exposure_gui_state_t *const gs = dt_iop_exposure_gui_state(self);
  return gs ? gs->effective_exposure : 0.0f;
}

dt_iop_exposure_spot_changes_t dt_iop_exposure_apply_spot_correction
  (dt_iop_module_t *self,
   dt_dev_pixelpipe_t *pipe,
   const dt_iop_exposure_spot_action_t action,
   const float target_lightness)
{
  dt_iop_exposure_spot_changes_t changes = DT_IOP_EXPOSURE_SPOT_CHANGED_NONE;

  if(self->picked_color_max[0] < self->picked_color_min[0]) return changes;

  dt_iop_exposure_gui_state_t *gs = dt_iop_exposure_gui_state(self);
  if(!gs) return changes;

  const dt_iop_order_iccprofile_info_t *const input_profile =
    dt_ioppr_get_pipe_input_profile_info(pipe);
  if(input_profile == NULL) return changes;

  const float *RGB = self->picked_color;

  // Convert picked RGB to a neutral-grey XYZ at the same luma.
  dt_aligned_pixel_t XYZ;
  dt_aligned_pixel_t Lab;
  dot_product(RGB, input_profile->matrix_in, XYZ);
  dt_XYZ_to_Lab(XYZ, Lab);
  Lab[1] = Lab[2] = Lab[3] = 0.f;
  dt_Lab_to_XYZ(Lab, XYZ);

  // sRGB swatch for the origin spot, and L coordinate for the label.
  dt_XYZ_to_sRGB(XYZ, gs->spot_RGB);
  dt_aligned_pixel_t Lch;
  dt_Lab_2_LCH(Lab, Lch);
  gs->spot_origin_lightness = Lch[0];
  changes |= DT_IOP_EXPOSURE_SPOT_CHANGED_ORIGIN;

  dt_iop_exposure_params_t *p = self->params;

  if(action == DT_IOP_EXPOSURE_SPOT_MEASURE)
  {
    // Project the picked grey through the currently applied
    // compensations and store its target lightness; the GUI will then
    // set that value on the lightness slider.
    float expo = p->exposure;
    if(p->compensate_exposure_bias) expo -= _get_exposure_bias(self);
    if(p->compensate_hilite_pres)   expo += _get_highlight_bias(self);
    const float white = exposure2white(-expo);

    dt_aligned_pixel_t XYZ_out = { 0.0f };
    for(int c = 0; c < 3; c++) XYZ_out[c] = XYZ[c] * white;

    dt_aligned_pixel_t Lab_out;
    dt_XYZ_to_Lab(XYZ_out, Lab_out);
    Lab_out[1] = Lab_out[2] = 0.f;

    gs->spot_target_lightness = Lab_out[0];
    dt_iop_exposure_save_spot_lightness(Lab_out[0]);
    changes |= DT_IOP_EXPOSURE_SPOT_CHANGED_TARGET;
  }
  else /* DT_IOP_EXPOSURE_SPOT_CORRECT */
  {
    // Solve for the exposure that maps the picked grey onto the
    // user-supplied target lightness, then push the result through
    // dt_iop_exposure_set_white (which handles the black cascade and
    // the history snapshot).
    const dt_aligned_pixel_t Lch_target = { target_lightness, 0.f, 0.f, 0.f };
    dt_aligned_pixel_t Lab_target = { 0.f };
    dt_LCH_2_Lab(Lch_target, Lab_target);
    dt_aligned_pixel_t XYZ_target = { 0.f };
    dt_Lab_to_XYZ(Lab_target, XYZ_target);

    float white = XYZ[1] / XYZ_target[1];
    float expo = -white2exposure(white);

    if(p->compensate_exposure_bias) expo -= _get_exposure_bias(self);
    if(p->compensate_hilite_pres)   expo += _get_highlight_bias(self);

    white = exposure2white(-expo);
    if(dt_iop_exposure_set_white(self, white))
      changes |= DT_IOP_EXPOSURE_SPOT_CHANGED_EXPOSURE;
  }

  return changes;
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
