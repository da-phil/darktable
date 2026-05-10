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

#pragma once

// GUI-side port consumed by the IOP business logic in
// src/iop/exposure.c. Business pushes derived display values into
// dt_iop_exposure_gui_state_t and notifies the GUI via
// dt_iop_exposure_gui_schedule_show_computed() when a fresh
// deflicker EC value is ready. This header is intentionally toolkit
// free; the matching business->GUI API lives in src/iop/exposure.h.

#include "common/dttypes.h"
#include "common/histogram.h"

#include <stdint.h>

G_BEGIN_DECLS

struct dt_iop_module_t;

/** Non-toolkit slice of exposure.c's gui data. The IOP business
 * writes every field here from src/iop/exposure.c (reload_defaults,
 * commit_params, dt_iop_exposure_apply_spot_correction, ...); the
 * GUI only reads it for display and never recomputes any of the
 * derived values. The full dt_iop_exposure_gui_data_t (which also
 * holds GtkWidget pointers) is an internal type of
 * src/gui/iop_exposure_gui.c and embeds this struct as its first
 * member. */
typedef struct dt_iop_exposure_gui_state_t
{
  /* deflicker histogram, fully owned by the IOP business side */
  dt_dev_histogram_stats_t deflicker_histogram_stats;
  uint32_t *deflicker_histogram;
  float deflicker_computed_exposure;

  /* effective exposure after all compensations, written from
     commit_params, read from the proxy getter */
  float effective_exposure;

  /* cached EXIF biases, refreshed when the image or params change */
  float exposure_bias;
  float highlight_bias;

  /* derived spot-picker state, written by
     dt_iop_exposure_apply_spot_correction() */
  dt_aligned_pixel_t spot_RGB;        /* sRGB of the picked area     */
  float spot_origin_lightness;        /* Lab L for the "L:" label    */
  float spot_target_lightness;        /* Lab L proposed in MEASURE   */
} dt_iop_exposure_gui_state_t;

/** Returns a pointer to the shared, non-GTK gui state, or NULL when
 * the module currently has no gui (e.g. headless export). */
dt_iop_exposure_gui_state_t *dt_iop_exposure_gui_state
  (struct dt_iop_module_t *self);

/** Schedule a refresh of the "computed EC" label on the next idle.
 * Called from the IOP business side after a deflicker pass updates
 * state.deflicker_computed_exposure. Implemented in
 * iop_exposure_gui.c so business never has to know about the
 * underlying widget. */
void dt_iop_exposure_gui_schedule_show_computed
  (struct dt_iop_module_t *self);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
