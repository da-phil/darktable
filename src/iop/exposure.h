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

// Business API of the exposure IOP, exposed to its GUI translation
// unit (src/gui/iop_exposure_gui.c). Everything in here lives in
// src/iop/exposure.c and is intentionally toolkit-free. The GUI side
// calls these functions to:
//   - apply user input (slider edits, spot picker, color picker hook)
//   - read params / derived state for display
// and then re-syncs its widgets based on what changed.

#include <glib.h>

G_BEGIN_DECLS

struct dt_iop_module_t;
struct dt_dev_pixelpipe_t;

/** Mode of the area-exposure-mapping spot picker. */
typedef enum dt_iop_exposure_spot_action_t
{
  DT_IOP_EXPOSURE_SPOT_MEASURE,
  DT_IOP_EXPOSURE_SPOT_CORRECT,
} dt_iop_exposure_spot_action_t;

/** Bitmask describing which derived values
 * dt_iop_exposure_apply_spot_correction() touched. The GUI uses it to
 * decide which widgets to refresh. */
typedef enum dt_iop_exposure_spot_changes_t
{
  DT_IOP_EXPOSURE_SPOT_CHANGED_NONE     = 0,
  /* state.spot_RGB and state.spot_origin_lightness updated */
  DT_IOP_EXPOSURE_SPOT_CHANGED_ORIGIN   = 1 << 0,
  /* state.spot_target_lightness updated (MEASURE mode) */
  DT_IOP_EXPOSURE_SPOT_CHANGED_TARGET   = 1 << 1,
  /* p->exposure (and possibly p->black) updated (CORRECT mode) */
  DT_IOP_EXPOSURE_SPOT_CHANGED_EXPOSURE = 1 << 2,
} dt_iop_exposure_spot_changes_t;

/** Apply the spot picker to params and/or derived state. Performs all
 * color-space math and EXIF bias compensation internally. Reads
 * self->picked_color and the input profile from `pipe`. */
dt_iop_exposure_spot_changes_t dt_iop_exposure_apply_spot_correction
  (struct dt_iop_module_t *self,
   struct dt_dev_pixelpipe_t *pipe,
   dt_iop_exposure_spot_action_t action,
   float target_lightness);

/** Color math: convert a Lab lightness (0..100) to sRGB for a neutral
 * grey of that lightness. Used by the GUI's target swatch. */
void dt_iop_exposure_lightness_to_srgb(float lightness, float rgb[3]);

/** Color math: fill n_stops sRGB triples sweeping Lab lightness from
 * `min` to `max`. Used to colorize the GUI's lightness slider. */
void dt_iop_exposure_compute_lightness_stops
  (int n_stops, float min, float max, float (*stops_rgb)[3]);

/** Param mutators. Update the corresponding param, enforce the
 * black < exposure2white(exposure) constraint via a one-step cascade,
 * and record one or two history items. Return TRUE iff anything in p
 * actually changed; the caller can use this to decide whether widget
 * re-sync is necessary. */
gboolean dt_iop_exposure_set_white(struct dt_iop_module_t *self, float white);
gboolean dt_iop_exposure_set_black(struct dt_iop_module_t *self, float black);

/** Constraint helpers used by gui_changed when one of the two sliders
 * has been edited directly by the user. They push the other param
 * just past the boundary, add one history item and return TRUE iff
 * something changed. */
gboolean dt_iop_exposure_clamp_black_below_white(struct dt_iop_module_t *self);
gboolean dt_iop_exposure_clamp_white_above_black(struct dt_iop_module_t *self);

/** Persist the user's spot-mode target lightness in the per-user
 * config (no toolkit dependency on the call side). */
void dt_iop_exposure_save_spot_lightness(float lightness);

/** TRUE iff the current image satisfies the constraints required for
 * automatic deflicker (raw, single channel, 16-bit). */
gboolean dt_iop_exposure_is_deflicker_eligible(const struct dt_iop_module_t *self);

/** If the params currently request DEFLICKER on an ineligible image,
 * force them back to MANUAL and add a history item. Returns TRUE iff
 * anything was changed. */
gboolean dt_iop_exposure_force_manual_mode_if_ineligible(struct dt_iop_module_t *self);

/** Proxy getters registered on darktable.develop->proxy.exposure by
 * the GUI side at gui_init time. Pure reads of params / state. */
float dt_iop_exposure_proxy_get_exposure(struct dt_iop_module_t *self);
float dt_iop_exposure_proxy_get_black(struct dt_iop_module_t *self);
float dt_iop_exposure_proxy_get_effective_exposure(struct dt_iop_module_t *self);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
