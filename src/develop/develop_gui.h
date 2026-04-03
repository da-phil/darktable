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

/**
 * @file develop_gui.h
 * @brief GUI API for the develop module.
 *
 * This header provides a proper API between the develop business logic
 * (develop.h/develop.c) and the GUI layer (darkroom.c).  It defines both
 * the GUI data structure (dt_develop_gui_t) and API functions that the
 * business logic can call to communicate with the GUI without depending
 * on GTK directly.
 *
 * The GUI data structure is defined in full here so that darkroom.c
 * (the GUI translation unit) can access widget pointers directly.
 * The business logic in develop.c should ONLY use the API functions
 * declared below, never access the struct members.
 *
 * See issue #18559.
 */

#include <gtk/gtk.h>

struct dt_develop_t;

/** GUI widgets associated with a viewport (full or second window) */
typedef struct dt_dev_viewport_gui_t
{
  GtkWidget *widget;
  GtkWidget *pin_button;  // only used for preview2 viewport
} dt_dev_viewport_gui_t;

/** All GUI-specific data for the develop module.
 *
 *  This struct holds ALL GtkWidget pointers that were previously
 *  stored directly in dt_develop_t and dt_dev_viewport_t.
 *
 *  darkroom.c (the GUI layer) accesses these members directly.
 *  develop.c (business logic) should use the API functions below.
 */
typedef struct dt_develop_gui_t
{
  // viewport widgets
  dt_dev_viewport_gui_t full;
  dt_dev_viewport_gui_t preview2;

  // for the overexposure indicator
  struct
  {
    GtkWidget *floating_window, *button;
  } overexposed;

  // for the raw overexposure indicator
  struct
  {
    GtkWidget *floating_window, *button;
  } rawoverexposed;

  // Color assessment conditions
  struct
  {
    GtkWidget *floating_window, *button;
  } color_assessment;

  // late scaling down from full roi
  struct
  {
    GtkWidget *button;
  } late_scaling;

  // the display profile related things (softproof, gamut check, profiles ...)
  struct
  {
    GtkWidget *floating_window, *softproof_button, *gamut_button;
  } profile;

  // second window and its toggle button
  GtkWidget *second_wnd, *second_wnd_button;
} dt_develop_gui_t;


/*
 * API functions for business logic -> GUI communication.
 *
 * These functions provide a clean interface that develop.c can call
 * without including <gtk/gtk.h> or knowing widget internals.
 * They are implemented in develop_gui.c.
 */

/** Allocate and zero-initialize the GUI data struct. */
dt_develop_gui_t *dt_dev_gui_init(void);

/** Free the GUI data struct. */
void dt_dev_gui_cleanup(dt_develop_gui_t *gui);

/** Check whether the preview2 viewport widget exists and is valid. */
gboolean dt_dev_gui_preview2_widget_valid(const dt_develop_gui_t *gui);

/** Request a redraw of the preview2 viewport widget. */
void dt_dev_gui_preview2_queue_draw(dt_develop_gui_t *gui);

/** Clear the preview2 widget reference (sets it to NULL).
 *  Used when tearing down a pinned dev that shares the gui struct. */
void dt_dev_gui_preview2_widget_clear(dt_develop_gui_t *gui);

/** Set the full viewport widget reference during initialization. */
void dt_dev_gui_set_full_widget(dt_develop_gui_t *gui, GtkWidget *widget);

/** Update the pin button state to reflect whether an image is pinned.
 *  Blocks signal handlers during the update to prevent recursive callbacks.
 *  @param gui the GUI data
 *  @param dev the develop struct (used for signal matching)
 *  @param pinned TRUE if image is pinned, FALSE otherwise
 */
void dt_dev_gui_update_pin_button(dt_develop_gui_t *gui,
                                  struct dt_develop_t *dev,
                                  gboolean pinned);

/** Get the GtkWidget* for the second window, or NULL if not open.
 *  This is needed by colorspaces.c and gtk.c to query the display
 *  for color profile information. */
GtkWidget *dt_dev_gui_get_second_wnd(const dt_develop_gui_t *gui);

/** Open the second window if not already open.
 *  Activates the second window toggle button.
 *  Used by filmstrip.c when pinning from filmstrip. */
void dt_dev_gui_ensure_second_wnd_open(dt_develop_gui_t *gui);
