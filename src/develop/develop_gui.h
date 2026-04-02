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

#include <gtk/gtk.h>

/**
 * @file develop_gui.h
 * @brief GUI-specific data for the develop module.
 *
 * This header separates GTK widget pointers from the core develop data
 * structures defined in develop.h, keeping the business logic structs
 * free from GUI dependencies (see issue #18559).
 */

/** GUI widgets associated with a viewport (full or second window) */
typedef struct dt_dev_viewport_gui_t
{
  GtkWidget *widget;
  GtkWidget *pin_button;  // only used for preview2 viewport
} dt_dev_viewport_gui_t;

/** All GUI-specific data for the develop module */
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

/** Allocate and zero-initialize the GUI data struct */
static inline dt_develop_gui_t *dt_develop_gui_alloc(void)
{
  return (dt_develop_gui_t *)g_malloc0(sizeof(dt_develop_gui_t));
}

/** Free the GUI data struct */
static inline void dt_develop_gui_free(dt_develop_gui_t *gui)
{
  g_free(gui);
}
