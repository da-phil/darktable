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
 * @file develop_gui_struct.h
 * @brief Full struct definition for dt_develop_gui_t (GTK-dependent).
 *
 * This header defines the dt_develop_gui_t struct which holds all GTK
 * widget pointers for the develop module.  It must only be included by
 * GUI translation units that need direct access to widget pointers
 * (e.g. darkroom.c, develop_gui.c, colorspaces.c, gtk.c).
 *
 * Business logic code (develop.c, filmstrip.c, etc.) should include
 * develop_gui.h instead, which provides an opaque forward declaration
 * and GTK-free API functions.
 *
 * See issue #18559.
 */

#include "develop/develop_gui.h"
#include <gtk/gtk.h>

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
 *  develop.c (business logic) should use the API functions in
 *  develop_gui.h.
 */
struct dt_develop_gui_t
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
};
