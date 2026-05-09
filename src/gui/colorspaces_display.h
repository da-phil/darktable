/*
    This file is part of darktable,
    Copyright (C) 2010-2026 darktable developers.

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

// API surface implemented by src/gui/colorspaces_display.c and called
// from common/colorspaces.c. The GUI side knows about widgets,
// monitors, X atoms, colord and win32 ICM; the business side
// (dt_colorspaces_set_display_profile) is the orchestrator and uses
// these functions as toolkit-aware queries.
//
// The header is intentionally toolkit-free (no <gtk/gtk.h>, no
// GtkWidget) so common/colorspaces.c can include it without picking
// up any GTK header.

#include "common/colorspaces.h"

#include <stdint.h>

G_BEGIN_DECLS

/** Synchronously query the display profile bytes for the given
 * profile type from the OS (X atom on X11, ICM on win32, ...). On
 * success, *buffer is set to a freshly allocated byte array of length
 * *buffer_size and *profile_source to a short human-readable
 * description of where the bytes came from; both are owned by the
 * caller and must be released with g_free(). On failure all three
 * outputs are cleared and the function is a no-op. */
void dt_gui_display_profile_query_sync
  (const dt_colorspaces_color_profile_type_t profile_type,
   uint8_t **buffer,
   int *buffer_size,
   char **profile_source);

/** If colord is available and configured for this profile type, fire
 * an asynchronous lookup. When it completes the GUI side calls
 * dt_colorspaces_install_profile_from_colord_file() in
 * common/colorspaces.c to deliver the result. No-op on platforms or
 * builds without colord. */
void dt_gui_display_profile_query_colord_async_if_configured
  (const dt_colorspaces_color_profile_type_t profile_type);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
