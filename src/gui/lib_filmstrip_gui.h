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

#pragma once

#include <glib.h>

G_BEGIN_DECLS

struct dt_lib_module_t;
struct dt_develop_t;

/** Build the filmstrip lib's container widget, attach the draw
 * callback that re-parents the thumbtable into it, and wire the
 * actions it owns. Implemented in src/gui/lib_filmstrip_gui.c so
 * src/libs/tools/filmstrip.c stays free of any GTK header. */
void dt_lib_filmstrip_gui_init(struct dt_lib_module_t *self);

/** Release whatever dt_lib_filmstrip_gui_init() set up. */
void dt_lib_filmstrip_gui_cleanup(struct dt_lib_module_t *self);

/** If the second window is currently closed and the toggle button has
 * been built, simulate a click on it so the window opens. Called from
 * the "pin in second window" action handler in filmstrip.c. */
void dt_lib_filmstrip_gui_ensure_second_window(struct dt_develop_t *dev);

G_END_DECLS

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
