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

// All the GTK specific bits of the filmstrip lib module live here.
// src/libs/tools/filmstrip.c is the toolkit-free entry point; this
// translation unit is compiled into the same filmstrip module .so
// (see src/libs/CMakeLists.txt).

#include "gui/lib_filmstrip_gui.h"

#include "common/darktable.h"
#include "develop/develop.h"
#include "dtgtk/thumbtable.h"
#include "gui/gtk.h"
#include "libs/lib.h"

#include <gtk/gtk.h>

static gboolean _draw_callback(GtkWidget *widget,
                               cairo_t *wcr,
                               gpointer user_data)
{
  // we only ensure that the thumbtable is inside our container
  if(!gtk_bin_get_child(GTK_BIN(widget)))
  {
    dt_thumbtable_t *tt = dt_ui_thumbtable(darktable.gui->ui);
    dt_thumbtable_set_parent(tt, widget, DT_THUMBTABLE_MODE_FILMSTRIP);
    gtk_widget_show(widget);
    gtk_widget_show(tt->widget);
    gtk_widget_queue_draw(tt->widget);
  }
  return FALSE;
}

void dt_lib_filmstrip_gui_init(dt_lib_module_t *self)
{
  /* creating container area */
  self->widget = gtk_event_box_new();

  /* connect callbacks */
  g_signal_connect(G_OBJECT(self->widget), "draw",
                   G_CALLBACK(_draw_callback), self);
}

void dt_lib_filmstrip_gui_cleanup(dt_lib_module_t *self)
{
  // The widget is owned by the lib framework; nothing to free here
  // beyond what gui_cleanup() in filmstrip.c already does.
  (void)self;
}

void dt_lib_filmstrip_gui_ensure_second_window(dt_develop_t *dev)
{
  if(!dev) return;
  if(!dev->second_wnd && dev->second_wnd_button)
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(dev->second_wnd_button), TRUE);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
