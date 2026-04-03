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

/**
 * @file develop_gui.c
 * @brief Implementation of the develop GUI API.
 *
 * This translation unit implements the API functions declared in
 * develop_gui.h, providing a clean boundary between the develop
 * business logic (develop.c) and the GTK GUI layer (darkroom.c).
 *
 * See issue #18559.
 */

#include "develop/develop_gui.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "common/darktable.h"
#include "gui/gtk.h"

#include <gtk/gtk.h>

dt_develop_gui_t *dt_dev_gui_init(void)
{
  return (dt_develop_gui_t *)g_malloc0(sizeof(dt_develop_gui_t));
}

void dt_dev_gui_cleanup(dt_develop_gui_t *gui)
{
  g_free(gui);
}

gboolean dt_dev_gui_preview2_widget_valid(const dt_develop_gui_t *gui)
{
  return gui
    && gui->preview2.widget
    && GTK_IS_WIDGET(gui->preview2.widget);
}

void dt_dev_gui_preview2_queue_draw(dt_develop_gui_t *gui)
{
  if(dt_dev_gui_preview2_widget_valid(gui))
    gtk_widget_queue_draw(gui->preview2.widget);
}

void dt_dev_gui_preview2_widget_clear(dt_develop_gui_t *gui)
{
  if(gui)
    gui->preview2.widget = NULL;
}

void dt_dev_gui_init_full_viewport(dt_develop_gui_t *gui)
{
  if(gui && darktable.gui)
    gui->full.widget = dt_ui_center(darktable.gui->ui);
}

void dt_dev_gui_update_pin_button(dt_develop_gui_t *gui,
                                  dt_develop_t *dev,
                                  gboolean pinned)
{
  if(!gui || !gui->preview2.pin_button)
    return;

  // Block signal handlers to prevent recursive callbacks
  g_signal_handlers_block_matched(G_OBJECT(gui->preview2.pin_button),
                                  G_SIGNAL_MATCH_DATA, 0, 0, NULL, NULL, dev);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gui->preview2.pin_button),
                               pinned);
  g_signal_handlers_unblock_matched(G_OBJECT(gui->preview2.pin_button),
                                    G_SIGNAL_MATCH_DATA, 0, 0, NULL, NULL, dev);
  gtk_widget_set_tooltip_text(gui->preview2.pin_button,
                              pinned ? _("unpin image") : _("pin current image"));
}

void dt_dev_gui_ensure_second_wnd_open(dt_develop_gui_t *gui)
{
  if(!gui) return;
  if(!gui->second_wnd && gui->second_wnd_button
     && GTK_IS_WIDGET(gui->second_wnd_button))
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(gui->second_wnd_button), TRUE);
}

void dt_dev_reorder_gui_module_list(dt_develop_t *dev)
{
  int pos_module = 0;
  for(const GList *modules = g_list_last(dev->iop);
      modules;
      modules = g_list_previous(modules))
  {
    dt_iop_module_t *module = modules->data;

    GtkWidget *expander = module->expander;
    if(expander)
    {
      gtk_box_reorder_child(dt_ui_get_container(darktable.gui->ui,
                                                DT_UI_CONTAINER_PANEL_RIGHT_CENTER),
                            expander,
                            pos_module++);
    }
  }
}
