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
 * @brief GTK-free public API for the develop GUI module.
 *
 * This header provides an opaque forward declaration of dt_develop_gui_t
 * and API functions that business logic code (develop.c) can call to
 * communicate with the GUI without depending on GTK directly.
 *
 * Business logic code should include ONLY this header.
 * GUI code that needs direct access to widget pointers should include
 * develop_gui_struct.h instead (which includes this header).
 *
 * See issue #18559.
 */

#include <glib.h>

struct dt_develop_t;
struct dt_dev_viewport_t;

/**
 * Opaque type for GUI data.  Full definition in develop_gui_struct.h.
 */
typedef struct dt_develop_gui_t dt_develop_gui_t;


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

/** Initialize the full viewport widget from the global GUI center area.
 *  Called during dt_dev_init() instead of the business logic passing a widget. */
void dt_dev_gui_init_full_viewport(dt_develop_gui_t *gui);

/** Update the pin button state to reflect whether an image is pinned.
 *  Blocks signal handlers during the update to prevent recursive callbacks.
 *  @param gui the GUI data
 *  @param dev the develop struct (used for signal matching)
 *  @param pinned TRUE if image is pinned, FALSE otherwise
 */
void dt_dev_gui_update_pin_button(dt_develop_gui_t *gui,
                                  struct dt_develop_t *dev,
                                  gboolean pinned);

/** Open the second window if not already open.
 *  Activates the second window toggle button.
 *  Used by filmstrip.c when pinning from filmstrip. */
void dt_dev_gui_ensure_second_wnd_open(dt_develop_gui_t *gui);

/** Queue a redraw of the widget associated with a viewport.
 *  Maps the viewport pointer to the correct GUI widget and schedules
 *  a redraw.  Safe to call when gui is NULL or the widget doesn't exist.
 *  @param gui the GUI data (may be NULL)
 *  @param port the viewport whose widget should be redrawn
 *  @param dev the develop struct (used to determine which viewport)
 */
void dt_dev_gui_queue_redraw_viewport(dt_develop_gui_t *gui,
                                      const struct dt_dev_viewport_t *port,
                                      const struct dt_develop_t *dev);
