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

// Toolkit side of the display-profile lookup. The orchestration —
// acquiring the xprofile_lock, comparing bytes against the cached
// profile, installing them and raising the signal — lives in
// common/colorspaces.c. This file only knows how to talk to GTK / GDK
// / colord-gtk / win32 to obtain raw profile bytes for the monitor
// the main or second darkroom window currently sits on.

#include "gui/colorspaces_display.h"

#include "common/colorspaces.h"
#include "common/darktable.h"
#include "common/debug.h"
#include "control/conf.h"
#include "develop/develop.h"
#include "gui/gtk.h"

#include <gtk/gtk.h>
#include <string.h>

#ifdef USE_COLORDGTK
#include "colord-gtk.h"
#endif

#ifdef _WIN32
#include <dwmapi.h>
#include <gdk/gdkwin32.h>
#endif

#if 0
#include <ApplicationServices/ApplicationServices.h>
#include <Carbon/Carbon.h>
#include <CoreServices/CoreServices.h>
#endif

// Resolve the widget that hosts the monitor we want to query for the
// given profile type. For DT_COLORSPACE_DISPLAY2 we use the second
// darkroom window; otherwise we use the main center area.
static GtkWidget *_display_widget(const dt_colorspaces_color_profile_type_t profile_type)
{
  return (profile_type == DT_COLORSPACE_DISPLAY2)
    ? darktable.develop->second_wnd
    : dt_ui_center(darktable.gui->ui);
}

#if defined GDK_WINDOWING_X11
static int _gtk_get_monitor_num(GdkMonitor *monitor)
{
  GdkDisplay *display = gdk_monitor_get_display(monitor);
  const int n_monitors = gdk_display_get_n_monitors(display);
  for(int i = 0; i < n_monitors; i++)
  {
    if(gdk_display_get_monitor(display, i) == monitor) return i;
  }

  return -1;
}

// Read the _ICC_PROFILE / _ICC_PROFILE_n root-window atom that holds
// the ICC profile for the monitor under "widget". Allocates the
// returned buffer; caller frees with g_free(). Sets *profile_source
// to a newly allocated description string.
static gboolean _query_xatom(GtkWidget *widget,
                             uint8_t **buffer,
                             int *buffer_size,
                             char **profile_source)
{
  GdkWindow *window = gtk_widget_get_window(widget);
  GdkScreen *screen = gtk_widget_get_screen(widget);
  if(screen == NULL) screen = gdk_screen_get_default();

  GdkDisplay *display = gtk_widget_get_display(widget);
  const int monitor =
    _gtk_get_monitor_num(gdk_display_get_monitor_at_window(display, window));

  char *atom_name;
  if(monitor > 0)
    atom_name = g_strdup_printf("_ICC_PROFILE_%d", monitor);
  else
    atom_name = g_strdup("_ICC_PROFILE");

  *profile_source = g_strdup_printf("xatom %s", atom_name);

  GdkAtom type = GDK_NONE;
  gint format = 0;
  gint size_out = 0;
  guint8 *buf_out = NULL;
  gdk_property_get(gdk_screen_get_root_window(screen),
                   gdk_atom_intern(atom_name, FALSE), GDK_NONE, 0,
                   64 * 1024 * 1024, FALSE, &type, &format, &size_out, &buf_out);
  g_free(atom_name);

  *buffer = buf_out;
  *buffer_size = size_out;
  return buf_out != NULL && size_out > 0;
}
#endif

void dt_gui_display_profile_query_sync
  (const dt_colorspaces_color_profile_type_t profile_type,
   uint8_t **buffer,
   int *buffer_size,
   char **profile_source)
{
  *buffer = NULL;
  *buffer_size = 0;
  *profile_source = NULL;

#if defined GDK_WINDOWING_X11

  // On X11 we may consult the X atom, the colord daemon, or both,
  // depending on user configuration. The async colord path goes
  // through dt_gui_display_profile_query_colord_async_if_configured;
  // here we only do the synchronous xatom read.
  gboolean use_xatom = TRUE;
#if defined USE_COLORDGTK
  const char *display_profile_source =
    (profile_type == DT_COLORSPACE_DISPLAY2)
    ? dt_conf_get_string_const("ui_last/display2_profile_source")
    : dt_conf_get_string_const("ui_last/display_profile_source");
  if(display_profile_source && !strcmp(display_profile_source, "colord"))
    use_xatom = FALSE;
#endif

  if(use_xatom)
    _query_xatom(_display_widget(profile_type),
                 buffer, buffer_size, profile_source);

#elif defined GDK_WINDOWING_QUARTZ
#if 0
  GtkWidget *widget = _display_widget(profile_type);
  GdkScreen *screen = gtk_widget_get_screen(widget);
  if(screen == NULL) screen = gdk_screen_get_default();
  const int monitor =
    gdk_screen_get_monitor_at_window(screen, gtk_widget_get_window(widget));

  CGDirectDisplayID ids[monitor + 1];
  uint32_t total_ids;
  CMProfileRef prof = NULL;
  if(CGGetOnlineDisplayList(monitor + 1, &ids[0], &total_ids)
     == kCGErrorSuccess && total_ids == monitor + 1)
    CMGetProfileByAVID(ids[monitor], &prof);

  if(prof != NULL)
  {
    CFDataRef data;
    data = CMProfileCopyICCData(NULL, prof);
    CMCloseProfile(prof);

    UInt8 *tmp_buffer = (UInt8 *)g_malloc(CFDataGetLength(data));
    CFDataGetBytes(data, CFRangeMake(0, CFDataGetLength(data)), tmp_buffer);

    *buffer = (uint8_t *)tmp_buffer;
    *buffer_size = CFDataGetLength(data);

    CFRelease(data);
  }
  *profile_source = g_strdup("osx color profile api");
#endif
#elif defined G_OS_WIN32
  GtkWidget *widget = _display_widget(profile_type);
  GdkWindow *window = gtk_widget_get_window(widget);
  HWND hwnd = (HWND)gdk_win32_window_get_handle(window);
  HMONITOR hMonitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
  if(!hMonitor)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[win32 dt_gui_display_profile_query_sync] error getting monitor handle");
    return;
  }
  MONITORINFOEX monitorInfo;
  monitorInfo.cbSize = sizeof(MONITORINFOEX);
  if(!GetMonitorInfoW(hMonitor, (LPMONITORINFO)&monitorInfo))
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[win32 dt_gui_display_profile_query_sync] error getting monitor info");
    return;
  }
  HDC hdc = CreateIC(L"MONITOR", monitorInfo.szDevice, NULL, NULL);
  if(hdc != NULL)
  {
    DWORD len = 0;
    GetICMProfile(hdc, &len, NULL);
    wchar_t *wpath = g_new(wchar_t, len);

    if(GetICMProfileW(hdc, &len, wpath))
    {
      gchar *path = g_utf16_to_utf8(wpath, -1, NULL, NULL, NULL);
      if(path)
      {
        gsize size;
        g_file_get_contents(path, (gchar **)buffer, &size, NULL);
        *buffer_size = size;
        g_free(path);
      }
    }
    g_free(wpath);
    DeleteDC(hdc);
  }
  *profile_source = g_strdup("windows color profile api");
#else
  (void)profile_type;
#endif
}

#ifdef USE_COLORDGTK
static void _colord_callback(GObject *source,
                             GAsyncResult *res,
                             gpointer user_data)
{
  const dt_colorspaces_color_profile_type_t profile_type
      = (dt_colorspaces_color_profile_type_t)GPOINTER_TO_INT(user_data);

  CdWindow *window = CD_WINDOW(source);
  GError *error = NULL;
  CdProfile *profile = cd_window_get_profile_finish(window, res, &error);
  if(error == NULL && profile != NULL)
  {
    const gchar *filename = cd_profile_get_filename(profile);
    if(filename)
    {
      // Hand the result to the business orchestrator. It owns the
      // xprofile lock, the byte comparison and the signal.
      dt_colorspaces_install_profile_from_colord_file(profile_type, filename);
    }
  }
  if(profile)
    g_object_unref(profile);
  g_object_unref(window);
}
#endif

void dt_gui_display_profile_query_colord_async_if_configured
  (const dt_colorspaces_color_profile_type_t profile_type)
{
#if defined GDK_WINDOWING_X11 && defined USE_COLORDGTK
  // User-configurable preference: "xatom" disables the colord path.
  const char *display_profile_source =
    (profile_type == DT_COLORSPACE_DISPLAY2)
    ? dt_conf_get_string_const("ui_last/display2_profile_source")
    : dt_conf_get_string_const("ui_last/display_profile_source");
  if(display_profile_source && !strcmp(display_profile_source, "xatom"))
    return;

  CdWindow *window = cd_window_new();
  GtkWidget *center_widget = _display_widget(profile_type);
  cd_window_get_profile(window, center_widget, NULL,
                        _colord_callback,
                        GINT_TO_POINTER(profile_type));
#else
  (void)profile_type;
#endif
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
