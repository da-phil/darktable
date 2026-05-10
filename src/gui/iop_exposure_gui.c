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

// GUI half of the exposure IOP. Pure widget plumbing:
//   - builds the bauhaus widget tree in gui_init
//   - syncs widgets from params / gui state in gui_update
//   - reads widgets, dispatches to the business API
//     (src/iop/exposure.c via iop/exposure.h) on user input, then
//     re-syncs any widgets the business reports having changed.
//
// All math (color transforms, EXIF bias, deflicker, spot picker
// correction, slider stop computation, param mutation with cascade,
// history, conf persistence) lives in src/iop/exposure.c. This
// translation unit must never duplicate any of that logic; it only
// reads dt_iop_exposure_gui_state_t for display.

#include "bauhaus/bauhaus.h"
#include "common/image_cache.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_gui.h"
#include "dtgtk/paint.h"
#include "dtgtk/resetlabel.h"
#include "gui/accelerators.h"
#include "gui/color_picker_proxy.h"
#include "gui/gtk.h"
#include "gui/iop_exposure_gui.h"
#include "iop/exposure.h"

#include <gtk/gtk.h>

#define EXPOSURE_CORRECTION_UNDEFINED (-FLT_MAX)

// Mirror the enums and params struct from exposure.c. The
// introspection scanner only looks at the IOP main file, so we just
// need a layout-compatible declaration here for field access.
typedef enum dt_iop_exposure_mode_t
{
  EXPOSURE_MODE_MANUAL,
  EXPOSURE_MODE_DEFLICKER
} dt_iop_exposure_mode_t;

typedef enum dt_spot_mode_t
{
  DT_SPOT_MODE_CORRECT = 0,
  DT_SPOT_MODE_MEASURE = 1,
  DT_SPOT_MODE_LAST
} dt_spot_mode_t;

typedef struct dt_iop_exposure_params_t
{
  dt_iop_exposure_mode_t mode;
  float black;
  float exposure;
  float deflicker_percentile;
  float deflicker_target_level;
  gboolean compensate_exposure_bias;
  gboolean compensate_hilite_pres;
} dt_iop_exposure_params_t;

typedef struct dt_iop_exposure_gui_data_t
{
  // The shared, GTK-free state must be the first member so that
  // dt_iop_exposure_gui_state() can return a pointer to it.
  dt_iop_exposure_gui_state_t state;

  GtkWidget *mode;
  GtkWidget *black;
  GtkStack *mode_stack;
  GtkWidget *exposure;
  GtkWidget *deflicker_percentile;
  GtkWidget *deflicker_target_level;
  GtkLabel *deflicker_used_EC;
  GtkWidget *compensate_exposure_bias;
  GtkWidget *compensate_hilite_preserv;

  GtkWidget *spot_mode;
  GtkWidget *lightness_spot;
  GtkWidget *origin_spot, *target_spot;
  GtkWidget *Lch_origin;

  dt_gui_collapsible_section_t cs;
} dt_iop_exposure_gui_data_t;

// --- GUI port implementation (called from iop/exposure.c) ---

dt_iop_exposure_gui_state_t *dt_iop_exposure_gui_state(dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;
  return g ? &g->state : NULL;
}

static gboolean _show_computed(gpointer user_data);

void dt_iop_exposure_gui_schedule_show_computed(dt_iop_module_t *self)
{
  g_idle_add(_show_computed, self);
}

// --- Widget sync helpers ---

// Push p->exposure / p->black into their sliders without re-triggering
// the bauhaus value-changed callback (which would otherwise add a
// duplicate history item).
static void _sync_exposure_widgets(dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;
  dt_iop_exposure_params_t *p = self->params;
  ++darktable.gui->reset;
  dt_bauhaus_slider_set(g->exposure, p->exposure);
  dt_bauhaus_slider_set(g->black, p->black);
  --darktable.gui->reset;
}

// Apply business-computed lightness stops to the slider and queue
// redraws. The math itself is done in
// dt_iop_exposure_compute_lightness_stops().
static void _refresh_lightness_stops(dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;

  const float lmin = dt_bauhaus_slider_get_hard_min(g->lightness_spot);
  const float lmax = dt_bauhaus_slider_get_hard_max(g->lightness_spot);

  float stops[DT_BAUHAUS_SLIDER_MAX_STOPS][3];
  dt_iop_exposure_compute_lightness_stops(DT_BAUHAUS_SLIDER_MAX_STOPS,
                                          lmin, lmax, stops);

  for(int i = 0; i < DT_BAUHAUS_SLIDER_MAX_STOPS; i++)
  {
    const float pos = (float)i / (float)(DT_BAUHAUS_SLIDER_MAX_STOPS - 1);
    dt_bauhaus_slider_set_stop(g->lightness_spot, pos,
                               stops[i][0], stops[i][1], stops[i][2]);
  }

  gtk_widget_queue_draw(g->lightness_spot);
  gtk_widget_queue_draw(g->target_spot);
}

static void _refresh_origin_label(dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;
  gchar *str = g_strdup_printf(_("L : \t%.1f %%"), g->state.spot_origin_lightness);
  ++darktable.gui->reset;
  gtk_label_set_text(GTK_LABEL(g->Lch_origin), str);
  --darktable.gui->reset;
  g_free(str);
  gtk_widget_queue_draw(g->origin_spot);
}

// --- IOP API: color picker apply hook ---

void color_picker_apply(dt_iop_module_t *self,
                        GtkWidget *picker,
                        dt_dev_pixelpipe_t *pipe)
{
  if(darktable.gui->reset) return;
  dt_iop_exposure_gui_data_t *g = self->gui_data;

  // Read the user's current intent from the picker widgets ...
  const int mode = dt_bauhaus_combobox_get(g->spot_mode);
  const float lightness = dt_bauhaus_slider_get(g->lightness_spot);

  // ... and let the business side do the actual color math + param
  // update. It writes results into the shared gui state and returns
  // a bitmask telling us which widgets need refreshing.
  const dt_iop_exposure_spot_changes_t changes =
    dt_iop_exposure_apply_spot_correction
      (self, pipe,
       (mode == DT_SPOT_MODE_MEASURE)
         ? DT_IOP_EXPOSURE_SPOT_MEASURE
         : DT_IOP_EXPOSURE_SPOT_CORRECT,
       lightness);

  if(changes & DT_IOP_EXPOSURE_SPOT_CHANGED_ORIGIN)
    _refresh_origin_label(self);

  if(changes & DT_IOP_EXPOSURE_SPOT_CHANGED_TARGET)
  {
    ++darktable.gui->reset;
    dt_bauhaus_slider_set(g->lightness_spot, g->state.spot_target_lightness);
    _refresh_lightness_stops(self);
    --darktable.gui->reset;
  }

  if(changes & DT_IOP_EXPOSURE_SPOT_CHANGED_EXPOSURE)
    _sync_exposure_widgets(self);
}

// --- IOP API: gui_changed dispatch ---

static void _autoexp_disable(dt_iop_module_t *self)
{
  dt_iop_color_picker_reset(self, TRUE);
}

void gui_changed(dt_iop_module_t *self,
                 GtkWidget *w,
                 void *previous)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;
  dt_iop_exposure_params_t *p = self->params;

  if(w == g->mode)
  {
    // Mode toggled. The deflicker histogram is owned by the IOP and
    // will be (re)built lazily in commit_params next time we run with
    // DEFLICKER mode active. Here we only adjust the visible stack
    // and ensure the params stay valid for this image.
    switch(p->mode)
    {
      case EXPOSURE_MODE_DEFLICKER:
        _autoexp_disable(self);
        if(!dt_iop_exposure_is_deflicker_eligible(self))
        {
          // Business: force MANUAL + history. GUI: re-sync combobox + sensitivity.
          dt_iop_exposure_force_manual_mode_if_ineligible(self);
          dt_bauhaus_combobox_set(g->mode, p->mode);
          gtk_widget_set_sensitive(GTK_WIDGET(g->mode), FALSE);
          break;
        }
        gtk_stack_set_visible_child_name(GTK_STACK(g->mode_stack), "deflicker");
        break;
      case EXPOSURE_MODE_MANUAL:
      default:
        gtk_stack_set_visible_child_name(GTK_STACK(g->mode_stack), "manual");
        break;
    }
  }
  else if(w == g->exposure)
  {
    // User edited the exposure slider; enforce the black < white
    // constraint via the business helper and resync if necessary.
    if(dt_iop_exposure_clamp_black_below_white(self))
    {
      ++darktable.gui->reset;
      dt_bauhaus_slider_set(g->black, p->black);
      --darktable.gui->reset;
    }
  }
  else if(w == g->black)
  {
    if(dt_iop_exposure_clamp_white_above_black(self))
    {
      ++darktable.gui->reset;
      dt_bauhaus_slider_set(g->exposure, p->exposure);
      --darktable.gui->reset;
    }
  }
}

// --- IOP API: gui_update ---

void gui_update(dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;
  dt_iop_exposure_params_t *p = self->params;

  // Business decides whether DEFLICKER is allowed on this image; the
  // GUI just reflects the decision in the combobox sensitivity.
  dt_iop_exposure_force_manual_mode_if_ineligible(self);
  gtk_widget_set_sensitive(GTK_WIDGET(g->mode),
                           dt_iop_exposure_is_deflicker_eligible(self));

  dt_iop_color_picker_reset(self, TRUE);

  // Read cached EXIF biases from the gui state. The IOP refreshes
  // them in reload_defaults / commit_params; we never recompute them.
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->compensate_exposure_bias),
                               p->compensate_exposure_bias);
  gchar *label = g_strdup_printf(_("compensate camera exposure (%+.1f EV)"),
                                 g->state.exposure_bias);
  gtk_button_set_label(GTK_BUTTON(g->compensate_exposure_bias), label);
  gtk_label_set_ellipsize
    (GTK_LABEL(gtk_bin_get_child(GTK_BIN(g->compensate_exposure_bias))),
     PANGO_ELLIPSIZE_MIDDLE);
  g_free(label);

  const float hlbias = g->state.highlight_bias;
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(g->compensate_hilite_preserv),
                               p->compensate_hilite_pres);
  /* xgettext:no-c-format */
  label = g_strdup_printf(_("highlight preservation mode (%.1f EV)"), hlbias);
  gtk_button_set_label(GTK_BUTTON(g->compensate_hilite_preserv), label);
  gtk_label_set_ellipsize
    (GTK_LABEL(gtk_bin_get_child(GTK_BIN(g->compensate_hilite_preserv))),
     PANGO_ELLIPSIZE_MIDDLE);
  g_free(label);
  gtk_widget_set_visible(GTK_WIDGET(g->compensate_hilite_preserv), hlbias > 0.0f);

  // The spot-picker derived state was reset by reload_defaults; just
  // mirror it into the lightness slider.
  dt_bauhaus_slider_set(g->lightness_spot, g->state.spot_target_lightness);

  gtk_label_set_text(g->deflicker_used_EC, "");

  switch(p->mode)
  {
    case EXPOSURE_MODE_DEFLICKER:
      _autoexp_disable(self);
      gtk_stack_set_visible_child_name(GTK_STACK(g->mode_stack), "deflicker");
      break;
    case EXPOSURE_MODE_MANUAL:
    default:
      gtk_stack_set_visible_child_name(GTK_STACK(g->mode_stack), "manual");
      break;
  }

  dt_bauhaus_combobox_set(g->spot_mode, 0);

  dt_gui_update_collapsible_section(&g->cs);
}

void gui_reset(dt_iop_module_t *self)
{
  dt_iop_color_picker_reset(self, TRUE);
}

// --- "Computed EC" deflicker label refresh (scheduled by business) ---

static gboolean _show_computed(gpointer user_data)
{
  dt_iop_module_t *self = user_data;
  dt_iop_exposure_gui_data_t *g = self->gui_data;
  if(!g) return G_SOURCE_REMOVE;

  dt_iop_gui_enter_critical_section(self);
  if(g->state.deflicker_computed_exposure != EXPOSURE_CORRECTION_UNDEFINED)
  {
    gchar *str = g_strdup_printf(_("%.2f EV"), g->state.deflicker_computed_exposure);
    gtk_label_set_text(g->deflicker_used_EC, str);
    g_free(str);
  }
  dt_iop_gui_leave_critical_section(self);

  return G_SOURCE_REMOVE;
}

// --- Cairo draw callbacks for the spot swatches ---

static gboolean _target_color_draw(GtkWidget *widget,
                                   cairo_t *crf,
                                   const dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  int width = allocation.width, height = allocation.height;
  cairo_surface_t *cst = dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
  cairo_t *cr = cairo_create(cst);

  const double INNER_PADDING = 4.0;
  const float margin = 2. * DT_PIXEL_APPLY_DPI(1.5);
  width -= 2 * INNER_PADDING;
  height -= 2 * margin;

  // Ask the business side for the sRGB color of a neutral grey at the
  // current target lightness; we just paint a rectangle with it.
  const float lightness = dt_bauhaus_slider_get(g->lightness_spot);
  float rgb[3];
  dt_iop_exposure_lightness_to_srgb(lightness, rgb);

  cairo_set_source_rgb(cr, rgb[0], rgb[1], rgb[2]);
  cairo_rectangle(cr, INNER_PADDING, margin, width, height);
  cairo_fill(cr);

  cairo_stroke(cr);
  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return TRUE;
}

static gboolean _origin_color_draw(GtkWidget *widget,
                                   cairo_t *crf,
                                   dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;

  GtkAllocation allocation;
  gtk_widget_get_allocation(widget, &allocation);
  int width = allocation.width;
  int height = allocation.height;
  cairo_surface_t *cst = dt_cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
  cairo_t *cr = cairo_create(cst);

  const double INNER_PADDING = 4.0;
  const float margin = 2. * DT_PIXEL_APPLY_DPI(1.5);
  width -= 2 * INNER_PADDING;
  height -= 2 * margin;

  // sRGB is stored in the shared state by
  // dt_iop_exposure_apply_spot_correction(); we just paint it.
  cairo_set_source_rgb(cr,
                       g->state.spot_RGB[0],
                       g->state.spot_RGB[1],
                       g->state.spot_RGB[2]);
  cairo_rectangle(cr, INNER_PADDING, margin, width, height);
  cairo_fill(cr);

  cairo_stroke(cr);
  cairo_destroy(cr);
  cairo_set_source_surface(crf, cst, 0, 0);
  cairo_paint(crf);
  cairo_surface_destroy(cst);
  return TRUE;
}

// --- Lightness slider value-changed callback ---

static void _spot_settings_changed_callback(GtkWidget *slider,
                                            dt_iop_module_t *self)
{
  if(darktable.gui->reset) return;
  dt_iop_exposure_gui_data_t *g = self->gui_data;

  const float lightness = dt_bauhaus_slider_get(g->lightness_spot);

  // Business: persist the new value.
  dt_iop_exposure_save_spot_lightness(lightness);
  g->state.spot_target_lightness = lightness;

  // GUI: repaint the slider's hue stops.
  ++darktable.gui->reset;
  _refresh_lightness_stops(self);
  --darktable.gui->reset;

  // If the picker is in CORRECT mode, reapply the correction with
  // the new target.
  const int mode = dt_bauhaus_combobox_get(g->spot_mode);
  if(mode == DT_SPOT_MODE_CORRECT)
  {
    const dt_iop_exposure_spot_changes_t changes =
      dt_iop_exposure_apply_spot_correction
        (self, darktable.develop->full.pipe,
         DT_IOP_EXPOSURE_SPOT_CORRECT, lightness);
    if(changes & DT_IOP_EXPOSURE_SPOT_CHANGED_EXPOSURE)
      _sync_exposure_widgets(self);
  }
}

// --- Proxy handle_event: external scroll/drag dispatched to a slider ---

static void _exposure_proxy_handle_event(int n_press,
                                         gdouble delta,
                                         GdkModifierType state,
                                         const gboolean is_blackpoint)
{
  const dt_iop_module_t *const self = darktable.develop->proxy.exposure.module;
  if(self && self->gui_data)
  {
    const dt_iop_exposure_params_t *const p = self->params;
    const dt_iop_exposure_gui_data_t *const g = self->gui_data;
    GtkWidget *const widget =
      is_blackpoint
        ? g->black
        : (p->mode == EXPOSURE_MODE_DEFLICKER
             ? g->deflicker_target_level
             : g->exposure);
    const float val = dt_bauhaus_slider_get(widget);
    const float accel = dt_accel_get_speed_multiplier(widget, state);
    if(is_blackpoint)
      delta = -delta;

    if(n_press == 2)
      dt_bauhaus_widget_reset(widget);
    else if(!n_press)
    { // scroll
      const float step = dt_bauhaus_slider_get_step(widget);
      dt_bauhaus_slider_set(widget, val + delta * step * accel);
    }
    else
    { // drag
      const float s_min = dt_bauhaus_slider_get_soft_min(widget);
      const float s_max = dt_bauhaus_slider_get_soft_max(widget);
      dt_bauhaus_slider_set(widget, val + delta * (s_max - s_min) * accel);
    }

    gchar *text = dt_bauhaus_slider_get_text(widget, dt_bauhaus_slider_get(widget));
    dt_action_widget_toast(DT_ACTION(self), widget, "%s", text);
    g_free(text);
  }
}

// --- IOP API: gui_init / gui_cleanup ---

void gui_init(dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = IOP_GUI_ALLOC(exposure);

  g->state.deflicker_histogram = NULL;
  g->state.deflicker_computed_exposure = EXPOSURE_CORRECTION_UNDEFINED;

  g->mode_stack = GTK_STACK(gtk_stack_new());
  gtk_stack_set_homogeneous(GTK_STACK(g->mode_stack), FALSE);

  GtkWidget *vbox_manual = self->widget = dt_gui_vbox();
  gtk_stack_add_named(GTK_STACK(g->mode_stack), vbox_manual, "manual");

  g->compensate_exposure_bias = dt_bauhaus_toggle_from_params
    (self, "compensate_exposure_bias");
  gtk_widget_set_tooltip_text(g->compensate_exposure_bias,
                              _("automatically remove the camera exposure bias\n"
                                "this is useful if you exposed the image to the right."));

  g->compensate_hilite_preserv = dt_bauhaus_toggle_from_params
    (self, "compensate_hilite_pres");
  gtk_widget_set_tooltip_text(g->compensate_hilite_preserv,
                              _("remove the camera's hidden exposure bias in\n"
                                "HDR / highlight preservation / dynamic range / HLG tone mode.\n"
                                "\n"
                                "when enabled on an image with nonzero bias, tone mapping\n"
                                "(e.g. sigmoid) is required to avoid blown-out highlights."));

  g->exposure = dt_color_picker_new(self, DT_COLOR_PICKER_AREA,
                                    dt_bauhaus_slider_from_params(self, N_("exposure")));
  gtk_widget_set_tooltip_text(g->exposure, _("adjust the exposure correction"));
  dt_bauhaus_slider_set_digits(g->exposure, 3);
  dt_bauhaus_slider_set_format(g->exposure, _(" EV"));
  dt_bauhaus_slider_set_soft_range(g->exposure, -3.0, 4.0);
  dt_bauhaus_widget_set_quad_tooltip(g->exposure, _("set the exposure adjustment using the selected area"));
  dt_shortcut_register(dt_action_widget(g->exposure), 0, 0, GDK_KEY_e, 0);

  GtkWidget *vbox_deflicker = self->widget = dt_gui_vbox();
  gtk_stack_add_named(GTK_STACK(g->mode_stack), vbox_deflicker, "deflicker");

  g->deflicker_percentile = dt_bauhaus_slider_from_params(self, "deflicker_percentile");
  dt_bauhaus_slider_set_format(g->deflicker_percentile, "%");
  gtk_widget_set_tooltip_text
    (g->deflicker_percentile,
     // xgettext:no-c-format
     _("where in the histogram to meter for deflicking. E.g. 50% is median"));

  g->deflicker_target_level = dt_bauhaus_slider_from_params(self, "deflicker_target_level");
  dt_bauhaus_slider_set_format(g->deflicker_target_level, _(" EV"));
  gtk_widget_set_tooltip_text
    (g->deflicker_target_level,
     _("where to place the exposure level for processed pics, EV below overexposure."));

  g->deflicker_used_EC = GTK_LABEL(dt_ui_label_new("")); // This gets filled in by process
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->deflicker_used_EC),
                              _("what exposure correction has actually been used"));

  dt_gui_box_add(vbox_deflicker, dt_gui_hbox(dt_ui_label_new(_("computed EC: ")),
                                             g->deflicker_used_EC));

  // Start building top level widget
  self->widget = dt_gui_vbox();

  g->mode = dt_bauhaus_combobox_from_params(self, N_("mode"));

  dt_gui_box_add(self->widget, g->mode_stack);

  g->black = dt_bauhaus_slider_from_params(self, "black");
  gtk_widget_set_tooltip_text
    (g->black,
     _("adjust the black level to unclip negative RGB values.\n"
       "you should never use it to add more density in blacks!\n"
       "if poorly set, it will clip near-black colors out of gamut\n"
       "by pushing RGB values into negatives."));
  dt_bauhaus_slider_set_digits(g->black, 4);
  dt_bauhaus_slider_set_soft_range(g->black, -0.1, 0.1);

  dt_gui_new_collapsible_section
    (&g->cs,
     "plugins/darkroom/exposure/mapping",
     _("area exposure mapping"),
     GTK_BOX(self->widget),
     DT_ACTION(self));

  gtk_widget_set_tooltip_text
    (g->cs.expander,
     _("define a target brightness, in terms of exposure,\n"
       "for a selected region of the image (the control sample),\n"
       "which you then match against the same target brightness\n"
       "in other images. the control sample can either\n"
       "be a critical part of your subject or a non-moving and\n"
       "consistently-lit surface over your series of images."));

  DT_BAUHAUS_COMBOBOX_NEW_FULL
    (g->spot_mode, self, NULL, N_("area mode"),
     _("\"correction\" automatically adjust exposure\n"
       "such that the input lightness is mapped to the target.\n"
       "\"measure\" simply shows how an input color is mapped by\n"
       "the exposure compensation and can be used to define a target."),
     0, _spot_settings_changed_callback, self,
     N_("correction"),
     N_("measure"));

  g->origin_spot = GTK_WIDGET(gtk_drawing_area_new());
  gtk_widget_set_vexpand(g->origin_spot, TRUE);
  gtk_widget_set_size_request(g->origin_spot,
                              2 * DT_PIXEL_APPLY_DPI(darktable.bauhaus->quad_width),
                              DT_PIXEL_APPLY_DPI(darktable.bauhaus->quad_width));
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->origin_spot),
                              _("the input color that should be mapped to the target"));

  g_signal_connect(G_OBJECT(g->origin_spot), "draw", G_CALLBACK(_origin_color_draw), self);

  g->Lch_origin = gtk_label_new(_("L : \tN/A"));
  gtk_widget_set_tooltip_text
    (GTK_WIDGET(g->Lch_origin),
     _("these LCh coordinates are computed from CIE Lab 1976 coordinates"));
  g->target_spot = GTK_WIDGET(gtk_drawing_area_new());
  gtk_widget_set_vexpand(g->target_spot, TRUE);
  gtk_widget_set_size_request(g->target_spot,
                              2 * DT_PIXEL_APPLY_DPI(darktable.bauhaus->quad_width),
                              DT_PIXEL_APPLY_DPI(darktable.bauhaus->quad_width));
  gtk_widget_set_tooltip_text(GTK_WIDGET(g->target_spot),
                              _("the desired target exposure after mapping"));

  g_signal_connect(G_OBJECT(g->target_spot), "draw", G_CALLBACK(_target_color_draw), self);

  g->lightness_spot = dt_bauhaus_slider_new_with_range(self, 0., 100., 0, 50.f, 1);
  dt_bauhaus_widget_set_label(g->lightness_spot, NULL, N_("lightness"));
  dt_bauhaus_slider_set_format(g->lightness_spot, "%");
  g_signal_connect(G_OBJECT(g->lightness_spot), "value-changed",
                   G_CALLBACK(_spot_settings_changed_callback), self);

  dt_gui_box_add(g->cs.container,
                 g->spot_mode,
                 dt_gui_hbox( dt_gui_vbox(dt_ui_section_label_new(C_("section", "input")),
                                          g->origin_spot, g->Lch_origin),
                              gtk_label_new("   "), dt_gui_expand( // spacer
                              dt_gui_vbox(dt_ui_section_label_new(C_("section", "target")),
                                          g->target_spot, g->lightness_spot))));

  // Wire the proxy callbacks: getters live in business; the
  // handle_event slider manipulator lives here in the GUI side.
  dt_dev_proxy_exposure_t *instance = &darktable.develop->proxy.exposure;
  instance->module = self;
  instance->get_exposure = dt_iop_exposure_proxy_get_exposure;
  instance->get_effective_exposure = dt_iop_exposure_proxy_get_effective_exposure;
  instance->get_black = dt_iop_exposure_proxy_get_black;
  instance->handle_event = _exposure_proxy_handle_event;
}

void gui_cleanup(dt_iop_module_t *self)
{
  dt_iop_exposure_gui_data_t *g = self->gui_data;

  if(darktable.develop->proxy.exposure.module == self)
    darktable.develop->proxy.exposure.module = NULL;

  // The histogram lives in the gui state and was allocated by the
  // business side; release it on widget teardown.
  dt_free_align(g->state.deflicker_histogram);
  g->state.deflicker_histogram = NULL;

  g_idle_remove_by_data(self);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
