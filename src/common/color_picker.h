/*
    This file is part of darktable,
    Copyright (C) 2016-2026 darktable developers.

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

#include "common/iop_profile.h"
#include "libs/colorpicker.h"

struct dt_iop_buffer_dsc_t;
struct dt_iop_roi_t;
enum dt_iop_colorspace_type_t;

typedef enum dt_pixelpipe_picker_source_t
{
  PIXELPIPE_PICKER_INPUT = 0,
  PIXELPIPE_PICKER_OUTPUT = 1
} dt_pixelpipe_picker_source_t;

void dt_color_picker_backtransform_box(dt_develop_t *dev,
                                       const int num,
                                       const float *in,
                                       float *out);
void dt_color_picker_transform_box(dt_develop_t *dev,
                                   const int num,
                                   const float *in,
                                   float *out,
                                   gboolean scale);
gboolean dt_color_picker_box(dt_iop_module_t *module,
                             const dt_iop_roi_t *roi,
                             const dt_colorpicker_sample_t *const sample,
                             dt_pixelpipe_picker_source_t picker_source,
                             int *box);
void dt_color_picker_helper(const struct dt_iop_buffer_dsc_t *dsc,
                            const float *const pixel,
                            const struct dt_iop_roi_t *roi,
                            const int *const box,
                            const gboolean denoise,
                            lib_colorpicker_stats pick,
                            const enum dt_iop_colorspace_type_t image_cst,
                            const enum dt_iop_colorspace_type_t picker_cst,
                            const dt_iop_order_iccprofile_info_t *const profile);

#ifdef HAVE_VULKAN
/** Vulkan color-picker reduction (DAG milestone M5,
 *  gpu_resident_pixelpipe_dag.md §5.4). Reduces the box of the device
 *  image `dev_in` (float4, width×height) to per-channel mean/min/max
 *  in `pick`, on the GPU with float atomics, matching the CPU
 *  `dt_color_picker_helper` for the no-conversion picker path
 *  (image colorspace already equals the picker colorspace). Returns
 *  FALSE — leaving the caller on the CPU reducer — for the unported
 *  cases: denoise (needs the blur), a colorspace-converting picker
 *  (LCH/HSL/JzCzhz), or an out-of-bounds/empty box. `box` is the
 *  half-open [x0,y0,x1,y1] region. Caller holds the device lock. */
gboolean dt_color_picker_helper_vk(int devid,
                                   dt_vk_mem_t *dev_in,
                                   int width,
                                   int height,
                                   const int *const box,
                                   const gboolean denoise,
                                   lib_colorpicker_stats pick,
                                   const enum dt_iop_colorspace_type_t image_cst,
                                   const enum dt_iop_colorspace_type_t picker_cst);
#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
