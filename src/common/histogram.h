/*
    This file is part of darktable,
    Copyright (C) 2014-2023 darktable developers.

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

#include <stdint.h>

#include "develop/imageop.h"
#include "develop/pixelpipe.h"
#include "common/iop_profile.h"

/*
 * histogram region of interest
 *
 * image is located in (0,     0)      .. (width,           height)
 * but only            (crop_x,crop_y) .. (width-crop_width,height-crop_height)
 * will be sampled
 */
typedef struct dt_histogram_roi_t
{
  int width, height, crop_x, crop_y, crop_right, crop_bottom;
} dt_histogram_roi_t;

// allocates an aligned histogram buffer if needed, callers
// (pixelpipe, exposure, global histogram) must garbage collect this
// buffer via dt_free_align()
void dt_histogram_helper(dt_dev_histogram_collection_params_t *histogram_params,
                         dt_dev_histogram_stats_t *histogram_stats,
                         const dt_iop_colorspace_type_t cst,
                         const dt_iop_colorspace_type_t cst_to,
                         const void *pixel,
                         uint32_t **histogram, uint32_t *histogram_max,
                         const gboolean compensate_middle_grey,
                         const dt_iop_order_iccprofile_info_t *const profile_info);

#ifdef HAVE_VULKAN
/** Vulkan histogram reduction (DAG milestone M5,
 *  gpu_resident_pixelpipe_dag.md §5.4). Bins the device image
 *  `dev_in` (float4, width×height, row-major) into `histogram`
 *  (bins_count·4 uint32, layout hist[bin*4+k]) on the GPU with atomic
 *  increments, matching the CPU `dt_histogram_helper` for the RGB /
 *  Lab / Lab→LCh binnings. The caller supplies the pre-zeroed host
 *  `histogram` (it is (re)zeroed on device internally). Returns FALSE
 *  — leaving the caller to fall back to the CPU reducer — for the
 *  unported cases: RAW (uint16), and middle-grey-compensated RGB
 *  (needs the profile TRC). Caller holds the device lock. Under an
 *  active capture the dispatch is captured and the small readback
 *  flushes the span, exactly like any other sync tap. */
gboolean dt_histogram_helper_vk(int devid,
                                dt_vk_mem_t *dev_in,
                                int width,
                                int height,
                                const dt_histogram_roi_t *roi,
                                int bins_count,
                                dt_iop_colorspace_type_t cst,
                                dt_iop_colorspace_type_t cst_to,
                                gboolean compensate_middle_grey,
                                uint32_t *histogram);

/** Collect-level Vulkan sibling of dt_histogram_helper: same
 *  (re)allocation contract for *histogram, same stats and channel-max
 *  computation, but the reduction runs on the device image via
 *  dt_histogram_helper_vk (taking the device lock internally). Returns
 *  FALSE — with *histogram/stats untouched apart from a possible
 *  buffer grow — when the case isn't ported (RAW, compensated RGB, a
 *  module roi whose dimensions differ from the device buffer) so the
 *  caller falls back to the CPU collection. Used by the pixelpipe's
 *  §5.3.1 site-2 histogram tap: a few-KB histogram readback instead of
 *  materializing the whole trunk to host. */
gboolean dt_histogram_helper_vk_collect(dt_dev_histogram_collection_params_t *histogram_params,
                                        dt_dev_histogram_stats_t *histogram_stats,
                                        const dt_iop_colorspace_type_t cst,
                                        const dt_iop_colorspace_type_t cst_to,
                                        dt_vk_mem_t *dev_in,
                                        const int width,
                                        const int height,
                                        uint32_t **histogram,
                                        uint32_t *histogram_max,
                                        const gboolean compensate_middle_grey);
#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
