/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

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
#include <glib.h>

#define DT_HDR_ALIGN_MESH_COLS 3
#define DT_HDR_ALIGN_MESH_ROWS 3
#define DT_HDR_ALIGN_MESH_NODES (DT_HDR_ALIGN_MESH_COLS * DT_HDR_ALIGN_MESH_ROWS)

typedef struct dt_hdr_alignment_t
{
    float H[8];
    /* Residual output-space mesh shifts in full-resolution pixels.
     * Nodes are stored in row-major order over a DT_HDR_ALIGN_MESH_ROWS x
     * DT_HDR_ALIGN_MESH_COLS regular grid spanning the frame. */
    float mesh_dx[DT_HDR_ALIGN_MESH_NODES];
    float mesh_dy[DT_HDR_ALIGN_MESH_NODES];
} dt_hdr_alignment_t;

/** Compute an 8-DOF projective homography between a reference and candidate
 *  raw mosaic image using multi-resolution ECC refinement.
 *
 *  Both images must be single-channel float Bayer mosaic data of identical
 *  dimensions (as produced by the rawprepare IOP).
 *
 *  @param ref_mosaic  Reference image (first exposure), single-channel float
 *  @param img_mosaic  Candidate image to align, single-channel float
 *  @param wd          Image width in pixels
 *  @param ht          Image height in pixels
 *  @param out_align   [out] Full-resolution backward homography plus a
 *                     residual regular mesh in output coordinates.
 *  @return TRUE on success, FALSE on failure (e.g. allocation error)
 */
gboolean dt_hdr_align_compute(const float *ref_mosaic,
                              const float *img_mosaic,
                              const int wd,
                              const int ht,
                              dt_hdr_alignment_t *out_align);

/** Apply alignment transformation to a Bayer mosaic image using
 *  CFA-aware per-color-plane warping.
 *
 *  Decomposes the mosaic into 4 half-resolution color planes, warps each
 *  independently using the given homography, then recomposes.
 *  Pixels that fall outside the source image bounds are set to 0.0f.
 *
 *  @param in_mosaic   Input mosaic image, single-channel float
 *  @param out_mosaic  Output aligned mosaic image (pre-allocated, same size as input)
 *  @param wd          Image width in pixels (must be even)
 *  @param ht          Image height in pixels (must be even)
 *  @param filters     Bayer filter pattern (dcraw-style packed uint32_t)
 *  @param align       Full-resolution homography plus residual mesh
 */
void dt_hdr_align_apply(const float *in_mosaic,
                        float *out_mosaic,
                        const int wd,
                        const int ht,
                        const uint32_t filters,
                        const dt_hdr_alignment_t *align);

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
