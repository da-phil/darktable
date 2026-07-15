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
/*
 * cmocka tests for the whole-pipe output-tiling planner
 * (dt_dev_pixelpipe_plan_vk_tiles, ME.1/ME.2 of
 * dev-doc/gpu_resident_pixelpipe_dag.md §5.10.1). Pure geometry — no
 * Vulkan device is needed, so these run everywhere.
 */
#include <limits.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <cmocka.h>

#include "develop/pixelpipe_hb.h"

// Bytes one tile's trunk buffer needs, incl. halo (double math avoids
// intermediate overflow).
static size_t _tile_bytes(const dt_dev_vk_tile_plan_t *p, const size_t bpp,
                          const float factor, const float maxbuf)
{
  const double w = p->tile_w + 2 * p->overlap;
  const double h = p->tile_h + 2 * p->overlap;
  return (size_t)(w * h * (double)bpp * (double)factor * (double)maxbuf);
}

// A buffer well under budget stays a single span (no tiling).
static void test_fits_single_span(void **state)
{
  (void)state;
  // 1 MP @ 16 bpp = 16 MB, budget 385 MB
  const dt_dev_vk_tile_plan_t p =
      dt_dev_pixelpipe_plan_vk_tiles(1024, 1024, 16, 1.0f, 1.0f, 0, 1,
                                     (size_t)385u << 20);
  assert_true(p.tileable);
  assert_int_equal(p.tiles_x, 1);
  assert_int_equal(p.tiles_y, 1);
  assert_int_equal(p.tile_w, 1024);
  assert_int_equal(p.tile_h, 1024);
}

// The export case from the field log: ~54 MP (865 MB) with a 385 MB
// budget must split, cover the image, and keep every tile under budget.
static void test_oversized_export_tiles(void **state)
{
  (void)state;
  const int W = 9000, H = 6008;
  const size_t bpp = 16, budget = (size_t)385u << 20;
  const dt_dev_vk_tile_plan_t p =
      dt_dev_pixelpipe_plan_vk_tiles(W, H, bpp, 1.0f, 1.0f, 0, 1, budget);
  assert_true(p.tileable);
  assert_true(p.tiles_x * p.tiles_y >= 2);
  assert_true(p.tiles_x * p.tile_w >= W);       // cores cover the image
  assert_true(p.tiles_y * p.tile_h >= H);
  assert_true(_tile_bytes(&p, bpp, 1.0f, 1.0f) <= budget);
}

// A higher live-set factor forces tiling where factor 1 would not.
static void test_factor_forces_tiles(void **state)
{
  (void)state;
  const int W = 2048, H = 2048;                 // 64 MB @ 16 bpp
  const size_t bpp = 16, budget = (size_t)128u << 20;
  const dt_dev_vk_tile_plan_t p1 =
      dt_dev_pixelpipe_plan_vk_tiles(W, H, bpp, 1.0f, 1.0f, 0, 1, budget);
  assert_int_equal(p1.tiles_x * p1.tiles_y, 1); // 64 MB < 128 MB
  const dt_dev_vk_tile_plan_t p4 =
      dt_dev_pixelpipe_plan_vk_tiles(W, H, bpp, 4.0f, 1.0f, 0, 1, budget);
  assert_true(p4.tileable);                      // 256 MB > 128 MB
  assert_true(p4.tiles_x * p4.tiles_y >= 2);
  assert_true(_tile_bytes(&p4, bpp, 4.0f, 1.0f) <= budget);
}

// Tiled dimensions honour the module alignment requirement.
static void test_alignment_respected(void **state)
{
  (void)state;
  const size_t bpp = 16, budget = (size_t)64u << 20;
  const unsigned int align = 32;
  const dt_dev_vk_tile_plan_t p =
      dt_dev_pixelpipe_plan_vk_tiles(8000, 8000, bpp, 1.0f, 1.0f, 0, align, budget);
  assert_true(p.tileable);
  assert_int_equal((p.tile_w + 2 * p.overlap) % (int)align, 0);
  assert_int_equal((p.tile_h + 2 * p.overlap) % (int)align, 0);
}

// With a real halo the cores still cover the image and the haloed tile
// stays under budget; the overlap is rounded up to the alignment.
static void test_overlap_covers_and_fits(void **state)
{
  (void)state;
  const int W = 6000, H = 4000;
  const size_t bpp = 16, budget = (size_t)48u << 20;
  const int overlap = 16;
  const dt_dev_vk_tile_plan_t p =
      dt_dev_pixelpipe_plan_vk_tiles(W, H, bpp, 1.5f, 1.0f, overlap, 8, budget);
  assert_true(p.tileable);
  assert_true(p.overlap >= overlap);
  assert_int_equal(p.overlap % 8, 0);
  assert_true(p.tiles_x * p.tile_w >= W);
  assert_true(p.tiles_y * p.tile_h >= H);
  assert_true(_tile_bytes(&p, bpp, 1.5f, 1.0f) <= budget);
}

// Degenerate inputs are reported as not tileable (caller declines).
static void test_degenerate_not_tileable(void **state)
{
  (void)state;
  assert_false(dt_dev_pixelpipe_plan_vk_tiles(0, 100, 16, 1.0f, 1.0f, 0, 1,
                                              (size_t)1u << 20).tileable);
  assert_false(dt_dev_pixelpipe_plan_vk_tiles(100, 100, 16, 1.0f, 1.0f, 0, 1,
                                              0).tileable);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_fits_single_span),
    cmocka_unit_test(test_oversized_export_tiles),
    cmocka_unit_test(test_factor_forces_tiles),
    cmocka_unit_test(test_alignment_respected),
    cmocka_unit_test(test_overlap_covers_and_fits),
    cmocka_unit_test(test_degenerate_not_tileable),
  };
  return cmocka_run_group_tests(tests, NULL, NULL);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
