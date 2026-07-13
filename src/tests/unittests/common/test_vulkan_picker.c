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
 * cmocka tests for the Vulkan color-picker reduction kernel (DAG
 * milestone M5, dev-doc/gpu_resident_pixelpipe_dag.md §5.4).
 *
 * dt_color_picker_helper_vk reduces the picker box to per-channel
 * mean/min/max with GPU float atomics; this validates it against
 * darktable's own CPU dt_color_picker_helper for the no-conversion
 * path. min/max are order-independent, so they must match exactly;
 * the mean is a sum/count whose accumulation order differs between the
 * OMP reduction and the atomic path, so it is compared with a
 * scale-aware tolerance.
 *
 * Runs on any Vulkan implementation (lavapipe in CI); skips cleanly
 * when no device is available.
 */
#include <float.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cmocka.h>

#include "common/darktable.h"
#include "common/vulkan.h"
#include "common/color_picker.h"
#include "control/conf.h"
#include "develop/format.h"
#include "develop/imageop.h"

#ifndef HAVE_VULKAN
#error "test_vulkan_picker requires USE_VULKAN builds"
#endif

#define TW 71
#define TH 59
#define TN ((size_t)TW * TH)

static dt_vulkan_t s_vk;
static dt_conf_t s_conf;
static gboolean s_have_device = FALSE;

#define REQUIRE_DEVICE() do { if(!s_have_device) skip(); } while(0)

static int group_setup(void **state)
{
  (void)state;
  darktable.conf = &s_conf;
  dt_conf_init(darktable.conf, "/nonexistent-dt-vk-picker-test.rc", FALSE, NULL);
  dt_conf_set_bool("opencl_use_vulkan", TRUE);
  darktable.datadir = g_strdup(TEST_VK_DATADIR);

  darktable.vulkan = &s_vk;
  dt_vulkan_init(&s_vk);
  if(!dt_vulkan_running())
  {
    fprintf(stderr, "no Vulkan device available — picker tests will be skipped\n");
    return 0;
  }
  s_have_device = TRUE;
  return 0;
}

static int group_teardown(void **state)
{
  (void)state;
  dt_vulkan_cleanup(&s_vk);
  darktable.vulkan = NULL;
  return 0;
}

static uint32_t s_rng;
static float frand(float lo, float hi)
{
  s_rng = s_rng * 1664525u + 1013904223u;
  return lo + (hi - lo) * ((s_rng >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
}

// Run the GPU picker over `box`, comparing against the CPU picker.
// Runs the device sequence, releases the lock, then asserts.
static void run_box(const float *pixels, const int *box, const char *tag)
{
  const dt_iop_roi_t roi = { 0, 0, TW, TH, 1.0f };
  dt_iop_buffer_dsc_t dsc;
  memset(&dsc, 0, sizeof(dsc));
  dsc.channels = 4u;
  dsc.filters = 0u;
  dsc.datatype = TYPE_FLOAT;

  lib_colorpicker_stats cpu, gpu;
  dt_color_picker_helper(&dsc, pixels, &roi, box, FALSE, cpu,
                         IOP_CS_RGB, IOP_CS_RGB, NULL);

  const size_t bytes = TN * 4 * sizeof(float);
  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  gboolean ran = FALSE;
  dt_vk_mem_t *din = NULL;
  if(!rc)
  {
    din = dt_vulkan_alloc_buffer(dev, bytes);
    if(!din) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, din, pixels, bytes);
  if(!rc)
    ran = dt_color_picker_helper_vk(dev, din, TW, TH, box, FALSE, gpu,
                                    IOP_CS_RGB, IOP_CS_RGB);
  if(din) dt_vulkan_free_buffer(dev, din);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  assert_int_equal(rc, 0);
  assert_true(ran);
  for(int c = 0; c < 4; c++)
  {
    // min/max are order-independent -> exact
    assert_float_equal(gpu[DT_PICK_MIN][c], cpu[DT_PICK_MIN][c], 0.0f);
    assert_float_equal(gpu[DT_PICK_MAX][c], cpu[DT_PICK_MAX][c], 0.0f);
    // mean: summation order differs, scale-aware tolerance
    const float eps = 1e-4f * fmaxf(1.0f, fabsf(cpu[DT_PICK_MEAN][c]));
    if(fabsf(gpu[DT_PICK_MEAN][c] - cpu[DT_PICK_MEAN][c]) > eps)
      fail_msg("%s: mean ch%d gpu %g vs cpu %g (eps %g)",
               tag, c, (double)gpu[DT_PICK_MEAN][c],
               (double)cpu[DT_PICK_MEAN][c], (double)eps);
  }
}

static void fill(float *p)
{
  for(size_t i = 0; i < TN; i++)
  {
    p[i*4+0] = frand(-0.2f, 1.2f); p[i*4+1] = frand(-0.2f, 1.2f);
    p[i*4+2] = frand(-0.2f, 1.2f); p[i*4+3] = frand(0.0f, 1.0f);
  }
}

static void test_full_box(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0x11;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  fill(p);
  const int box[4] = { 0, 0, TW, TH };
  run_box(p, box, "full box");
  dt_free_align(p);
}

static void test_sub_box(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0x22;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  fill(p);
  const int box[4] = { 7, 5, 7 + 40, 5 + 33 };
  run_box(p, box, "sub box");
  dt_free_align(p);
}

static void test_single_pixel_box(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0x33;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  fill(p);
  const int box[4] = { 20, 30, 21, 31 }; // 1x1: mean == min == max
  run_box(p, box, "1x1 box");
  dt_free_align(p);
}

static void test_subset_gates(void **state)
{
  (void)state; REQUIRE_DEVICE();
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  fill(p);
  lib_colorpicker_stats out;
  const int box[4] = { 0, 0, TW, TH };
  const int bad_box[4] = { 10, 10, 5, 5 }; // empty/inverted

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  gboolean denoise_g = TRUE, convert_g = TRUE, badbox_g = TRUE, oob_g = TRUE;
  dt_vk_mem_t *din = NULL;
  if(!rc)
  {
    din = dt_vulkan_alloc_buffer(dev, TN * 4 * sizeof(float));
    if(!din) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, din, p, TN * 4 * sizeof(float));
  if(!rc)
  {
    // denoise not ported
    denoise_g = dt_color_picker_helper_vk(dev, din, TW, TH, box, TRUE, out,
                                          IOP_CS_RGB, IOP_CS_RGB);
    // colorspace-converting picker not ported (RGB image, HSL picker)
    convert_g = dt_color_picker_helper_vk(dev, din, TW, TH, box, FALSE, out,
                                          IOP_CS_RGB, IOP_CS_HSL);
    // empty/inverted box
    badbox_g = dt_color_picker_helper_vk(dev, din, TW, TH, bad_box, FALSE, out,
                                         IOP_CS_RGB, IOP_CS_RGB);
    // out-of-bounds box
    const int oob[4] = { 0, 0, TW + 4, TH };
    oob_g = dt_color_picker_helper_vk(dev, din, TW, TH, oob, FALSE, out,
                                      IOP_CS_RGB, IOP_CS_RGB);
  }
  if(din) dt_vulkan_free_buffer(dev, din);
  if(dev == 0) dt_vulkan_unlock_device(dev);
  dt_free_align(p);

  assert_int_equal(rc, 0);
  assert_false(denoise_g);
  assert_false(convert_g);
  assert_false(badbox_g);
  assert_false(oob_g);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_full_box),
    cmocka_unit_test(test_sub_box),
    cmocka_unit_test(test_single_pixel_box),
    cmocka_unit_test(test_subset_gates),
  };
  return cmocka_run_group_tests(tests, group_setup, group_teardown);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
