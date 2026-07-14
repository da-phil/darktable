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
#include "common/iop_profile.h"
#include "common/colorspaces.h"
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

// Run the GPU picker over `box` in the given colorspace pair, comparing
// against the CPU picker. Runs the device sequence, releases the lock,
// then asserts. `exact_extrema` selects whether min/max must match
// exactly (no-conversion path) or within a tolerance (converting paths,
// where the GPU/CPU conversions differ by a few ulp so a different
// pixel may hold the extremum).
static void run_box_prof(const float *pixels, const int *box,
                         const dt_iop_colorspace_type_t image_cst,
                         const dt_iop_colorspace_type_t picker_cst,
                         const dt_iop_order_iccprofile_info_t *profile,
                         const gboolean exact_extrema, const char *tag)
{
  const dt_iop_roi_t roi = { 0, 0, TW, TH, 1.0f };
  dt_iop_buffer_dsc_t dsc;
  memset(&dsc, 0, sizeof(dsc));
  dsc.channels = 4u;
  dsc.filters = 0u;
  dsc.datatype = TYPE_FLOAT;

  lib_colorpicker_stats cpu, gpu;
  dt_color_picker_helper(&dsc, pixels, &roi, box, FALSE, cpu,
                         image_cst, picker_cst, profile);

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
                                    image_cst, picker_cst, profile);
  if(din) dt_vulkan_free_buffer(dev, din);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  assert_int_equal(rc, 0);
  assert_true(ran);
  for(int c = 0; c < 4; c++)
  {
    const float scale = fmaxf(1.0f, fabsf(cpu[DT_PICK_MEAN][c]));
    const float ext_eps = exact_extrema ? 0.0f : 5e-4f * scale;
    if(fabsf(gpu[DT_PICK_MIN][c] - cpu[DT_PICK_MIN][c]) > ext_eps)
      fail_msg("%s: min ch%d gpu %g vs cpu %g", tag, c,
               (double)gpu[DT_PICK_MIN][c], (double)cpu[DT_PICK_MIN][c]);
    if(fabsf(gpu[DT_PICK_MAX][c] - cpu[DT_PICK_MAX][c]) > ext_eps)
      fail_msg("%s: max ch%d gpu %g vs cpu %g", tag, c,
               (double)gpu[DT_PICK_MAX][c], (double)cpu[DT_PICK_MAX][c]);
    // mean: summation order differs, scale-aware tolerance
    const float eps = (exact_extrema ? 1e-4f : 5e-4f) * scale;
    if(fabsf(gpu[DT_PICK_MEAN][c] - cpu[DT_PICK_MEAN][c]) > eps)
      fail_msg("%s: mean ch%d gpu %g vs cpu %g (eps %g)",
               tag, c, (double)gpu[DT_PICK_MEAN][c],
               (double)cpu[DT_PICK_MEAN][c], (double)eps);
  }
}

static void run_box_cst(const float *pixels, const int *box,
                        const dt_iop_colorspace_type_t image_cst,
                        const dt_iop_colorspace_type_t picker_cst,
                        const gboolean exact_extrema, const char *tag)
{
  run_box_prof(pixels, box, image_cst, picker_cst, NULL, exact_extrema, tag);
}

static void run_box(const float *pixels, const int *box, const char *tag)
{
  run_box_cst(pixels, box, IOP_CS_RGB, IOP_CS_RGB, TRUE, tag);
}

// sRGB D50 matrices matching dt_vulkan_common.h's vk_sRGB_to_XYZ /
// vk_XYZ_to_sRGB; linear TRC. Both matrix and its transpose filled (the
// CPU path reads matrix_in_transposed).
static void make_srgb_profile(dt_iop_order_iccprofile_info_t *p)
{
  static const float SRGB_TO_XYZ[9] = {
    0.4360747f, 0.3850649f, 0.1430804f,
    0.2225045f, 0.7168786f, 0.0606169f,
    0.0139322f, 0.0971045f, 0.7141733f };
  static const float XYZ_TO_SRGB[9] = {
     3.1338561f, -1.6168667f, -0.4906146f,
    -0.9787684f,  1.9161415f,  0.0334540f,
     0.0719453f, -0.2289914f,  1.4052427f };
  dt_ioppr_init_profile_info(p, 0);
  p->type = DT_COLORSPACE_SRGB;
  g_strlcpy(p->filename, "srgb-test", sizeof(p->filename));
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
    {
      p->matrix_in[i][j]  = SRGB_TO_XYZ[i * 3 + j];
      p->matrix_out[i][j] = XYZ_TO_SRGB[i * 3 + j];
      p->matrix_in_transposed[j][i]  = SRGB_TO_XYZ[i * 3 + j];
      p->matrix_out_transposed[j][i] = XYZ_TO_SRGB[i * 3 + j];
    }
  p->nonlinearlut = 0;
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

static void test_hsl_picker(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0x44;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  // RGB image, HSL picker: keep values in gamut so the HSL conversion
  // is well-conditioned
  for(size_t i = 0; i < TN; i++)
  {
    p[i*4+0] = frand(0.0f, 1.0f); p[i*4+1] = frand(0.0f, 1.0f);
    p[i*4+2] = frand(0.0f, 1.0f); p[i*4+3] = 1.0f;
  }
  const int box[4] = { 4, 6, 4 + 50, 6 + 40 };
  run_box_cst(p, box, IOP_CS_RGB, IOP_CS_HSL, FALSE, "hsl picker");
  dt_free_align(p);
}

static void test_lch_picker(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0x55;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  // Lab image, LCH picker
  for(size_t i = 0; i < TN; i++)
  {
    p[i*4+0] = frand(0.0f, 100.0f); p[i*4+1] = frand(-90.0f, 90.0f);
    p[i*4+2] = frand(-90.0f, 90.0f); p[i*4+3] = 1.0f;
  }
  const int box[4] = { 4, 6, 4 + 50, 6 + 40 };
  run_box_cst(p, box, IOP_CS_LAB, IOP_CS_LCH, FALSE, "lch picker");
  dt_free_align(p);
}

static void test_jzczhz_picker(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0x66;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  // scene-referred RGB in a sane range for the JzAzBz chain
  for(size_t i = 0; i < TN; i++)
  {
    p[i*4+0] = frand(0.0f, 1.0f); p[i*4+1] = frand(0.0f, 1.0f);
    p[i*4+2] = frand(0.0f, 1.0f); p[i*4+3] = 1.0f;
  }
  dt_iop_order_iccprofile_info_t prof;
  make_srgb_profile(&prof);
  const int box[4] = { 3, 4, 3 + 55, 4 + 45 };
  run_box_prof(p, box, IOP_CS_RGB, IOP_CS_JZCZHZ, &prof, FALSE, "jzczhz picker");
  dt_ioppr_cleanup_profile_info(&prof);
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
  gboolean denoise_g = TRUE, jzczhz_g = TRUE, badbox_g = TRUE, oob_g = TRUE;
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
                                          IOP_CS_RGB, IOP_CS_RGB, NULL);
    // JzCzhz picker WITHOUT a profile -> refuse (needs the matrix)
    jzczhz_g = dt_color_picker_helper_vk(dev, din, TW, TH, box, FALSE, out,
                                         IOP_CS_RGB, IOP_CS_JZCZHZ, NULL);
    // empty/inverted box
    badbox_g = dt_color_picker_helper_vk(dev, din, TW, TH, bad_box, FALSE, out,
                                         IOP_CS_RGB, IOP_CS_RGB, NULL);
    // out-of-bounds box
    const int oob[4] = { 0, 0, TW + 4, TH };
    oob_g = dt_color_picker_helper_vk(dev, din, TW, TH, oob, FALSE, out,
                                      IOP_CS_RGB, IOP_CS_RGB, NULL);
  }
  if(din) dt_vulkan_free_buffer(dev, din);
  if(dev == 0) dt_vulkan_unlock_device(dev);
  dt_free_align(p);

  assert_int_equal(rc, 0);
  assert_false(denoise_g);
  assert_false(jzczhz_g);
  assert_false(badbox_g);
  assert_false(oob_g);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_full_box),
    cmocka_unit_test(test_sub_box),
    cmocka_unit_test(test_single_pixel_box),
    cmocka_unit_test(test_hsl_picker),
    cmocka_unit_test(test_lch_picker),
    cmocka_unit_test(test_jzczhz_picker),
    cmocka_unit_test(test_subset_gates),
  };
  return cmocka_run_group_tests(tests, group_setup, group_teardown);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
