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
 * cmocka tests for the Vulkan histogram reduction kernel (DAG
 * milestone M5, dev-doc/gpu_resident_pixelpipe_dag.md §5.4).
 *
 * dt_histogram_helper_vk bins a device image with GPU atomics; this
 * validates it against darktable's own CPU dt_histogram_helper for the
 * three ported binnings (RGB, Lab, Lab->LCh). Two comparison regimes:
 *
 *  - bin-centre inputs (values placed at (bin+0.5)/max so the float
 *    product lands mid-bin): the integer bin indices are then immune
 *    to CPU-vs-GPU float rounding, so the histograms must match
 *    *exactly*.
 *  - random inputs: near a bin boundary a few pixels may quantise to
 *    an adjacent bin between the two float paths, so we bound the
 *    total variation (sum of |Δcount|) to a small fraction of the
 *    pixel count rather than demand exact equality.
 *
 * Runs on any Vulkan implementation (lavapipe in CI); all tests skip
 * cleanly when no device is available.
 */
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
#include "common/histogram.h"
#include "control/conf.h"
#include "develop/imageop.h"

#ifndef HAVE_VULKAN
#error "test_vulkan_histogram requires USE_VULKAN builds"
#endif

#define TW 67
#define TH 43
#define TN ((size_t)TW * TH)
#define BINS 256

static dt_vulkan_t s_vk;
static dt_conf_t s_conf;
static gboolean s_have_device = FALSE;

#define REQUIRE_DEVICE() do { if(!s_have_device) skip(); } while(0)

static int group_setup(void **state)
{
  (void)state;
  darktable.conf = &s_conf;
  dt_conf_init(darktable.conf, "/nonexistent-dt-vk-hist-test.rc", FALSE, NULL);
  dt_conf_set_bool("opencl_use_vulkan", TRUE);
  darktable.datadir = g_strdup(TEST_VK_DATADIR);

  darktable.vulkan = &s_vk;
  dt_vulkan_init(&s_vk);
  if(!dt_vulkan_running())
  {
    fprintf(stderr, "no Vulkan device available — histogram tests will be skipped\n");
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

// CPU reference via darktable's own reducer.
static void cpu_histogram(const float *pixels,
                          const dt_iop_colorspace_type_t cst,
                          const dt_iop_colorspace_type_t cst_to,
                          uint32_t *out /* BINS*4 */)
{
  dt_histogram_roi_t roi = { TW, TH, 0, 0, 0, 0 };
  dt_dev_histogram_collection_params_t params = { &roi, BINS };
  dt_dev_histogram_stats_t stats = { 0 };
  uint32_t *hist = NULL;
  dt_histogram_helper(&params, &stats, cst, cst_to, pixels,
                      &hist, NULL, FALSE, NULL);
  assert_non_null(hist);
  memcpy(out, hist, sizeof(uint32_t) * BINS * 4);
  dt_free_align(hist);
}

// GPU path: upload pixels, run the kernel, read the histogram back.
// Runs the whole device sequence, releases the lock, returns rc so
// the caller asserts outside the lock (cmocka longjmps on failure).
static int gpu_histogram(const float *pixels,
                         const dt_iop_colorspace_type_t cst,
                         const dt_iop_colorspace_type_t cst_to,
                         uint32_t *out /* BINS*4 */,
                         gboolean *ran)
{
  const size_t bytes = TN * 4 * sizeof(float);
  const dt_histogram_roi_t roi = { TW, TH, 0, 0, 0, 0 };
  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  *ran = FALSE;
  dt_vk_mem_t *din = NULL;
  if(!rc)
  {
    din = dt_vulkan_alloc_buffer(dev, bytes);
    if(!din) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, din, pixels, bytes);
  if(!rc)
    *ran = dt_histogram_helper_vk(dev, din, TW, TH, &roi, BINS,
                                  cst, cst_to, FALSE, out);
  if(din) dt_vulkan_free_buffer(dev, din);
  if(dev == 0) dt_vulkan_unlock_device(dev);
  return rc;
}

// total variation across the 3 active channels
static uint64_t total_variation(const uint32_t *a, const uint32_t *b)
{
  uint64_t tv = 0;
  for(int bin = 0; bin < BINS; bin++)
    for(int k = 0; k < 3; k++)
    {
      const uint32_t x = a[bin*4+k], y = b[bin*4+k];
      tv += (x > y) ? (x - y) : (y - x);
    }
  return tv;
}

// per-channel sum (must equal the sampled pixel count on both paths)
static void channel_sums(const uint32_t *h, uint64_t s[3])
{
  s[0] = s[1] = s[2] = 0;
  for(int bin = 0; bin < BINS; bin++)
    for(int k = 0; k < 3; k++) s[k] += h[bin*4+k];
}

static void run_case(const float *pixels,
                     const dt_iop_colorspace_type_t cst,
                     const dt_iop_colorspace_type_t cst_to,
                     const gboolean exact, const char *tag)
{
  uint32_t *cpu = malloc(sizeof(uint32_t) * BINS * 4);
  uint32_t *gpu = malloc(sizeof(uint32_t) * BINS * 4);
  assert_non_null(cpu); assert_non_null(gpu);

  gboolean ran = FALSE;
  const int rc = gpu_histogram(pixels, cst, cst_to, gpu, &ran);
  cpu_histogram(pixels, cst, cst_to, cpu);

  uint64_t cs[3], gs[3];
  channel_sums(cpu, cs);
  channel_sums(gpu, gs);
  const uint64_t tv = total_variation(cpu, gpu);
  free(cpu); free(gpu);

  assert_int_equal(rc, 0);
  assert_true(ran);
  // both paths must bin every sampled pixel exactly once per channel
  for(int k = 0; k < 3; k++)
  {
    assert_int_equal((int)cs[k], (int)TN);
    assert_int_equal((int)gs[k], (int)TN);
  }
  if(exact)
  {
    if(tv != 0)
      fail_msg("%s: histograms differ (total variation %llu, expected 0)",
               tag, (unsigned long long)tv);
  }
  else
  {
    // allow a handful of boundary reclassifications: < 1% of the
    // 3*TN binned samples
    const uint64_t budget = (3 * TN) / 100 + 4;
    if(tv > budget)
      fail_msg("%s: total variation %llu exceeds budget %llu",
               tag, (unsigned long long)tv, (unsigned long long)budget);
  }
}

// fill with values quantised to bin centres for the given mode so the
// integer bin is rounding-invariant
static void fill_bin_centres(float *p, const int mode)
{
  const float mb = (float)(BINS - 1);
  for(size_t i = 0; i < TN; i++)
  {
    for(int k = 0; k < 3; k++)
    {
      const int bin = (int)(frand(0.0f, 1.0f) * (BINS - 1));
      const float centred = ((float)bin + 0.5f) / mb; // -> mb*centred = bin+0.5
      if(mode == 0) // RGB: value in [0,1]
        p[i*4+k] = centred;
      else // Lab: invert scale/shift so the binning lands mid-bin
      {
        if(k == 0)      p[i*4+k] = centred * 100.0f;         // L: scale mb/100
        else            p[i*4+k] = centred * 256.0f - 128.0f; // ab: scale mb/256, shift 128
      }
    }
    p[i*4+3] = 1.0f;
  }
}

static void test_rgb_bin_centres(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0xa1;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  fill_bin_centres(p, 0);
  run_case(p, IOP_CS_RGB, IOP_CS_NONE, TRUE, "rgb bin-centres");
  dt_free_align(p);
}

static void test_lab_bin_centres(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0xb2;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  fill_bin_centres(p, 1);
  run_case(p, IOP_CS_LAB, IOP_CS_LAB, TRUE, "lab bin-centres");
  dt_free_align(p);
}

static void test_rgb_random(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0xc3;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  for(size_t i = 0; i < TN; i++)
  {
    p[i*4+0] = frand(-0.1f, 1.1f); p[i*4+1] = frand(-0.1f, 1.1f);
    p[i*4+2] = frand(-0.1f, 1.1f); p[i*4+3] = 1.0f;
  }
  run_case(p, IOP_CS_RGB, IOP_CS_NONE, FALSE, "rgb random");
  dt_free_align(p);
}

static void test_lab_random(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0xd4;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  for(size_t i = 0; i < TN; i++)
  {
    p[i*4+0] = frand(0.0f, 100.0f); p[i*4+1] = frand(-128.0f, 128.0f);
    p[i*4+2] = frand(-128.0f, 128.0f); p[i*4+3] = 1.0f;
  }
  run_case(p, IOP_CS_LAB, IOP_CS_LAB, FALSE, "lab random");
  dt_free_align(p);
}

static void test_lch_random(void **state)
{
  (void)state; REQUIRE_DEVICE();
  s_rng = 0xe5;
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  for(size_t i = 0; i < TN; i++)
  {
    p[i*4+0] = frand(0.0f, 100.0f); p[i*4+1] = frand(-100.0f, 100.0f);
    p[i*4+2] = frand(-100.0f, 100.0f); p[i*4+3] = 1.0f;
  }
  // LCh goes through atan2/hypot where the .cl uses a fast hypot; allow
  // the boundary budget rather than exact match
  run_case(p, IOP_CS_LAB, IOP_CS_LCH, FALSE, "lch random");
  dt_free_align(p);
}

static void test_subset_gates(void **state)
{
  (void)state; REQUIRE_DEVICE();
  // RAW and compensated-RGB must be refused (caller keeps the CPU path)
  const dt_histogram_roi_t roi = { TW, TH, 0, 0, 0, 0 };
  uint32_t out[BINS * 4];
  float *p = dt_alloc_align_float(TN * 4);
  assert_non_null(p);
  memset(p, 0, TN * 4 * sizeof(float));

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  gboolean raw = TRUE, comp = TRUE;
  dt_vk_mem_t *din = NULL;
  if(!rc)
  {
    din = dt_vulkan_alloc_buffer(dev, TN * 4 * sizeof(float));
    if(!din) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, din, p, TN * 4 * sizeof(float));
  if(!rc)
  {
    raw  = dt_histogram_helper_vk(dev, din, TW, TH, &roi, BINS,
                                  IOP_CS_RAW, IOP_CS_NONE, FALSE, out);
    comp = dt_histogram_helper_vk(dev, din, TW, TH, &roi, BINS,
                                  IOP_CS_RGB, IOP_CS_NONE, TRUE, out);
  }
  if(din) dt_vulkan_free_buffer(dev, din);
  if(dev == 0) dt_vulkan_unlock_device(dev);
  dt_free_align(p);

  assert_int_equal(rc, 0);
  assert_false(raw);
  assert_false(comp);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_rgb_bin_centres),
    cmocka_unit_test(test_lab_bin_centres),
    cmocka_unit_test(test_rgb_random),
    cmocka_unit_test(test_lab_random),
    cmocka_unit_test(test_lch_random),
    cmocka_unit_test(test_subset_gates),
  };
  return cmocka_run_group_tests(tests, group_setup, group_teardown);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
