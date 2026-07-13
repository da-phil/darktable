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
 * cmocka tests for the Vulkan colorspace glue kernels (DAG milestone
 * M2, dev-doc/gpu_resident_pixelpipe_dag.md §5.4).
 *
 * These are the on-device Lab<->RGB working-space transforms that let
 * a GPU span survive the colorspace hop between modules instead of
 * flushing to convert on the host. The glue node moves the transform
 * from CPU to GPU, so its output is NOT bit-identical to the eager
 * (host-transform) path — it is float-precision equivalent. This test
 * validates that equivalence at the source: the same profile fed to
 * darktable's own CPU dt_ioppr_transform_image_colorspace and to
 * dt_ioppr_transform_image_colorspace_vk must agree to a tight
 * tolerance, for both a linear and a nonlinear (sRGB-gamma) TRC, in
 * both directions, plus a round-trip inversion check.
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
#include "common/iop_profile.h"
#include "control/conf.h"
#include "develop/imageop.h"

#ifndef HAVE_VULKAN
#error "test_vulkan_colorspace requires USE_VULKAN builds"
#endif

#define TW 53
#define TH 41
#define TN ((size_t)TW * TH)

static dt_vulkan_t s_vk;
static dt_conf_t s_conf;
static gboolean s_have_device = FALSE;

// a minimal module stand-in: dt_ioppr_transform_image_colorspace only
// touches self->op in debug prints, which are gated off at unmuted=0
static dt_iop_module_t s_module;

#define REQUIRE_DEVICE() do { if(!s_have_device) skip(); } while(0)

static int group_setup(void **state)
{
  (void)state;
  darktable.conf = &s_conf;
  dt_conf_init(darktable.conf, "/nonexistent-dt-vk-cst-test.rc", FALSE, NULL);
  dt_conf_set_bool("opencl_use_vulkan", TRUE);
  darktable.datadir = g_strdup(TEST_VK_DATADIR);

  darktable.vulkan = &s_vk;
  dt_vulkan_init(&s_vk);
  if(!dt_vulkan_running())
  {
    fprintf(stderr, "no Vulkan device available — colorspace tests will be skipped\n");
    return 0;
  }
  memset(&s_module, 0, sizeof(s_module));
  g_strlcpy(s_module.op, "csttest", sizeof(s_module.op));
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

// deterministic pseudo-random floats
static uint32_t s_rng;
static float frand(float lo, float hi)
{
  s_rng = s_rng * 1664525u + 1013904223u;
  return lo + (hi - lo) * ((s_rng >> 8) & 0xFFFFFF) / (float)0xFFFFFF;
}

// sRGB D50-adapted primaries matrix (matches dt_vulkan_common.h's
// vk_sRGB_to_XYZ / vk_XYZ_to_sRGB and colorspace.h). Row-major 3x3.
static const float SRGB_TO_XYZ[9] = {
  0.4360747f, 0.3850649f, 0.1430804f,
  0.2225045f, 0.7168786f, 0.0606169f,
  0.0139322f, 0.0971045f, 0.7141733f };
static const float XYZ_TO_SRGB[9] = {
   3.1338561f, -1.6168667f, -0.4906146f,
  -0.9787684f,  1.9161415f,  0.0334540f,
   0.0719453f, -0.2289914f,  1.4052427f };

// Build a profile_info with the sRGB matrices. When nonlinear, fill
// the in/out TRC LUTs with the sRGB gamma curve and its inverse and
// set unbounded_coeffs to the linear-extrapolation fast path so both
// CPU and GPU agree above x = 1.
static void make_profile(dt_iop_order_iccprofile_info_t *p, const gboolean nonlinear)
{
  const int lutsize = nonlinear ? 512 : 0;
  dt_ioppr_init_profile_info(p, lutsize);
  p->type = DT_COLORSPACE_SRGB;
  g_strlcpy(p->filename, "srgb-test", sizeof(p->filename));

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
    {
      p->matrix_in[i][j]  = SRGB_TO_XYZ[i * 3 + j];
      p->matrix_out[i][j] = XYZ_TO_SRGB[i * 3 + j];
      // the CPU matrix path (_transform_rgb_to_lab_matrix et al.)
      // reads the transposed copies; the GPU kernel reads matrix_in/
      // matrix_out directly. Fill both so the two agree.
      p->matrix_in_transposed[j][i]  = SRGB_TO_XYZ[i * 3 + j];
      p->matrix_out_transposed[j][i] = XYZ_TO_SRGB[i * 3 + j];
    }
  p->nonlinearlut = nonlinear ? 1 : 0;
  p->grey = 0.0f;

  if(nonlinear)
  {
    for(int c = 0; c < 3; c++)
    {
      for(int k = 0; k < lutsize; k++)
      {
        const float x = (float)k / (float)(lutsize - 1);
        // sRGB EOTF (encoded->linear) as lut_in, its inverse as lut_out
        p->lut_in[c][k]  = (x <= 0.04045f) ? x / 12.92f
                                           : powf((x + 0.055f) / 1.055f, 2.4f);
        p->lut_out[c][k] = (x <= 0.0031308f) ? x * 12.92f
                                             : 1.055f * powf(x, 1.0f / 2.4f) - 0.055f;
      }
      // linear-extrapolation fast path for x >= 1 (a[0] >= 0): both
      // CPU (extrapolate_lut) and GPU (vk_lerp_lookup_unbounded) use
      // coeff[1] * pow(x * coeff[0], coeff[2]); identity coefficients
      // keep them exactly equal there.
      p->unbounded_coeffs_in[c][0]  = 1.0f;
      p->unbounded_coeffs_in[c][1]  = 1.0f;
      p->unbounded_coeffs_in[c][2]  = 1.0f;
      p->unbounded_coeffs_out[c][0] = 1.0f;
      p->unbounded_coeffs_out[c][1] = 1.0f;
      p->unbounded_coeffs_out[c][2] = 1.0f;
    }
  }
}

// scale-aware error: absolute below |ref| = 1, relative above. Lab L
// runs to ~100, so an absolute epsilon calibrated near RGB [0,1] would
// be far too tight on the L channel.
static float scaled_err(const float *g, const float *r, const int ch)
{
  float e = 0.0f;
  for(int c = 0; c < ch; c++)
  {
    const float d = fabsf(g[c] - r[c]) / fmaxf(1.0f, fabsf(r[c]));
    if(d > e) e = d;
  }
  return e;
}

// Run one VK transform and compare against the CPU transform of the
// same profile. Returns via cmocka asserts; releases the device lock
// before asserting (cmocka longjmps on failure).
static void check_transform(const dt_iop_order_iccprofile_info_t *p,
                            const dt_iop_colorspace_type_t cst_from,
                            const dt_iop_colorspace_type_t cst_to,
                            const float *in, float eps, const char *tag)
{
  const size_t bytes = TN * 4 * sizeof(float);
  float *cpu = dt_alloc_align_float(TN * 4);
  float *gpu = malloc(bytes);
  assert_non_null(cpu); assert_non_null(gpu);

  // CPU reference (out-of-place)
  dt_iop_colorspace_type_t cpu_cst = cst_from;
  memcpy(cpu, in, bytes);
  dt_ioppr_transform_image_colorspace(&s_module, in, cpu, TW, TH,
                                      cst_from, cst_to, &cpu_cst, p);

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  dt_iop_colorspace_type_t gpu_cst = cst_from;
  dt_vk_mem_t *da = NULL, *db = NULL;
  if(!rc)
  {
    da = dt_vulkan_alloc_buffer(dev, bytes);
    db = dt_vulkan_alloc_buffer(dev, bytes);
    if(!da || !db) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, da, in, bytes);
  gboolean ran = FALSE;
  if(!rc)
    ran = dt_ioppr_transform_image_colorspace_vk(dev, da, db, TW, TH,
                                                 cst_from, cst_to, &gpu_cst, p, tag);
  if(!rc && ran) rc = dt_vulkan_read_from_device(dev, gpu, db, bytes);
  if(da) dt_vulkan_free_buffer(dev, da);
  if(db) dt_vulkan_free_buffer(dev, db);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  float maxerr = 0.0f;
  size_t worst = 0;
  if(!rc && ran && gpu_cst == cst_to && cpu_cst == cst_to)
    for(size_t i = 0; i < TN; i++)
    {
      const float e = scaled_err(gpu + 4 * i, cpu + 4 * i, 3);
      if(e > maxerr) { maxerr = e; worst = i; }
    }

  float ga[4] = {0}, ca[4] = {0};
  if(!rc && ran) { memcpy(ga, gpu + 4 * worst, sizeof(ga)); memcpy(ca, cpu + 4 * worst, sizeof(ca)); }
  dt_free_align(cpu); free(gpu);

  assert_int_equal(rc, 0);
  assert_true(ran);
  assert_int_equal(gpu_cst, cst_to);
  if(maxerr > eps)
    fail_msg("%s: max scaled error %g exceeds eps %g at px %zu\n"
             "  gpu  = (%g, %g, %g)\n  cpu  = (%g, %g, %g)",
             tag, (double)maxerr, (double)eps, worst,
             (double)ga[0], (double)ga[1], (double)ga[2],
             (double)ca[0], (double)ca[1], (double)ca[2]);
}

static void test_linear_rgb_to_lab(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  dt_iop_order_iccprofile_info_t p;
  make_profile(&p, FALSE);
  s_rng = 0x1001;
  float *in = dt_alloc_align_float(TN * 4);
  assert_non_null(in);
  for(size_t i = 0; i < TN; i++)
  {
    in[4*i+0] = frand(0.0f, 1.2f); in[4*i+1] = frand(0.0f, 1.2f);
    in[4*i+2] = frand(0.0f, 1.2f); in[4*i+3] = frand(0.0f, 1.0f);
  }
  check_transform(&p, IOP_CS_RGB, IOP_CS_LAB, in, 1e-4f, "linear rgb->lab");
  dt_free_align(in);
  dt_ioppr_cleanup_profile_info(&p);
}

static void test_linear_lab_to_rgb(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  dt_iop_order_iccprofile_info_t p;
  make_profile(&p, FALSE);
  s_rng = 0x2002;
  float *in = dt_alloc_align_float(TN * 4);
  assert_non_null(in);
  for(size_t i = 0; i < TN; i++)
  {
    in[4*i+0] = frand(0.0f, 100.0f); in[4*i+1] = frand(-90.0f, 90.0f);
    in[4*i+2] = frand(-90.0f, 90.0f); in[4*i+3] = frand(0.0f, 1.0f);
  }
  check_transform(&p, IOP_CS_LAB, IOP_CS_RGB, in, 1e-4f, "linear lab->rgb");
  dt_free_align(in);
  dt_ioppr_cleanup_profile_info(&p);
}

static void test_nonlinear_rgb_to_lab(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  dt_iop_order_iccprofile_info_t p;
  make_profile(&p, TRUE);
  s_rng = 0x3003;
  float *in = dt_alloc_align_float(TN * 4);
  assert_non_null(in);
  for(size_t i = 0; i < TN; i++)
  {
    in[4*i+0] = frand(0.0f, 1.0f); in[4*i+1] = frand(0.0f, 1.0f);
    in[4*i+2] = frand(0.0f, 1.0f); in[4*i+3] = frand(0.0f, 1.0f);
  }
  // the TRC LUT lerp diverges by up to a few 1e-4 between the CPU's
  // 0x10000-scaled 2-D tile lookup and the GPU's flat lutsize lerp
  check_transform(&p, IOP_CS_RGB, IOP_CS_LAB, in, 3e-3f, "srgb rgb->lab");
  dt_free_align(in);
  dt_ioppr_cleanup_profile_info(&p);
}

static void test_roundtrip(void **state)
{
  (void)state;
  REQUIRE_DEVICE();
  dt_iop_order_iccprofile_info_t p;
  make_profile(&p, FALSE);
  s_rng = 0x4004;
  const size_t bytes = TN * 4 * sizeof(float);
  float *in = dt_alloc_align_float(TN * 4);
  float *out = malloc(bytes);
  assert_non_null(in); assert_non_null(out);
  for(size_t i = 0; i < TN; i++)
  {
    in[4*i+0] = frand(0.05f, 0.95f); in[4*i+1] = frand(0.05f, 0.95f);
    in[4*i+2] = frand(0.05f, 0.95f); in[4*i+3] = frand(0.0f, 1.0f);
  }

  const int dev = dt_vulkan_lock_device();
  int rc = (dev == 0) ? 0 : -100;
  dt_iop_colorspace_type_t c1 = IOP_CS_RGB, c2 = IOP_CS_LAB;
  gboolean r1 = FALSE, r2 = FALSE;
  dt_vk_mem_t *da = NULL, *db = NULL;
  if(!rc)
  {
    da = dt_vulkan_alloc_buffer(dev, bytes);
    db = dt_vulkan_alloc_buffer(dev, bytes);
    if(!da || !db) rc = -101;
  }
  if(!rc) rc = dt_vulkan_write_to_device(dev, da, in, bytes);
  if(!rc) r1 = dt_ioppr_transform_image_colorspace_vk(dev, da, db, TW, TH,
                                                     IOP_CS_RGB, IOP_CS_LAB, &c1, &p, "rt fwd");
  if(!rc && r1) r2 = dt_ioppr_transform_image_colorspace_vk(dev, db, da, TW, TH,
                                                          IOP_CS_LAB, IOP_CS_RGB, &c2, &p, "rt inv");
  if(!rc && r2) rc = dt_vulkan_read_from_device(dev, out, da, bytes);
  if(da) dt_vulkan_free_buffer(dev, da);
  if(db) dt_vulkan_free_buffer(dev, db);
  if(dev == 0) dt_vulkan_unlock_device(dev);

  float maxerr = 0.0f;
  if(!rc && r1 && r2)
    for(size_t i = 0; i < TN; i++)
      for(int c = 0; c < 3; c++)
        maxerr = fmaxf(maxerr, fabsf(out[4*i+c] - in[4*i+c]));
  dt_free_align(in); free(out);

  assert_int_equal(rc, 0);
  assert_true(r1); assert_true(r2);
  assert_int_equal(c1, IOP_CS_LAB);
  assert_int_equal(c2, IOP_CS_RGB);
  if(maxerr > 1e-4f)
    fail_msg("roundtrip rgb->lab->rgb max error %g exceeds 1e-4", (double)maxerr);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_linear_rgb_to_lab),
    cmocka_unit_test(test_linear_lab_to_rgb),
    cmocka_unit_test(test_nonlinear_rgb_to_lab),
    cmocka_unit_test(test_roundtrip),
  };
  return cmocka_run_group_tests(tests, group_setup, group_teardown);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
