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
 * cmocka tests for the Vulkan capture executor (M1 of
 * dev-doc/gpu_resident_pixelpipe_dag.md): captured multi-dispatch
 * chains must be bit-identical to their eager execution and collapse
 * to a single queue submission per segment.
 *
 * Runs on any Vulkan implementation, including lavapipe in headless
 * CI. All tests skip cleanly when no device is available.
 */
#include <limits.h>
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
#include "control/conf.h"

#ifdef _WIN32
#include "win/main_wrapper.h"
#endif

#ifndef HAVE_VULKAN
#error "test_vulkan_capture requires USE_VULKAN builds"
#endif

// non-multiples of the 8x8 local size so the ceil-div workgroup count
// and the in-kernel bounds check are exercised
#define TW 67
#define TH 41
#define TN ((size_t)TW * TH)
#define TBYTES (TN * sizeof(float))

typedef struct pc_addmul_t
{
  int32_t width, height;
  float k;
} pc_addmul_t;

typedef struct pc_lut_t
{
  int32_t width, height;
} pc_lut_t;

static dt_vulkan_t s_vk;
static dt_conf_t s_conf;
static gboolean s_have_device = FALSE;
static dt_vk_module_kernel_t k_add = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t k_mul = DT_VK_MODULE_KERNEL_INIT;
static dt_vk_module_kernel_t k_lut = DT_VK_MODULE_KERNEL_INIT;

static void _fill_input(float *buf)
{
  for(size_t i = 0; i < TN; i++)
    buf[i] = 0.5f * (float)i - 100.0f;
}

// require a working device or skip; cmocka's skip() long-jumps out
#define REQUIRE_DEVICE() do { if(!s_have_device) skip(); } while(0)

/*
 * SETUP / TEARDOWN
 */

static int group_setup(void **state)
{
  (void)state;
  // Minimal darktable bootstrap: vulkan.c needs dt_conf for the
  // enable pref and dt_print for logging; everything else in the
  // zero-initialized `darktable` global is fine as-is.
  darktable.conf = &s_conf;
  dt_conf_init(darktable.conf, "/nonexistent-dt-vk-capture-test.rc", FALSE, NULL);
  dt_conf_set_bool("opencl_use_vulkan", TRUE);

  darktable.vulkan = &s_vk;
  dt_vulkan_init(&s_vk);
  if(!dt_vulkan_running())
  {
    fprintf(stderr, "no Vulkan device available — capture tests will be skipped\n");
    return 0;
  }

  const int p_add = dt_vulkan_load_program("capt_add", TEST_VK_KERNEL_DIR "/capt_add.spv");
  const int p_mul = dt_vulkan_load_program("capt_mul", TEST_VK_KERNEL_DIR "/capt_mul.spv");
  const int p_lut = dt_vulkan_load_program("capt_lut", TEST_VK_KERNEL_DIR "/capt_lut.spv");
  dt_vulkan_module_kernel_create_from(&k_add, p_add, "main", 2, sizeof(pc_addmul_t), 8, 8, 1);
  dt_vulkan_module_kernel_create_from(&k_mul, p_mul, "main", 2, sizeof(pc_addmul_t), 8, 8, 1);
  dt_vulkan_module_kernel_create_from(&k_lut, p_lut, "main", 3, sizeof(pc_lut_t), 8, 8, 1);
  if(k_add.kernel < 0 || k_mul.kernel < 0 || k_lut.kernel < 0)
  {
    fprintf(stderr, "test kernels failed to load — capture tests will be skipped\n");
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

/*
 * TESTS
 */

// A 3-kernel chain captured into one segment must (a) be bit-identical
// to the eager execution of the same calls and (b) reach the queue as
// exactly one submission, where the eager path needs five.
static void test_captured_chain_bitexact_single_submit(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  float *in = malloc(TBYTES), *out_eager = malloc(TBYTES),
        *out_capt = malloc(TBYTES), *ref = malloc(TBYTES);
  assert_non_null(in); assert_non_null(out_eager);
  assert_non_null(out_capt); assert_non_null(ref);
  _fill_input(in);
  for(size_t i = 0; i < TN; i++)
    ref[i] = (in[i] + 1.5f) * 2.0f + 0.25f;

  const pc_addmul_t pc1 = { TW, TH, 1.5f };
  const pc_addmul_t pc2 = { TW, TH, 2.0f };
  const pc_addmul_t pc3 = { TW, TH, 0.25f };

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  dt_vk_mem_t *a = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *b = dt_vulkan_alloc_buffer(dev, TBYTES);
  assert_non_null(a); assert_non_null(b);

  // ---- eager reference run ----
  uint64_t s0 = dt_vulkan_submission_count(dev);
  assert_int_equal(dt_vulkan_write_to_device(dev, a, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc1, sizeof(pc1)), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_mul, b, a, TW, TH, &pc2, sizeof(pc2)), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc3, sizeof(pc3)), 0);
  assert_int_equal(dt_vulkan_read_from_device(dev, out_eager, b, TBYTES), 0);
  assert_int_equal((int)(dt_vulkan_submission_count(dev) - s0), 5);

  // ---- captured run: same calls, one flush ----
  s0 = dt_vulkan_submission_count(dev);
  assert_true(dt_vulkan_capture_begin(dev));
  assert_true(dt_vulkan_capture_active());
  assert_int_equal(dt_vulkan_write_to_device(dev, a, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc1, sizeof(pc1)), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_mul, b, a, TW, TH, &pc2, sizeof(pc2)), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc3, sizeof(pc3)), 0);
  assert_int_equal(dt_vulkan_capture_pending(), 4);
  // nothing may have reached the queue while capturing
  assert_int_equal((int)(dt_vulkan_submission_count(dev) - s0), 0);
  assert_int_equal(dt_vulkan_capture_end(dev), 0);
  assert_false(dt_vulkan_capture_active());
  // the whole 4-node span: exactly one submission
  assert_int_equal((int)(dt_vulkan_submission_count(dev) - s0), 1);
  assert_int_equal(dt_vulkan_read_from_device(dev, out_capt, b, TBYTES), 0);

  assert_memory_equal(out_eager, out_capt, TBYTES);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(out_capt[i], ref[i], 1e-4);

  dt_vulkan_free_buffer(dev, a);
  dt_vulkan_free_buffer(dev, b);
  dt_vulkan_unlock_device(dev);
  free(in); free(out_eager); free(out_capt); free(ref);
}

// A mid-capture readback is a sync tap: it must flush the pending
// segment, serve correct data, and leave the capture active for the
// rest of the span.
static void test_sync_tap_mid_capture(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  float *in = malloc(TBYTES), *mid = malloc(TBYTES), *out = malloc(TBYTES);
  assert_non_null(in); assert_non_null(mid); assert_non_null(out);
  _fill_input(in);

  const pc_addmul_t pc1 = { TW, TH, 1.5f };
  const pc_addmul_t pc2 = { TW, TH, 2.0f };

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  dt_vk_mem_t *a = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *b = dt_vulkan_alloc_buffer(dev, TBYTES);
  assert_non_null(a); assert_non_null(b);

  const uint64_t s0 = dt_vulkan_submission_count(dev);
  assert_true(dt_vulkan_capture_begin(dev));
  assert_int_equal(dt_vulkan_write_to_device(dev, a, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc1, sizeof(pc1)), 0);
  assert_int_equal(dt_vulkan_capture_pending(), 2);

  // sync tap: read what only the pending nodes can have produced
  assert_int_equal(dt_vulkan_read_from_device(dev, mid, b, TBYTES), 0);
  assert_true(dt_vulkan_capture_active());
  assert_int_equal(dt_vulkan_capture_pending(), 0);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(mid[i], in[i] + 1.5f, 1e-4);

  // the span continues as a second segment
  assert_int_equal(dt_vulkan_dispatch_inout(&k_mul, b, a, TW, TH, &pc2, sizeof(pc2)), 0);
  assert_int_equal(dt_vulkan_capture_end(dev), 0);
  assert_int_equal(dt_vulkan_read_from_device(dev, out, a, TBYTES), 0);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(out[i], (in[i] + 1.5f) * 2.0f, 1e-4);

  // tap flush + tap read + end flush + final read
  assert_int_equal((int)(dt_vulkan_submission_count(dev) - s0), 4);

  dt_vulkan_free_buffer(dev, a);
  dt_vulkan_free_buffer(dev, b);
  dt_vulkan_unlock_device(dev);
  free(in); free(mid); free(out);
}

// Batched-upload payloads (module LUTs live in stack arrays) must be
// snapshotted at capture time: clobbering the host array between
// capture and flush must not change the result.
static void test_upload_snapshot_survives_clobber(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  float *in = malloc(TBYTES), *out = malloc(TBYTES);
  assert_non_null(in); assert_non_null(out);
  _fill_input(in);
  float lutvals[256];
  for(int i = 0; i < 256; i++) lutvals[i] = 3.0f;

  const pc_lut_t pc = { TW, TH };

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  dt_vk_mem_t *a = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *b = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *lutbuf = dt_vulkan_alloc_buffer(dev, sizeof(lutvals));
  assert_non_null(a); assert_non_null(b); assert_non_null(lutbuf);

  assert_true(dt_vulkan_capture_begin(dev));
  assert_int_equal(dt_vulkan_write_to_device(dev, a, in, TBYTES), 0);
  const dt_vk_upload_t up = { .dst = lutbuf, .host = lutvals, .size = sizeof(lutvals) };
  dt_vk_mem_t *bufs[3] = { a, b, lutbuf };
  assert_int_equal(dt_vulkan_dispatch_n_batched(&k_lut, bufs, 3, &up, 1,
                                                TW, TH, &pc, sizeof(pc)), 0);
  // the module's stack LUT dies before the flush — simulate the worst case
  memset(lutvals, 0, sizeof(lutvals));
  assert_int_equal(dt_vulkan_capture_end(dev), 0);

  assert_int_equal(dt_vulkan_read_from_device(dev, out, b, TBYTES), 0);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(out[i], in[i] + 3.0f, 1e-4);

  dt_vulkan_free_buffer(dev, a);
  dt_vulkan_free_buffer(dev, b);
  dt_vulkan_free_buffer(dev, lutbuf);
  dt_vulkan_unlock_device(dev);
  free(in); free(out);
}

// The borrowed-upload variant must produce the same bytes as the
// snapshotting one when the pointer stays alive (its documented
// contract), both under capture and eagerly.
static void test_borrowed_upload(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  float *in = malloc(TBYTES), *out = malloc(TBYTES);
  assert_non_null(in); assert_non_null(out);
  _fill_input(in);
  const pc_addmul_t pc0 = { TW, TH, 0.0f };

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  dt_vk_mem_t *a = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *b = dt_vulkan_alloc_buffer(dev, TBYTES);
  assert_non_null(a); assert_non_null(b);

  // captured: borrow `in`, which outlives the end() below
  assert_true(dt_vulkan_capture_begin(dev));
  assert_int_equal(dt_vulkan_write_to_device_borrowed(dev, a, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc0, sizeof(pc0)), 0);
  assert_int_equal(dt_vulkan_capture_end(dev), 0);
  assert_int_equal(dt_vulkan_read_from_device(dev, out, b, TBYTES), 0);
  assert_memory_equal(out, in, TBYTES);

  // eager: identical to the plain write
  memset(out, 0, TBYTES);
  assert_int_equal(dt_vulkan_write_to_device_borrowed(dev, b, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_read_from_device(dev, out, b, TBYTES), 0);
  assert_memory_equal(out, in, TBYTES);

  dt_vulkan_free_buffer(dev, a);
  dt_vulkan_free_buffer(dev, b);
  dt_vulkan_unlock_device(dev);
  free(in); free(out);
}

// dt_vulkan_free_buffer during capture must defer: the freed scratch
// may not re-enter the allocation pool before the flush, or a later
// alloc in the same span could alias a buffer that pending nodes
// still reference (or, worse, the VkBuffer could be destroyed under
// them when the pool is full).
static void test_free_is_deferred_until_flush(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  // pool-unique size so a best-fit re-issue of the "freed" scratch is
  // the only same-size candidate
  const size_t SBYTES = TBYTES + 64 * sizeof(float);

  float *in = malloc(TBYTES), *out = malloc(TBYTES), *zeros = calloc(1, TBYTES);
  assert_non_null(in); assert_non_null(out); assert_non_null(zeros);
  _fill_input(in);
  const pc_addmul_t pc1 = { TW, TH, 1.0f };
  const pc_addmul_t pc2 = { TW, TH, 2.0f };

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  dt_vk_mem_t *x = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *y = dt_vulkan_alloc_buffer(dev, TBYTES);
  assert_non_null(x); assert_non_null(y);

  assert_true(dt_vulkan_capture_begin(dev));
  assert_int_equal(dt_vulkan_write_to_device(dev, x, in, TBYTES), 0);
  dt_vk_mem_t *t = dt_vulkan_alloc_buffer(dev, SBYTES);
  assert_non_null(t);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, x, t, TW, TH, &pc1, sizeof(pc1)), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_mul, t, y, TW, TH, &pc2, sizeof(pc2)), 0);
  dt_vulkan_free_buffer(dev, t); // module done with its scratch — deferred

  // an allocation of the same size inside the same span must NOT get
  // the deferred buffer back
  dt_vk_mem_t *t2 = dt_vulkan_alloc_buffer(dev, SBYTES);
  assert_non_null(t2);
  assert_ptr_not_equal(t2, t);
  assert_int_equal(dt_vulkan_write_to_device(dev, t2, zeros, TBYTES), 0);

  assert_int_equal(dt_vulkan_capture_end(dev), 0);
  assert_int_equal(dt_vulkan_read_from_device(dev, out, y, TBYTES), 0);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(out[i], (in[i] + 1.0f) * 2.0f, 1e-4);

  dt_vulkan_free_buffer(dev, t2);
  dt_vulkan_free_buffer(dev, x);
  dt_vulkan_free_buffer(dev, y);
  dt_vulkan_unlock_device(dev);
  free(in); free(out); free(zeros);
}

// Module-granular rollback: nodes appended after the mark disappear,
// nodes before it survive and execute.
static void test_rollback_drops_failed_module(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  float *in = malloc(TBYTES), *out = malloc(TBYTES);
  assert_non_null(in); assert_non_null(out);
  _fill_input(in);
  const pc_addmul_t pc_add = { TW, TH, 1.5f };
  const pc_addmul_t pc_bad = { TW, TH, 9.0f };
  const pc_addmul_t pc_good = { TW, TH, 2.0f };

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  dt_vk_mem_t *a = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *b = dt_vulkan_alloc_buffer(dev, TBYTES);
  assert_non_null(a); assert_non_null(b);

  assert_true(dt_vulkan_capture_begin(dev));
  assert_int_equal(dt_vulkan_write_to_device(dev, a, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc_add, sizeof(pc_add)), 0);
  assert_int_equal(dt_vulkan_capture_pending(), 2);

  // "module" starts, allocates scratch, dispatches, then fails
  const dt_vk_capture_mark_t mark = dt_vulkan_capture_mark();
  dt_vk_mem_t *scratch = dt_vulkan_alloc_buffer(dev, TBYTES);
  assert_non_null(scratch);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_mul, b, scratch, TW, TH, &pc_bad, sizeof(pc_bad)), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_mul, scratch, a, TW, TH, &pc_bad, sizeof(pc_bad)), 0);
  dt_vulkan_free_buffer(dev, scratch);
  assert_int_equal(dt_vulkan_capture_pending(), 4);
  dt_vulkan_capture_rollback(dev, &mark);
  assert_int_equal(dt_vulkan_capture_pending(), 2);

  // the replacement path runs instead
  assert_int_equal(dt_vulkan_dispatch_inout(&k_mul, b, a, TW, TH, &pc_good, sizeof(pc_good)), 0);
  assert_int_equal(dt_vulkan_capture_end(dev), 0);
  assert_int_equal(dt_vulkan_read_from_device(dev, out, a, TBYTES), 0);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(out[i], (in[i] + 1.5f) * 2.0f, 1e-4);

  dt_vulkan_free_buffer(dev, a);
  dt_vulkan_free_buffer(dev, b);
  dt_vulkan_unlock_device(dev);
  free(in); free(out);
}

// COPY and COPY_ROWS nodes inside a captured segment.
static void test_copy_nodes(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  float *in = malloc(TBYTES), *out = malloc(TBYTES), *ref = malloc(TBYTES);
  assert_non_null(in); assert_non_null(out); assert_non_null(ref);
  _fill_input(in);
  const pc_addmul_t pc1 = { TW, TH, 1.0f };

  // CPU reference of the captured sequence below
  for(size_t i = 0; i < TN; i++) ref[i] = in[i]; // b = copy(a)
  const size_t sox = 5, soy = 7, dox = 11, doy = 3, rw = 13, rh = 9;
  for(size_t r = 0; r < rh; r++)
    for(size_t c = 0; c < rw; c++)
      ref[(doy + r) * TW + dox + c] = in[(soy + r) * TW + sox + c] + 1.0f;

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  dt_vk_mem_t *a = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *b = dt_vulkan_alloc_buffer(dev, TBYTES);
  assert_non_null(a); assert_non_null(b);

  const uint64_t s0 = dt_vulkan_submission_count(dev);
  assert_true(dt_vulkan_capture_begin(dev));
  assert_int_equal(dt_vulkan_write_to_device(dev, a, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_copy_device_to_device(dev, b, a, TBYTES), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, b, a, TW, TH, &pc1, sizeof(pc1)), 0);
  assert_int_equal(dt_vulkan_copy_subregion(dev, b, a, sox, soy, dox, doy,
                                            rw, rh, TW, TW, sizeof(float)), 0);
  assert_int_equal(dt_vulkan_capture_pending(), 4);
  assert_int_equal(dt_vulkan_capture_end(dev), 0);
  assert_int_equal((int)(dt_vulkan_submission_count(dev) - s0), 1);

  assert_int_equal(dt_vulkan_read_from_device(dev, out, b, TBYTES), 0);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(out[i], ref[i], 1e-4);

  dt_vulkan_free_buffer(dev, a);
  dt_vulkan_free_buffer(dev, b);
  dt_vulkan_unlock_device(dev);
  free(in); free(out); free(ref);
}

// Abort drops the pending nodes without touching the queue and leaves
// the HAL fully usable in eager mode.
static void test_abort_discards_pending_work(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  float *in = malloc(TBYTES), *out = malloc(TBYTES);
  assert_non_null(in); assert_non_null(out);
  _fill_input(in);
  const pc_addmul_t pc1 = { TW, TH, 1.0f };

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  dt_vk_mem_t *a = dt_vulkan_alloc_buffer(dev, TBYTES);
  dt_vk_mem_t *b = dt_vulkan_alloc_buffer(dev, TBYTES);
  assert_non_null(a); assert_non_null(b);

  const uint64_t s0 = dt_vulkan_submission_count(dev);
  assert_true(dt_vulkan_capture_begin(dev));
  assert_int_equal(dt_vulkan_write_to_device(dev, a, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc1, sizeof(pc1)), 0);
  assert_int_equal(dt_vulkan_capture_pending(), 2);
  dt_vulkan_capture_abort(dev);
  assert_false(dt_vulkan_capture_active());
  assert_int_equal(dt_vulkan_capture_pending(), 0);
  // nothing reached the queue
  assert_int_equal((int)(dt_vulkan_submission_count(dev) - s0), 0);

  // eager mode still works on the same buffers afterwards
  assert_int_equal(dt_vulkan_write_to_device(dev, a, in, TBYTES), 0);
  assert_int_equal(dt_vulkan_dispatch_inout(&k_add, a, b, TW, TH, &pc1, sizeof(pc1)), 0);
  assert_int_equal(dt_vulkan_read_from_device(dev, out, b, TBYTES), 0);
  for(size_t i = 0; i < TN; i++)
    assert_float_equal(out[i], in[i] + 1.0f, 1e-4);

  dt_vulkan_free_buffer(dev, a);
  dt_vulkan_free_buffer(dev, b);
  dt_vulkan_unlock_device(dev);
  free(in); free(out);
}

// A second begin on the same thread must refuse (and report inactive
// again once the first capture ends).
static void test_nested_begin_refused(void **state)
{
  (void)state;
  REQUIRE_DEVICE();

  const int dev = dt_vulkan_lock_device();
  assert_int_equal(dev, 0);
  assert_true(dt_vulkan_capture_begin(dev));
  assert_false(dt_vulkan_capture_begin(dev));
  assert_true(dt_vulkan_capture_active());
  assert_int_equal(dt_vulkan_capture_end(dev), 0);
  assert_false(dt_vulkan_capture_active());
  dt_vulkan_unlock_device(dev);
}

int main(void)
{
  const struct CMUnitTest tests[] = {
    cmocka_unit_test(test_captured_chain_bitexact_single_submit),
    cmocka_unit_test(test_sync_tap_mid_capture),
    cmocka_unit_test(test_upload_snapshot_survives_clobber),
    cmocka_unit_test(test_borrowed_upload),
    cmocka_unit_test(test_free_is_deferred_until_flush),
    cmocka_unit_test(test_rollback_drops_failed_module),
    cmocka_unit_test(test_copy_nodes),
    cmocka_unit_test(test_abort_discards_pending_work),
    cmocka_unit_test(test_nested_begin_refused),
  };
  return cmocka_run_group_tests(tests, group_setup, group_teardown);
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
