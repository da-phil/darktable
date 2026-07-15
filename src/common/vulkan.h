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
    Vulkan compute backend for darktable's pixelpipe (experimental).

    This is the API surface that IOP modules use to dispatch GPU work
    via Vulkan. It is intentionally a subset of the OpenCL host API in
    src/common/opencl.h — only the operations needed by modules that
    have been ported to Vulkan are exposed.

    Versus OpenCL:
      - kernels are loaded as pre-compiled SPIR-V modules (one per .cl
        source, produced by clspv at build time)
      - device buffers are explicit (no image2d_t — modules pass
        VkBuffer handles wrapped in dt_vk_mem_t)
      - dispatch is synchronous (single dispatch per kernel call);
        async batching is an optimisation for later
*/

#pragma once

#include "common/darktable.h"

#include <stddef.h>
#include <stdint.h>

#ifdef HAVE_VULKAN

#include <stdbool.h>
#include <vulkan/vulkan.h>

// One program slot is consumed per distinct .spv loaded; one kernel
// slot per entry-point dispatched. With all 73 ported modules plus
// the shared helpers (bilateral, guided-filter, local-laplacian,
// gaussian, à-trous) the in-tree build loads ~185 programs and ~200
// kernels — and dt_vulkan_load_program does NOT dedup, so every
// init_global call takes its own slot. The previous ceilings (64 /
// 256) silently starved every module whose init_global lost the
// (filesystem-order) race for the first 64 program slots: its
// load_program returned -1, its kernel slot stayed -1, and every
// process_vk dispatch fell back to CPU with "dispatch returned
// non-zero". Sized here with generous headroom over the current
// counts so new ports don't reintroduce the starvation.
#define DT_VULKAN_MAX_PROGRAMS 384
#define DT_VULKAN_MAX_KERNELS  512
// 20 lets the guided-filter `solve` kernel bind its 13 inputs + 4
// outputs (17) in a single dispatch. Real targets (RADV, lavapipe,
// MoltenVK→Metal ~30 buffers/stage) all allow well over this; the
// Vulkan spec floor of 4 is academic on any darktable-capable GPU.
#define DT_VULKAN_MAX_BINDINGS 20
#define DT_VULKAN_MAX_PUSH_CONSTANTS 128

G_BEGIN_DECLS

// Opaque buffer handle. Kept by-pointer so callers can null-check
// in the same idiom as cl_mem. The tag-and-typedef split lets
// iop_api.h forward-declare just `typedef struct dt_vk_mem_t dt_vk_mem_t;`
// without pulling in <vulkan/vulkan.h>.
typedef struct dt_vk_mem_t dt_vk_mem_t;
struct dt_vk_mem_t
{
  VkBuffer       buffer;
  VkDeviceMemory memory;
  VkDeviceSize   size;
  bool           host_visible;
};

typedef struct dt_vk_kernel_t
{
  bool                  used;
  char                  name[64];
  int                   program; // index into device->programs
  VkShaderModule        shader_module;
  VkDescriptorSetLayout dset_layout;
  VkPipelineLayout      pipeline_layout;
  VkPipeline            pipeline;
  // Layout discovered at kernel-create time from the kernel signature
  // the IOP module registered (see dt_vulkan_create_kernel).
  uint32_t              num_storage_buffer_bindings;
  uint32_t              push_constant_size;
  uint32_t              local_size_x;
  uint32_t              local_size_y;
  uint32_t              local_size_z;
} dt_vk_kernel_t;

typedef struct dt_vk_program_t
{
  bool      used;
  char      name[128];
  uint32_t *spirv;     // owned
  size_t    spirv_words;
} dt_vk_program_t;

typedef struct dt_vk_device_t
{
  VkPhysicalDevice phys;
  VkDevice         device;
  VkQueue          queue;
  uint32_t         queue_family_index;
  VkCommandPool    cmd_pool;
  VkDescriptorPool dset_pool;
  char             name[256];
  VkPhysicalDeviceMemoryProperties mem_props;

  dt_vk_program_t  programs[DT_VULKAN_MAX_PROGRAMS];
  dt_vk_kernel_t   kernels [DT_VULKAN_MAX_KERNELS];

  // Persistent host-visible staging buffer reused across every
  // dt_vulkan_write_to_device / dt_vulkan_read_from_device call.
  // Grows monotonically (never shrinks) — large export pipelines
  // can reach ~1 GB here on a 60 MPx canvas. Allocating a 1 GB
  // VkDeviceMemory per dispatch was the dominant cost in early
  // pixelpipe-VK timings; the cache cuts the steady-state
  // overhead per module to a single vkCmdCopyBuffer.
  struct dt_vk_mem_t *staging;

  // Device-buffer pool. dt_vulkan_alloc_buffer / dt_vulkan_free_buffer
  // route through a small free-list of recently freed device-local
  // buffers; alloc returns the smallest-fit buffer ≥ requested size,
  // free pushes back to the list (up to DT_VULKAN_BUF_POOL_CAP). This
  // cuts the per-dispatch `vkAllocateMemory` / `vkCreateBuffer` churn
  // that dominates the steady-state HAL cost (§10.1 follow-up).
  // Lifetime is bound to the device — buffers are released at
  // dt_vulkan_cleanup. Access serialised by g_vk_lock together with
  // the rest of the dispatch path.
  struct dt_vk_mem_t *buf_pool[64];
  int                 buf_pool_count;

  // Reusable one-shot submission resources. _submit_one_shot used to
  // allocate a fresh VkCommandBuffer and VkFence per call (4-5 per
  // module dispatch — upload, kernel, readback). Reusing them via
  // vkResetCommandBuffer + vkResetFences saves the create/destroy
  // pair on every submission.
  VkCommandBuffer     oneshot_cmd;
  VkFence             oneshot_fence;

  // Monotonic count of successful vkQueueSubmit calls on this device
  // (eager dispatches, staging copies, and capture flushes alike).
  // Diagnostic only: lets the capture unit tests and the -d vkgraph
  // log verify that a captured span really collapsed into one
  // submission. Updated under the same g_vk_lock discipline all
  // submitters already follow.
  uint64_t            submit_count;
} dt_vk_device_t;

#define DT_VULKAN_BUF_POOL_CAP 64

typedef struct dt_vulkan_t
{
  gboolean         inited;
  gboolean         enabled;       // mirror of USE_VULKAN/runtime toggle
  VkInstance       instance;
  int              num_devs;
  dt_vk_device_t  *dev;           // dev[0..num_devs-1]
} dt_vulkan_t;

// ---- lifecycle --------------------------------------------------------

/** Initialise the Vulkan subsystem. After this call, dt_vulkan_running()
 *  returns TRUE iff at least one compute-capable device was found. */
void dt_vulkan_init(dt_vulkan_t *vk);

/** Tear down the subsystem. Safe to call on an un-inited or partially
 *  inited dt_vulkan_t. */
void dt_vulkan_cleanup(dt_vulkan_t *vk);

/** TRUE iff Vulkan is initialised, enabled in prefs, and has ≥1
 *  device. */
gboolean dt_vulkan_running(void);

/** Locks a device for the calling thread. Returns the device id, or -1
 *  if none is available. Mirrors dt_opencl_lock_device. */
int dt_vulkan_lock_device(void);
void dt_vulkan_unlock_device(int devid);

// ---- programs and kernels --------------------------------------------

/** Load a SPIR-V program from disk and register it under name. Returns
 *  a program index ≥0 on success, -1 on failure. */
int dt_vulkan_load_program(const char *name, const char *path);

/** Convenience: build the canonical kernel path
 *  `<datadir>/kernels/vulkan/<name>.spv` and call dt_vulkan_load_program.
 *  Returns -1 if Vulkan isn't running or the file is missing. */
int dt_vulkan_load_program_by_name(const char *name);

// ---- module helpers --------------------------------------------------
//
// Per-module wiring boilerplate (load .spv, create pipeline, free both)
// collapses to one helper call. Modules keep a `dt_vk_module_kernel_t`
// field per kernel slot in their global_data; init_global passes it to
// dt_vulkan_module_kernel_load; cleanup_global passes it to
// dt_vulkan_module_kernel_unload; process_vk dispatches via
// dt_vulkan_dispatch_inout.
//
// All helpers tolerate "Vulkan not running" — they zero-init the slot
// (kernel = -1) so a later dt_vulkan_dispatch_inout returns failure
// and the pixelpipe falls back to OpenCL/CPU transparently.

typedef struct dt_vk_module_kernel_t
{
  int program;  // -1 if not loaded
  int kernel;   // -1 if not loaded
} dt_vk_module_kernel_t;

#define DT_VK_MODULE_KERNEL_INIT { -1, -1 }

/** Load `<datadir>/kernels/vulkan/<spv_name>.spv` and create the kernel
 *  whose entry point is named `entry`. For the common case where the
 *  .spv file is named after the kernel, `spv_name == entry`. */
void dt_vulkan_module_kernel_load(dt_vk_module_kernel_t *out,
                                  const char *spv_name,
                                  const char *entry,
                                  uint32_t num_storage_buffers,
                                  uint32_t push_constant_size,
                                  uint32_t local_x,
                                  uint32_t local_y,
                                  uint32_t local_z);

/** Same as kernel_load above but reuses an already-loaded program
 *  instead of loading a fresh one. Useful for multi-kernel modules
 *  (channelmixerrgb's 5 adaptation modes, future demosaicers, …)
 *  where all kernels live in one .spv with multiple entry points.
 *  Returns 0 on success or -1 if Vulkan isn't running / pipeline
 *  build failed (typical reason: entry point not present, e.g. the
 *  GLSL fallback .spv only carries one of the kernels). */
void dt_vulkan_module_kernel_create_from(dt_vk_module_kernel_t *out,
                                         int program,
                                         const char *entry,
                                         uint32_t num_storage_buffers,
                                         uint32_t push_constant_size,
                                         uint32_t local_x,
                                         uint32_t local_y,
                                         uint32_t local_z);

/** Tear down the pipeline. Safe on an un-loaded slot. */
void dt_vulkan_module_kernel_unload(dt_vk_module_kernel_t *k);

/** Dispatch the common 2-buffer (in, out) shape with a push-constant
 *  blob. Returns 0 on success or -1 if Vulkan isn't running / the
 *  kernel isn't loaded / dispatch failed. */
int dt_vulkan_dispatch_inout(const dt_vk_module_kernel_t *k,
                             dt_vk_mem_t *dev_in,
                             dt_vk_mem_t *dev_out,
                             size_t global_w,
                             size_t global_h,
                             const void *push_constants,
                             size_t push_constant_size);

/** Dispatch with an extra constant-data buffer bound at binding 2
 *  (e.g. a tone-curve LUT). The caller owns the buffer's lifetime. */
int dt_vulkan_dispatch_inout_lut(const dt_vk_module_kernel_t *k,
                                 dt_vk_mem_t *dev_in,
                                 dt_vk_mem_t *dev_out,
                                 dt_vk_mem_t *dev_lut,
                                 size_t global_w,
                                 size_t global_h,
                                 const void *push_constants,
                                 size_t push_constant_size);

/** General N-binding dispatch — buffers[0..count-1] become storage-
 *  buffer descriptors 0..count-1. Used by modules that need more
 *  than the convenience 2/3-binding shapes. */
int dt_vulkan_dispatch_n(const dt_vk_module_kernel_t *k,
                         dt_vk_mem_t *const *buffers,
                         size_t buffer_count,
                         size_t global_w,
                         size_t global_h,
                         const void *push_constants,
                         size_t push_constant_size);

// One pre-dispatch upload: copy `size` bytes from `host` into device
// buffer `dst` (at offset 0) before the kernel runs.
typedef struct dt_vk_upload_t
{
  dt_vk_mem_t *dst;
  const void  *host;
  size_t       size;
} dt_vk_upload_t;

/** Bundled upload + dispatch in a single submit/wait. Equivalent to a
 *  sequence of dt_vulkan_write_to_device(...) calls followed by
 *  dt_vulkan_dispatch_n(...), but issued as one command buffer with
 *  an internal pipeline barrier between the transfer phase and the
 *  compute phase. Saves N submit/wait round-trips per module dispatch
 *  (typically 30–150 ms when N small uploads — LUTs, matrices,
 *  param structs — accompany each kernel call). The staging buffer
 *  is partitioned by offset so all uploads share one DMA region. */
int dt_vulkan_dispatch_n_batched(const dt_vk_module_kernel_t *k,
                                 dt_vk_mem_t *const *buffers,
                                 size_t buffer_count,
                                 const dt_vk_upload_t *uploads,
                                 size_t upload_count,
                                 size_t global_w,
                                 size_t global_h,
                                 const void *push_constants,
                                 size_t push_constant_size);

/** Create a kernel handle. The Vulkan model needs the binding shape up
 *  front (descriptor set layout); pass it here. Returns kernel index
 *  ≥0, or -1 on failure. */
int dt_vulkan_create_kernel(int program,
                            const char *entry,
                            uint32_t num_storage_buffer_bindings,
                            uint32_t push_constant_size,
                            uint32_t local_size_x,
                            uint32_t local_size_y,
                            uint32_t local_size_z);

void dt_vulkan_free_kernel(int kernel);

// ---- memory ----------------------------------------------------------

/** Allocate a device-local buffer of size bytes, usable as storage. */
dt_vk_mem_t *dt_vulkan_alloc_buffer(int devid, size_t size);

/** Free a buffer; safe on NULL. */
void dt_vulkan_free_buffer(int devid, dt_vk_mem_t *mem);

/** Copy size bytes from host memory into a device buffer, blocking. */
int dt_vulkan_write_to_device(int devid, dt_vk_mem_t *dst,
                              const void *host, size_t size);

/** Copy size bytes from a device buffer into host memory, blocking. */
int dt_vulkan_read_from_device(int devid, void *host,
                               const dt_vk_mem_t *src, size_t size);

/** Device-to-device buffer copy (size bytes from src@0 to dst@0).
 *  Used by pass-through modules and the "stage in / stage out" path
 *  in the pixelpipe when a Vulkan-resident buffer needs to feed an
 *  OpenCL/CPU module further down the chain. Blocking. */
int dt_vulkan_copy_device_to_device(int devid,
                                    dt_vk_mem_t *dst,
                                    const dt_vk_mem_t *src,
                                    size_t size);

/** Sub-region 2-D copy from `src` to `dst`. Maps onto a single
 *  `vkCmdCopyBuffer` with `region_h` `VkBufferCopy` entries (one
 *  per row). The src/dst buffers are interpreted as row-major
 *  pixel grids with `*_row_pixels` pixels per row and
 *  `bytes_per_pixel` bytes per pixel; `*_offset_{x,y}` are the top-
 *  left of the rectangle in each. Used by rasterfile, and a building
 *  block for future ports that need cropped / shifted-origin copies. */
int dt_vulkan_copy_subregion(int devid,
                             dt_vk_mem_t *dst,
                             const dt_vk_mem_t *src,
                             size_t src_offset_x, size_t src_offset_y,
                             size_t dst_offset_x, size_t dst_offset_y,
                             size_t region_w, size_t region_h,
                             size_t src_row_pixels, size_t dst_row_pixels,
                             size_t bytes_per_pixel);

// ---- capture / deferred graph execution (DAG milestone M1) -----------
//
// See dev-doc/gpu_resident_pixelpipe_dag.md §5. When a capture context
// is active on the calling thread, the dispatch and copy entry points
// above stop submitting one-shot command buffers and instead append
// *nodes* to a per-thread graph:
//
//   dispatch_*            -> DISPATCH node (push constants snapshotted)
//   write_to_device       -> UPLOAD node (payload snapshotted — module
//                            LUTs/matrices live in stack arrays that
//                            die before the flush)
//   write_to_device_borrowed -> UPLOAD node borrowing the host pointer
//                            (caller guarantees it outlives the flush)
//   copy_device_to_device -> COPY node
//   copy_subregion        -> COPY node (row regions built at record)
//   free_buffer           -> deferred until the next flush, since
//                            pending nodes may still reference the
//                            buffer (naive M1 liveness; the M3 planner
//                            replaces this with interval aliasing)
//   read_from_device      -> SYNC TAP: flushes the pending segment,
//                            waits, then serves the read eagerly —
//                            mid-module host readbacks keep working
//                            unmodified, at eager-path cost
//
// dt_vulkan_capture_flush records every pending node into ONE command
// buffer — a leading staging map/memcpy for all uploads, pipeline
// barriers between dependent nodes, one descriptor pool for the whole
// segment — submits once and waits one fence. Node capture itself
// touches no device state and needs no lock; begin/mark/rollback are
// lock-free too. flush/end/abort/rollback release or submit device
// resources and follow the same rule as the eager entry points: the
// caller holds the device lock (dt_vulkan_lock_device).
//
// Error contract: a failed capture append returns -1 from the same
// entry point that would have failed eagerly, with the context wound
// back to its pre-call state — the caller's existing fallback path
// (module returns -1, pixelpipe rolls back and goes CPU) works
// unchanged. A failed flush drops the segment and returns -1; buffer
// contents produced by the dropped nodes are undefined, matching the
// eager path's contract after a failed dispatch.

typedef struct dt_vk_capture_mark_t
{
  uint32_t nodes;   // node count at mark time
  uint32_t dfrees;  // deferred-free count at mark time
  size_t   ring;    // snapshot-ring bytes used at mark time
  uint32_t taps;    // registered-tap count at mark time
} dt_vk_capture_mark_t;

/** Enter capture mode on the calling thread. Returns FALSE (and stays
 *  in eager mode) if Vulkan isn't running, devid is invalid, or a
 *  capture is already active on this thread. */
gboolean dt_vulkan_capture_begin(int devid);

/** TRUE iff the calling thread has an active capture context. */
gboolean dt_vulkan_capture_active(void);

/** Number of nodes captured but not yet flushed (0 outside capture). */
uint32_t dt_vulkan_capture_pending(void);

/** M3 liveness (gpu_resident_pixelpipe_dag.md §5.5): peak simultaneous
 *  live device memory, in bytes, across the flushes of the current (or
 *  most recently ended) capture session on this thread. Each captured
 *  buffer is live across [first, last] node that references it; this is
 *  the max over node positions and over the session's segments — the
 *  memory the deferred-free model keeps resident, i.e. the figure a
 *  future interval-aliasing planner (and spill decision) works from.
 *  0 outside capture / before any flush. Diagnostic and test-facing;
 *  also emitted per segment under `-d vkgraph`. */
uint64_t dt_vulkan_capture_peak_bytes(void);

/** Submit the pending segment (one command buffer, one fence wait),
 *  then execute deferred frees. Capture stays active — subsequent HAL
 *  calls start a new segment. No-op segment (0 nodes) skips the
 *  submit entirely. Returns 0 on success. Caller holds the device
 *  lock. */
int dt_vulkan_capture_flush(int devid);

/** Flush, then leave capture mode. Returns the flush result. Caller
 *  holds the device lock. */
int dt_vulkan_capture_end(int devid);

/** Drop all pending nodes WITHOUT executing them, run deferred frees,
 *  leave capture mode. For error/shutdown paths where the results are
 *  being abandoned. Caller holds the device lock. */
void dt_vulkan_capture_abort(int devid);

/** Checkpoint for module-granular rollback. Take a mark before
 *  calling a module's process_vk under capture; if the module fails,
 *  dt_vulkan_capture_rollback truncates the graph back to the mark
 *  (dropping the module's nodes and snapshots) and immediately frees
 *  buffers whose free was deferred after the mark — they can no
 *  longer be referenced. Lock-free to take; rollback needs the
 *  device lock (it releases buffers). */
dt_vk_capture_mark_t dt_vulkan_capture_mark(void);
void dt_vulkan_capture_rollback(int devid, const dt_vk_capture_mark_t *mark);

/** Register a GPU tap (DAG M5, gpu_resident_pixelpipe_dag.md §5.4): a
 *  small device result buffer whose readback is deferred to the end of
 *  the run instead of forcing a mid-span flush. Under an active
 *  capture the registry takes ownership of `buf` and, at
 *  dt_vulkan_capture_end, reads `size` bytes from it into `host_dst`
 *  (after the final fence, so the producing dispatch has executed)
 *  and frees it; on dt_vulkan_capture_abort or a rollback past this
 *  point the buffer is freed without a read. Returns TRUE when
 *  registered. Returns FALSE outside capture — the caller should then
 *  read `buf` back itself (eager path). Caller holds the device lock. */
gboolean dt_vulkan_tap_register(int devid, dt_vk_mem_t *buf,
                                void *host_dst, size_t size);

/** Like dt_vulkan_write_to_device, but under capture the host pointer
 *  is borrowed instead of snapshotted: the caller guarantees it stays
 *  valid and unmodified until the next flush/end. Outside capture the
 *  two calls are identical. For trunk-sized uploads whose lifetime
 *  the caller controls — snapshotting those would double peak host
 *  memory. */
int dt_vulkan_write_to_device_borrowed(int devid, dt_vk_mem_t *dst,
                                       const void *host, size_t size);

/** Diagnostic: successful vkQueueSubmit count on this device so far.
 *  Tests assert an N-dispatch captured span submits once; the eager
 *  path submits ≥N times. */
uint64_t dt_vulkan_submission_count(int devid);

// ---- dispatch --------------------------------------------------------

/** Bind storage buffers (count must match the kernel's registered
 *  binding count) and dispatch with the given push-constant blob and
 *  global work size. Synchronous: returns after the queue idles. */
int dt_vulkan_enqueue_kernel_2d(int devid, int kernel,
                                size_t global_w, size_t global_h,
                                dt_vk_mem_t *const *buffers,
                                size_t buffer_count,
                                const void *push_constants,
                                size_t push_constant_size);

G_END_DECLS

#else // HAVE_VULKAN

// Stub types so callers that include vulkan.h unconditionally still
// compile when USE_VULKAN=OFF. The real definitions live above.
struct dt_vk_mem_t;
typedef struct dt_vk_mem_t dt_vk_mem_t;
typedef struct dt_vulkan_t dt_vulkan_t;
// Modules always have `dt_vk_module_kernel_t vk_*` fields in their
// global_data structs (no preprocessor stunts there). When the
// backend is compiled out we just give them empty slots; the helpers
// below resolve to no-ops or constant-fail returns.
typedef struct dt_vk_module_kernel_t { int _unused; } dt_vk_module_kernel_t;
#define DT_VK_MODULE_KERNEL_INIT { 0 }
static inline void dt_vulkan_module_kernel_load(dt_vk_module_kernel_t *o,
                                                const char *n, const char *e,
                                                uint32_t a, uint32_t b,
                                                uint32_t c, uint32_t d, uint32_t f)
{ (void)o; (void)n; (void)e; (void)a; (void)b; (void)c; (void)d; (void)f; }
static inline void dt_vulkan_module_kernel_create_from(dt_vk_module_kernel_t *o,
                                                       int p, const char *e,
                                                       uint32_t a, uint32_t b,
                                                       uint32_t c, uint32_t d, uint32_t f)
{ (void)o; (void)p; (void)e; (void)a; (void)b; (void)c; (void)d; (void)f; }
static inline void dt_vulkan_module_kernel_unload(dt_vk_module_kernel_t *k) { (void)k; }
static inline int dt_vulkan_dispatch_inout(const dt_vk_module_kernel_t *k,
                                           dt_vk_mem_t *i, dt_vk_mem_t *o,
                                           size_t w, size_t h,
                                           const void *pc, size_t pcs)
{ (void)k; (void)i; (void)o; (void)w; (void)h; (void)pc; (void)pcs; return -1; }
static inline int dt_vulkan_dispatch_inout_lut(const dt_vk_module_kernel_t *k,
                                               dt_vk_mem_t *i, dt_vk_mem_t *o,
                                               dt_vk_mem_t *l,
                                               size_t w, size_t h,
                                               const void *pc, size_t pcs)
{ (void)k; (void)i; (void)o; (void)l; (void)w; (void)h; (void)pc; (void)pcs; return -1; }
static inline int dt_vulkan_dispatch_n(const dt_vk_module_kernel_t *k,
                                       dt_vk_mem_t *const *b, size_t bc,
                                       size_t w, size_t h,
                                       const void *pc, size_t pcs)
{ (void)k; (void)b; (void)bc; (void)w; (void)h; (void)pc; (void)pcs; return -1; }

typedef struct dt_vk_upload_t { int _unused; } dt_vk_upload_t;
static inline int dt_vulkan_dispatch_n_batched(const dt_vk_module_kernel_t *k,
                                               dt_vk_mem_t *const *b, size_t bc,
                                               const dt_vk_upload_t *u, size_t uc,
                                               size_t w, size_t h,
                                               const void *pc, size_t pcs)
{ (void)k; (void)b; (void)bc; (void)u; (void)uc; (void)w; (void)h; (void)pc; (void)pcs; return -1; }

static inline int dt_vulkan_copy_device_to_device(int devid, dt_vk_mem_t *d,
                                                  const dt_vk_mem_t *s, size_t sz)
{ (void)devid; (void)d; (void)s; (void)sz; return -1; }

static inline int dt_vulkan_copy_subregion(int devid, dt_vk_mem_t *d,
                                           const dt_vk_mem_t *s,
                                           size_t sox, size_t soy,
                                           size_t dox, size_t doy,
                                           size_t rw, size_t rh,
                                           size_t srp, size_t drp, size_t bpp)
{ (void)devid; (void)d; (void)s; (void)sox; (void)soy; (void)dox; (void)doy;
  (void)rw; (void)rh; (void)srp; (void)drp; (void)bpp; return -1; }

// Capture API stubs (see the HAVE_VULKAN branch for semantics).
typedef struct dt_vk_capture_mark_t { int _unused; } dt_vk_capture_mark_t;
static inline gboolean dt_vulkan_capture_begin(int devid) { (void)devid; return FALSE; }
static inline gboolean dt_vulkan_capture_active(void) { return FALSE; }
static inline uint32_t dt_vulkan_capture_pending(void) { return 0; }
static inline uint64_t dt_vulkan_capture_peak_bytes(void) { return 0; }
static inline int dt_vulkan_capture_flush(int devid) { (void)devid; return 0; }
static inline int dt_vulkan_capture_end(int devid) { (void)devid; return 0; }
static inline void dt_vulkan_capture_abort(int devid) { (void)devid; }
static inline dt_vk_capture_mark_t dt_vulkan_capture_mark(void)
{ dt_vk_capture_mark_t m = { 0 }; return m; }
static inline void dt_vulkan_capture_rollback(int devid, const dt_vk_capture_mark_t *mark)
{ (void)devid; (void)mark; }
static inline gboolean dt_vulkan_tap_register(int devid, dt_vk_mem_t *buf,
                                              void *host_dst, size_t size)
{ (void)devid; (void)buf; (void)host_dst; (void)size; return FALSE; }
static inline int dt_vulkan_write_to_device_borrowed(int devid, dt_vk_mem_t *dst,
                                                     const void *host, size_t size)
{ (void)devid; (void)dst; (void)host; (void)size; return -1; }
static inline uint64_t dt_vulkan_submission_count(int devid) { (void)devid; return 0; }

static inline void dt_vulkan_init(dt_vulkan_t *vk)    { (void)vk; }
static inline void dt_vulkan_cleanup(dt_vulkan_t *vk) { (void)vk; }
static inline gboolean dt_vulkan_running(void)        { return FALSE; }

#endif // HAVE_VULKAN
