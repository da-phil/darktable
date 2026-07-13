# A GPU-resident pixelpipe: DAG execution behind the linear pipe

**Status:** in implementation — M1 (capture HAL + segment executor),
M0 (VK blend kernels, uniform-mask blending on-device), and M2 (run-
scoped span capture with deferred readbacks + the module-input
colorspace glue node) landed; a `colorin → … → colorout` span now
captures as a single GPU submission. M5 in progress: the GPU histogram
reduction kernel and the deferred-readback tap registry are landed and
validated (pipe wiring pending a darkroom bench). See §11 for the
living implementation log and §9 for per-milestone state. §1–§10 are the
design and stay authoritative; deviations carry inline status notes.
**Scope:** when Vulkan is available, run the whole pixelpipe on the GPU as
one scheduled unit — no CPU↔GPU pixel transfers between iops, no
per-module queue synchronisation — while keeping darktable's linear
module list, history stack, and darkroom UI exactly as they are.
**Companion docs:**
[`gpu_acceleration_clspv_vulkan.md`](gpu_acceleration_clspv_vulkan.md)
(the clspv/Vulkan kernel + HAL migration this builds on) and
[`pixelpipe_architecture.md`](pixelpipe_architecture.md) (the current
pipe).
**Prior art:** vkdt's processing graph,
<https://github.com/hanatos/vkdt/blob/master/src/pipe/readme.md>.

---

## 1. Problem statement

The Vulkan work so far (see the companion doc) ports *kernels* and
*modules*. It deliberately did not change the pixelpipe's execution
model, which is eager and module-at-a-time:

```
for each module (recursion unwind order):
    get input          (host buffer, or cl_mem/dt_vk_mem_t if lucky)
    alloc output
    transform colorspace (sometimes in place, sometimes on host)
    process kernel(s)
    collect histogram / picker taps       (host readback)
    transform for blend                   (host, in place)
    blend                                 (CPU for the VK arm)
    sync the queue                        (per module)
    maybe write input back to host cache  (per module, screen pipes)
```

That model made sense when the GPU was an *accelerator bolted onto a
CPU pipeline*. It is the wrong model when the goal is a GPU-*resident*
pipeline, and no amount of per-module optimisation gets there, because
the costs are structural. Concretely, on the RX 9060 XT trace in
companion §10.2, a full export of a 5208×3472 image (≈289 MB as
float4) spends ~15 s, of which almost everything is data motion and
CPU work *between* kernels — the kernels themselves account for well
under a second.

### 1.1 Inventory of host touch points in today's run

Verified against `src/develop/pixelpipe_hb.c` as of this writing.
"Trunk" below means the main image buffer flowing module to module.

| # | Touch point | Where | Cost class |
|---|---|---|---|
| 1 | Per-module queue sync. Even the pure-OpenCL chain calls `dt_opencl_finish_sync_pipe()` after every module; `asyncmode` defaults to FALSE and exports always sync. | `_dev_pixelpipe_process_rec` CL arm | latency × #modules |
| 2 | VK host staging: upload input / read back output around each `process_vk`, mitigated only pairwise by the §5.14 hand-off cache. | `_pixelpipe_process_on_CPU` VK hook | 2 × trunk size × #VK modules (worst case) |
| 3 | CL↔VK boundary: full trunk `clEnqueueReadImage` + VK upload at every backend switch. | routing heuristic, companion §10.2 | 1–2 s per export |
| 4 | Blending after a VK module runs on CPU (`dt_develop_blend_process`), forcing readback + re-upload; the CL arm has `dt_develop_blend_process_cl` but the VK arm has no GPU blend. | `_pixelpipe_process_on_CPU` | 3–5 s per export |
| 5 | Colorspace glue: `dt_ioppr_transform_image_colorspace()` mutates the *host* input in place on the CPU arm (and invalidates the cacheline); blend-space transforms likewise. The CL arm at least does it in `cl_mem`. Each in-place host mutation also kills the VK hand-off (the §5.14 host-mutation invariant). | both arms | trunk-sized CPU passes |
| 6 | Histogram / color-picker taps copy the module's in/out image to a host scratch buffer and reduce on the CPU (`_histogram_collect_cl`, `_pixelpipe_picker_cl` "abuse the empty output buffer on host"). | per requesting module | trunk readback per tap |
| 7 | Pixelpipe cache writes: on screen pipes the focused module's *input* is copied device→host every run so the host cache can serve the next edit (`important_cl_input` block). | CL arm success path | trunk readback per run |
| 8 | Raster masks and the scharr/detail mask live in host memory (`piece->raster_masks` hash table, `pipe->scharr.data`); distortion walks between producer and consumer run on the CPU. | `dt_dev_get_raster_mask`, `dt_dev_write_scharr_mask` | mask-sized CPU work + sync |
| 9 | CPU-only modules (`toneequal`, …) and runtime fallbacks (lcms2 colorout, unported demosaic modes) split any GPU chain with a full round-trip. | dispatch arms | 2 × trunk per island |
| 10 | Final backcopy of the result to host (`_dev_pixelpipe_process_rec_and_backcopy`), then 8-bit conversion for the screen backbuf. | pipe tail | one trunk readback (unavoidable today) |

Row 2 has been whittled down by the hand-off cache, batched uploads,
and the buffer pool; rows 1 and 3–9 are *inherent to the execution
model*: as long as the pipe executes modules one at a time and the
canonical intermediate state lives in host memory, every one of these
stays.

### 1.2 The ceiling of the incremental lane

The companion doc's §10.2 paths (port blendop, extend chains, DMA-BUF
interop, split locks) all reduce the *number* of boundary crossings.
They are worth doing and this design builds on several of them. But
even in the limit — every module ported, blend on GPU, no CL in the
pipe — the eager model still: synchronises per module (row 1), cannot
alias or pre-plan buffer memory (every module allocates its own in/out
pair from a pool), re-uploads LUTs/params one module at a time, taps
host-side (rows 6–8), and writes the host cache synchronously (row 7).
vkdt demonstrates what the same hardware can do when the whole frame
is planned as one unit: full RAW → display graphs run in a handful of
milliseconds because *nothing* leaves VRAM and the driver sees one
submission, not eighty.

That is the gap this document is about.

---

## 2. What vkdt actually does (and what applies to darktable)

Short version of `hanatos/vkdt/src/pipe/readme.md` plus reading its
`graph.c`/`alloc.c`, filtered through "what can dt reuse":

- **Two layers.** *Modules* are the user-visible processing blocks
  wired by connectors into a DAG. Each module expands into one or more
  *nodes*; a node is exactly one compute shader dispatch. The node
  layer is invisible to users.
- **Graph compile per run.** The module DAG is topologically sorted;
  ROIs are negotiated in two passes (sources push full extents
  downstream, sinks pull requested regions upstream — the exact
  analogue of dt's `modify_roi_out` forward walk in
  `dt_dev_pixelpipe_get_dimensions()` and the `modify_roi_in`
  backward walk in `_dev_pixelpipe_process_rec()`).
- **Memory is planned, not allocated per node.** All intermediate
  images live in one big `vkAllocateMemory` arena. A liveness interval
  is computed for every buffer (first writer → last reader) and a
  greedy allocator packs buffers into the arena so that buffers with
  disjoint lifetimes share the same physical bytes. Peak VRAM is set
  by the *widest point of the graph*, not by the number of nodes.
- **One command buffer.** All dispatches are recorded with pipeline
  barriers between dependent nodes; the queue is submitted once and
  waited once per frame. Parameters live in uniform memory; push
  constants are used sparingly.
- **Run flags for incrementality.** A change is classified
  (params-only / roi / topology) and only the necessary phases re-run:
  upload params + resubmit, vs re-negotiate ROIs, vs rebuild
  nodes + realloc.
- **Sinks are graph nodes.** Display-out, histogram, color picker are
  nodes writing GPU buffers; the CPU reads back only tiny results.

**What we deliberately do *not* adopt:** the user-facing graph (dt's
UI, history stack, presets, and iop-order semantics stay linear and
untouched); the `.cfg` pipeline description files; GLSL as kernel
language (our kernels stay OpenCL C via clspv); feedback connectors /
animation; vkdt's parameter and UI systems.

**The key observation that makes adoption natural:** darktable's pipe
is *already a DAG semantically* — the linear module list is only its
trunk. The hidden edges are:

- every blended module is a *diamond*: its input feeds both its
  kernel(s) and the blend node that mixes kernel output with that same
  input (`dt_develop_blend_process*` takes `input` and `output`);
- raster masks are long-range edges from a producer module's blend
  stage to consumer modules further down, passing through per-module
  `distort_mask` transforms on the way;
- the scharr/detail mask is an edge from demosaic/rawprepare to every
  module that blends on details;
- colorspace conversions are implicit glue nodes between adjacent
  modules whenever `input_colorspace()` disagrees with the current
  buffer state, plus around blending (`_transform_for_blend`);
- histogram/picker/scope taps are side sinks hanging off specific
  module boundaries;
- the pixelpipe cache is a set of optional taps on trunk edges.

Today all of these edges are executed implicitly, eagerly, and mostly
through host memory. The proposal is to *reify* them: build the DAG
explicitly at run time — behind the unchanged linear pipe — plan its
memory, record it as very few Vulkan submissions, and keep every edge
in VRAM unless something genuinely needs host bytes.

---

## 3. Goals and non-goals

Goals, in priority order:

1. **Zero trunk-sized host transfers** in the steady state for a pipe
   whose modules are all VK-capable: pixels cross PCIe exactly twice —
   source upload and final sink readback (plus tiny tap results).
2. **One (or few) queue submissions per pipe run**; barriers instead
   of fences between modules; no per-module `vkQueueWaitIdle`
   semantics anywhere.
3. **Planned VRAM**: single arena per pipe context, liveness-based
   aliasing, `VK_EXT_memory_budget`-aware; graceful degradation when
   the arena doesn't fit.
4. **Keep darkroom interactivity mechanisms intact or better**: the
   pixelpipe cache's "edit module N, reprocess only N..end" property
   must survive, ideally upgraded to "…without re-uploading N's input".
5. **No flag-day for modules**: the 73 ported `process_vk`
   implementations must keep working unmodified on day one of the
   graph executor (they may run suboptimally until touched).
6. **Bit-identical output** to the current VK arm for the same module
   set (the existing lavapipe equality harness applies).

Non-goals:

- Changing what users see. No graph UI, no new history semantics, no
  new iop-order rules. (vkdt exposes its graph; we explicitly don't.)
- Multi-image / animation graphs, feedback edges.
- Replacing the CPU path or the OpenCL path during the transition —
  both remain as fallbacks (see §7).
- GPU-side mask *drawing* (cairo rasterisation of drawn shapes stays
  on CPU; the resulting mask planes are small uploads).

---

## 4. Candidate approaches

Four were considered. B is recommended, with C as the designated
end-state for the handful of modules that outgrow B.

### Approach A — keep the eager model, keep optimising boundaries

Continue the companion §10.2 lanes: port blendop to VK, extend chains,
DMA-BUF interop with OpenCL, split `g_vk_lock`. No graph IR.

*Why it's not enough:* §1.2. It shrinks rows 3–4 of the inventory but
rows 1, 5–8 are load-bearing parts of the eager design. It also keeps
compounding complexity in `pixelpipe_hb.c` — the §5.14 host-mutation
invariant is an example of the whack-a-mole this model forces: every
new host-side in-place mutation anywhere in the 4200-line file is a
potential silent corruption of the hand-off cache. A planned graph
eliminates that class of bug by construction (buffers are never
mutated behind the scheduler's back; glue transforms write new
temporaries).

Verdict: keep as the *fallback path* and as useful groundwork (GPU
blendop kernels are needed by B anyway), but it is not the
destination.

### Approach B — deferred command-graph capture behind the pixelpipe (recommended)

Keep the linear walk in `pixelpipe_hb.c` exactly where it is, but give
it a second mode: instead of executing each module's GPU work eagerly,
*capture* it. The pipe walk becomes the graph builder:

- module host code (`process_vk`) still runs eagerly, in pipe order,
  exactly once per run — it computes params, builds LUTs, mutates
  `pipe->dsc` sequentially, and calls the same HAL entry points it
  calls today;
- the HAL dispatch/copy/upload calls, when a capture context is
  active, *append nodes to a graph IR* instead of submitting work;
- glue work the pipe does between modules (colorspace transforms,
  blending, taps, cache writes) is appended as explicit nodes by the
  pipe itself;
- after the walk, the graph is memory-planned, recorded into one or
  few command buffers, submitted, and awaited once.

The decisive property: **capture reuses the imperative module code we
already have.** An audit of `src/iop/` shows every ported module goes
through a closed set of HAL entry points — the dispatch shapes
(`dt_vulkan_dispatch_n` ×71, `_n_batched` ×39, `_inout` ×32 call
sites), the copies (`dt_vulkan_copy_device_to_device` ×23,
`dt_vulkan_copy_subregion` ×1), buffer lifetime
(`dt_vulkan_alloc_buffer`/`_free_buffer`), uploads
(`dt_vulkan_write_to_device` ×14 plus the `dt_vk_upload_t` batches),
and a handful of mid-module readbacks (`dt_vulkan_read_from_device`,
7 modules — see the sync-tap discussion in §5.3). No module touches
`VkBuffer`/`vkCmd*` directly or dereferences `dt_vk_mem_t` internals,
and the multi-kernel helpers (gaussian, bilateral, guided filter,
dwt, local-laplacian, resampler) bottom out in the same calls.
Hooking capture at that boundary converts every existing port into a
graph citizen for free.

### Approach C — declarative node API (vkdt-style `create_nodes`)

Modules describe their nodes and connectors declaratively; the
pipeline is graph-native; ROIs can differ per node; multi-scale
pyramids (diffuse, local-laplacian, wavelets) become real sub-graphs
instead of host-orchestrated dispatch loops.

*Assessment:* this is the better long-term shape for the ~10 complex
modules whose host orchestration is itself expensive or whose
intermediate pyramid levels would benefit from planner-visible
lifetimes (`diffuse`'s 6 kernels × 10 scales × N iterations, demosaic
variants, exposure-fusion basecurve, filmicrgb reconstruction). But as
the *entry* strategy it would mean rewriting 70+ working ports before
seeing any benefit.

*Resolution:* C is layered **on top of** B, opt-in per module. A
captured B-graph and a declared C-subgraph are the same IR after the
build phase; a module that implements the (future, optional)
`create_nodes_vk` callback simply skips capture. No module is forced
to migrate.

### Approach D — adopt an existing engine (vkdt's pipe, or clvk underneath)

Running dt's processing on vkdt's engine founders on kernels (GLSL vs
our OpenCL C), parameter/history semantics, and module ecosystem; we
would inherit a whole second product's core. clvk (OpenCL ICD over
Vulkan) solves *backend unification* — one driver stack — but keeps
the OpenCL host model, i.e. the eager queue with per-module logic; it
does nothing about rows 1, 4–8. Both rejected for this purpose; clvk
remains interesting as a macOS bridge per companion §11, orthogonal to
this design.

### Comparison

| | A (eager++) | **B (capture graph)** | C (declared graph) | D (foreign engine) |
|---|---|---|---|---|
| Removes per-module sync (row 1) | no | **yes** | yes | clvk: no |
| Zero trunk host traffic (rows 2–5,7) | no | **yes** | yes | no |
| GPU taps (rows 6,8) | bolt-on | **yes, as nodes** | yes | n/a |
| Planned/aliased VRAM | no | **yes** | yes | vkdt: yes |
| Module rewrite required | none | **none** | all | all + kernels |
| `pixelpipe_hb.c` complexity | grows | **shrinks over time** | replaced | replaced |
| Incremental landing | yes | **yes** | poor | no |

---

## 5. Architecture (approach B)

### 5.1 The IR

A small, flat, per-pipe-context structure. Sketch (names indicative):

```c
typedef int32_t dt_vkg_res_t;   // virtual resource id, -1 = none

typedef enum dt_vkg_node_kind_t
{
  DT_VKG_NODE_DISPATCH,   // one compute dispatch (kernel, push const blob)
  DT_VKG_NODE_COPY,       // vkCmdCopyBuffer (d2d, subregion rows)
  DT_VKG_NODE_UPLOAD,     // staging-ring slice -> device buffer
  DT_VKG_NODE_READBACK,   // device buffer -> staging slice (tap/sink)
  DT_VKG_NODE_FILL,       // vkCmdFillBuffer (zeroing grids etc.)
  DT_VKG_NODE_SYNC_TAP,   // capture-time flush point (see 5.3)
} dt_vkg_node_kind_t;

typedef struct dt_vkg_node_t
{
  dt_vkg_node_kind_t kind;
  int kernel;                          // DISPATCH: HAL kernel index
  uint32_t gw, gh;                     // global size
  uint8_t push[DT_VULKAN_MAX_PUSH_CONSTANTS];
  uint32_t push_size;
  dt_vkg_res_t res[DT_VULKAN_MAX_BINDINGS]; // bindings / copy src+dst
  uint32_t nres;
  uint32_t read_mask, write_mask;      // which res[] are read vs written
  // provenance for logs/debug: module op, instance, phase tag
  const char *tag;
} dt_vkg_node_t;

typedef struct dt_vkg_res_desc_t
{
  size_t size;
  int32_t first_node, last_node;  // liveness interval (filled by planner)
  size_t arena_offset;            // filled by planner
  uint32_t flags;                 // PINNED (residency cache), EXTERNAL (source/sink),
                                  // HOSTVIS (tap results), TRANSIENT
  dt_hash_t content_hash;         // for residency cache entries
} dt_vkg_res_desc_t;

typedef struct dt_vkg_graph_t
{
  GArray *nodes;                  // dt_vkg_node_t
  GArray *res;                    // dt_vkg_res_desc_t
  // capture-time staging ring for upload payloads (LUTs, matrices,
  // params structs, mask planes): appended during capture, flushed
  // as one transfer batch at submit
  uint8_t *staging_ring; size_t staging_used, staging_cap;
  // segments: [start,end) node ranges split at SYNC_TAPs, CPU
  // islands, and (optionally) cancellation checkpoints
  GArray *segments;
  dt_hash_t topology_hash;        // for plan reuse (5.8)
} dt_vkg_graph_t;
```

Notes:

- **Buffer-only, matching the HAL.** Resources are storage buffers.
  If/when milestone §8.5 (images + samplers) lands in the HAL, a
  resource gains an image variant; nothing in the IR changes shape.
- **`dt_vk_mem_t` becomes the virtual handle in capture mode.** In
  graph mode `dt_vulkan_alloc_buffer()` returns a `dt_vk_mem_t*` whose
  backing is not a `VkBuffer` but a `dt_vkg_res_t`. Module code cannot
  tell the difference (it never dereferences the struct's internals —
  verified: modules treat it as opaque). `dt_vulkan_free_buffer()`
  ends the resource's liveness instead of freeing memory.
- **Trunk edges are just resources** created by the pipe walk itself:
  module N's output resource is module N+1's input resource.
- **Upload payload snapshotting is mandatory.** Modules build LUTs and
  matrices in stack arrays that die when `process_vk` returns, so
  `dt_vulkan_write_to_device()` / `dt_vk_upload_t` batches must copy
  the bytes into the graph's staging ring *at capture time*. This is
  the same partitioned-staging scheme `dt_vulkan_dispatch_n_batched`
  already implements — generalised from per-dispatch to per-graph.
- Push-constant blobs are snapshotted into the node (≤128 B each).

**M1 status note (landed).** The shipped M1 IR
(`src/common/vulkan.c`, `_capture_node_t` / `dt_vk_capture_ctx_t`) is
one notch simpler than the sketch above: nodes reference *real*
`dt_vk_mem_t` buffers allocated from the existing pool exactly as in
eager mode, and buffer-reuse correctness is guaranteed by deferring
`dt_vulkan_free_buffer` to the segment flush instead of by liveness
analysis (an eager free could destroy the `VkBuffer` under pending
nodes when the pool is full, or hand it to a concurrent eager thread).
Virtual resource ids and interval aliasing arrive with the M3 planner;
the node shape doesn't change for it. Trunk-sized caller-owned uploads
get `dt_vulkan_write_to_device_borrowed` so they aren't duplicated
into the ring; module-owned payloads keep the safe snapshot default.

### 5.2 Plan phase: partitioning the pipe

`dt_dev_pixelpipe_process()` gains a graph mode gate (per pipe run):

```
vk_graph_possible = dt_vulkan_running()
                 && pipe has no CL-only segments the planner can't bridge
                 && conf: pixelpipe_vulkan_graph=true
```

Before the walk, a cheap linear pre-pass over `pipe->nodes` classifies
every enabled piece:

- `VK` — `process_vk && process_vk_ready` (the predictive
  `commit_params` gating from companion §10.2 already makes this flag
  trustworthy);
- `CPU` — everything else (including CL-only modules while the
  transition lasts — see §7).

Consecutive `VK` pieces form **GPU spans**; `CPU` pieces form **CPU
islands**. The graph executor runs whenever there is at least one span
of length ≥ 2 (degenerate case: a single-module span is exactly
today's behaviour, so nothing is lost). Trunk buffers cross island
boundaries via *planned* staged copies (§5.9) — the same cost the
eager model pays there, but paid once and overlappable.

This subsumes today's `vk_chain_ahead` look-ahead heuristic: routing
stops being a per-module bet (with the mis-fire modes §10.2 documents)
and becomes a whole-pipe plan computed from the same
`process_vk_ready` bits.

### 5.3 Capture phase

The existing recursion (`_dev_pixelpipe_process_rec`) runs unchanged
in structure — same order, same `modify_roi_in` negotiation, same
cache-hash computation, same shutdown checks. Differences inside a GPU
span:

1. The pipe binds a **capture context** to the thread
   (`dt_vulkan_capture_begin(graph)`); every HAL dispatch/copy/upload
   call made from here appends nodes instead of submitting. Capture
   context is thread-local state in `vulkan.c`, so *no module
   signature changes*.
2. `process_vk` runs eagerly as host code (params, LUT building,
   `pipe->dsc` mutation — all sequential exactly as today) while its
   GPU calls are captured.
3. The pipe appends **glue nodes** itself (§5.4) where it currently
   calls host/CL glue.
4. On leaving the span (CPU island, pipe end, or capture fault), the
   pipe calls `dt_vulkan_capture_end()` and the graph goes to
   plan/record/submit (§5.5–5.6).

**The sync-tap escape hatch.** Seven `process_vk` implementations
call `dt_vulkan_read_from_device` mid-module today, in three shapes:
*trunk statistics on GUI-cache miss, with preview-pipe priming*
(`hazeremoval` ambient light, `globaltonemap` lwmax, `colormapping`
cluster training, `ashift` parameter-fitting snapshot — the preview
run computes and caches, full/export runs hit the cache), *small
reduction results* (`denoiseprofile` reads back a per-tile sum buffer
of a few KB — cheap under any model), and *host-side processing of
the trunk* (`retouch`'s heal/clone solvers, `overexposed`'s lcms2
branch). Under capture, a device→host read of a *not-yet-executed*
resource cannot be served lazily. The recorder handles it
transparently: it ends the current segment, plans/records/submits
nodes so far, waits the fence, services the read, and resumes capture
in a new segment. Semantics identical to today; cost identical to
today; and — crucially — the priming-pattern modules only pay it on
the small preview pipe, so in the full/export pipes sync-taps fire
rarely. The capture API makes the degradation *visible* (a `SYNC_TAP`
node in the graph dump) instead of silent.

The same mechanism cleanly handles anything unexpected: a module doing
something the recorder can't defer simply costs a segment split, never
a wrong result.

**M1 status note (landed).** The capture API shipped as designed:
`dt_vulkan_capture_begin/active/pending/flush/end/abort` plus
`dt_vulkan_capture_mark/rollback` for the module-fault path; the sync
tap lives inside `dt_vulkan_read_from_device` itself, so callers—and
the seven readback modules—needed zero changes. Unit-tested on
lavapipe (`src/tests/unittests/common/test_vulkan_capture.c`):
bit-identical to eager, exactly one submission per segment, snapshot
survival after host-array clobber, deferred-free non-aliasing,
rollback, sync-tap continuation, abort.

**Failure/fallback during capture.** If a captured module's HAL call
reports an unloadable kernel (or `process_vk` returns −1 while being
captured — modules may still do runtime gating), the recorder rolls
back the nodes appended by that module (node array truncation — cheap
by design), the pipe re-classifies the piece as `CPU`, splits the span
at that point, and continues. The §8a.3 `vk_fallback_reason` plumbing
carries over unchanged.

#### 5.3.1 Sync-point audit (M2a — the contract for span capture)

Result of auditing every host consumer of the trunk against the M1
module-scoped hook, so that the M2 span capture (capture stays open
across module boundaries; interior readbacks skipped) is
correct-by-construction rather than whack-a-mole. Line references are
to the audited revision (`d98a0b0`); the mechanism names are stable.

**Two load-bearing facts about the pipe cache**
(`src/develop/pixelpipe_cache.c`) shape the whole contract:

- **Only the current module's input and output cachelines are
  protected.** Victim selection (`_get_oldest_cacheline`, :203)
  excludes just `lastline` and lines aged ≤ 1; every older line —
  including the span-entry input two modules back — is reclaimable by
  any later `dt_dev_pixelpipe_cache_get`, and reclaim may
  `dt_free_align` the buffer on size change (:330–345).
- **Export and thumbnail pipes have exactly two cachelines** that
  alternate blindly (`_get_cacheline`, :245): a line is *guaranteed*
  reused two `cache_get`s later. Precisely the pipes where spans pay
  most.

Hence two design decisions for M2:

1. **Span-entry trunk uploads are snapshotted, not borrowed.** A
   borrowed host pointer is only safe while its cacheline is
   protected, i.e. within the module that borrowed it — the
   module-scoped M1 hook keeps the borrow; span mode pays one trunk
   memcpy into the capture ring per span *entry* (interior modules
   ride the §5.14 hand-off buffer and upload nothing). Cacheline
   pinning could remove that copy for the many-line darkroom pipes,
   but can never work for the 2-line pipes; revisit with the M3
   planner if profiles say the memcpy matters.
2. **A deferred readback never writes host cachelines at flush.**
   Interior modules' host outputs are simply never produced: their
   cachelines are invalidated (`DT_INVALID_HASH`) at capture time, the
   trunk stays in the hand-off device buffer, and any host consumer
   that fires mid-span performs a **sync-at-need**: flush the capture,
   then read the live trunk into the *current* module's input or
   output line — the only two lines that are provably still owned
   (fact 1). A later-run cache hit on a span interior is impossible
   (invalid hash), which keeps the cache-hit path (:2044) sound. The
   cost — span interiors stop populating the pipe cache in graph
   mode — is the documented interactivity trade until M4's async
   cache taps re-fill important lines from the graph.

**The consumer table.** Every host touch point between `process_vk`
and the next module's dispatch, with its span action
(`pixelpipe_hb.c` lines at `d98a0b0`):

| # | site | fires when | span action |
|---|------|-----------|-------------|
| 1 | input cst transform, in-place host (:1447) | module input cst ≠ pipe cst | **sync-at-need** now; becomes a device glue node (§5.4) later in M2 |
| 2 | `_collect_histogram_on_CPU` on input (:1468) | `request_histogram` | sync-at-need (GPU reduction tap lands M5) |
| 3 | CPU tiling arm (:1497) | module won't fit device | **boundary** (host process; hand-off choke point :1747 already invalidates) |
| 4 | blend-cache replay / store (:1508, :1543, :1713) | `want_bcache` | **boundary** — the M0 hook already refuses device blend for these |
| 5 | trunk upload (:1615) | span entry only | snapshot (decision 1); interior modules use the hand-off, no upload |
| 6 | M0 device blend (:1637) | uniform-mask subset | stays in capture; its refusal gates (picker, mask display, bcache, cs mismatch) all imply a boundary below |
| 7 | output readback (:1649) | every module today | interior: **skipped**, output line invalidated (decision 2); boundary/last module: sync tap exactly as M1 |
| 8 | pfm dump / NaN scan (:1751, :3183) | debug flags | sync-at-need (debug runs trade speed for visibility) |
| 9 | module color picker (:1767) | picker request | **boundary** (picker also forces CPU blend → :11) |
| 10 | blend cst transforms, in-place host (:1792) | CPU blend pending | **boundary** |
| 11 | CPU blend (:1834) | blending outside M0 subset | **boundary** |
| 12 | mask-display bypass copy (:2307) | GUI masking | sync-at-need (rare, GUI-only) |
| 13 | output `cache_get` (:2266) | every module | no pixel read; hazards removed by decisions 1+2 |
| 14 | cache-hit early return (:2044) | later runs | safe: span interiors carry invalid hashes |
| 15 | CL routing / host→image copy (:2513, :2536) | module prefers OpenCL | **boundary** (:2519 already invalidates the hand-off) |
| 16 | CPU module `process` (:1711) | non-VK module | **boundary** (choke point :1747) |
| 17 | preview-gamma picker + scope proxy (:3246) | preview pipe at gamma | no action: gamma is a CPU module, so a boundary already flushed at :16 |
| 18 | importance hints / `invalidate_later` (:3165, cache) | GUI | metadata only, no pixel access |
| 19 | final backcopy → backbuf/export (:3333, :3543) | pipe end | **final sync** at recursion root after the last module |
| 20 | shutdown early-returns (:1465, :1784, :2257 …) and pipe teardown (:422) | cancellation | capture abort at the recursion root's unwind — the span state lives on the pipe, is created and destroyed inside one `process_rec` run, and never survives it |

Items 3, 4, 9–11, 15, 16 end the span *between* modules (the capture
flushes at the previous module's readback, exactly like M1, then the
host path runs); items 1, 2, 8, 12 flush mid-module and the span
continues after the host read. Raster-mask producers stay host-only in
M0/M2 (uniform fill or CPU blend at a boundary), so raster consumers
always run behind a flushed span; `pipe->scharr` is likewise produced
on host paths only.

**M2 status note (span capture landed).** The contract above is
implemented in `pixelpipe_hb.{c,h}` (commit `4ac3479`): one capture
context per run (opened at the first VK module, closed at the
recursion root), `pipe->vk_graph_run` decided once at the root (debug
modes that read per-module pixels — NaN scan, pfm dumps, benchmarks —
simply run eager), readbacks skipped via
`dt_dev_pixelpipe_invalidate_cacheline` + a single
`vk_span_host_stale` marker, and `_vk_span_sync_host` wired at sites
1, 2, 3, 9, 10, 11, 15, 16 plus the root's final sync (19) and
capture close (20). Two refinements the implementation added: only
the *newest* skipped line can ever be consumed again, so one stale
pointer suffices (cleared when its module's successor completes, so
cacheline reuse can't false-match it); and a module fault that had
already consumed the hand-off as its input restores the host input
from the still-intact buffer before the CPU fallback runs. Site 8
became the debug-mode gate rather than a sync. Verified bit-identical
eager-vs-graph on lavapipe (PFM bytes, PNG pixel streams). Spans
currently flush at every host colorspace transform (site 1) — the
glue nodes below are what let them chain.

### 5.4 Glue nodes

Everything the eager pipe does between `process` calls becomes an
explicit node (or is planned away):

- **Colorspace transforms.**
  `dt_ioppr_transform_image_colorspace_cl` already exists as kernels
  (`colorspaces_transform_*`); the VK twins are part of the §5.12
  plumbing. In the graph they become DISPATCH nodes writing a *new*
  temporary — never mutating a shared buffer in place. This
  structurally retires the §5.14 host-mutation invariant and the
  cacheline-invalidation dance in `_pixelpipe_process_on_CPU`
  (`dt_dev_pixelpipe_invalidate_cacheline` after in-place cst
  changes): a graph resource has exactly one content state for its
  whole lifetime, and its `cst` is a static property assigned at
  capture.
  **M2 status note (module-input transform landed).** The Lab↔RGB
  matrix kernels are ported (`colorspaces_transform_rgb_matrix_to_lab`
  / `..._lab_to_rgb_matrix`, dedicated single-entry `.cl`/`.comp`
  siblings for the glslang path) with a general host dispatcher
  `dt_ioppr_transform_image_colorspace_vk` matching the CL twin's
  coverage (matrix profiles only; RAW/LCH/non-matrix fall back). The
  pipe uses it at §5.3.1 site 1 (`_vk_span_cst_glue`): rather than a
  fresh temporary, the live hand-off is transformed into a new buffer
  and swapped in — the resource is still write-once, but the buffer
  churn waits for the M3 planner. Fires only when the input is a
  deferred hand-off *and* the next module consumes it on-device; the
  host input stays deferred (a later sync reads the transformed
  hand-off). Because this moves the transform CPU→GPU, graph output
  stops being bit-identical to eager and becomes float-precision
  equivalent — validated at the source by `test_vulkan_colorspace`
  (VK vs darktable's own CPU transform, <1e-4 linear / <3e-3 sRGB
  gamma). Still host-side: the blend-space transform (site 10, always
  a CPU-blend boundary in M0) and the RGB↔RGB output transform
  (`_rgb_vk` exists but isn't wired into a span yet).
- **Blending.** Requires the blendop kernel port (companion Path A —
  it is on the critical path of *both* designs, so it's pure overlap,
  not extra scope). Per blended module the pipe appends: optional
  blend-space transform nodes for input and kernel output (again into
  temporaries), mask-generation nodes (drawn-mask plane uploaded once
  as an UPLOAD node from the staging ring; blendif parametric mask
  computed by kernel from input/output; feathering via the existing
  guided-filter helper; mask blur via the gaussian helper), the mask
  combine, then the blend-apply node mixing input and kernel output
  into the module's final output resource. Start with the
  normal-mode/no-blendif/no-raster subset exactly as Path A proposes,
  fall back to a sync-tap + CPU blend for the long tail until ported.
  **M0 status note (landed).** The kernel port shipped *wider on modes,
  narrower on masks* than the sketch: `blendop_{lab,rgb_hsl,rgb_jzczhz}`
  (`data/kernels/vulkan/`) carry the full mode switch of all three
  non-RAW blend colorspaces — the switch is one kernel either way —
  while the host path (`dt_develop_blend_process_vk`,
  `src/develop/blend.c`) accepts only uniform-opacity masks
  (`mask_mode == DEVELOP_MASK_ENABLED`) until the blendif/feathering/
  blur node set arrives with the M2 glue work. Two structural deltas:
  the blend applies **in place on the module's output buffer** (the CL
  path's `dev_tmp` copy exists only because image objects can't be
  read and written by the same kernel; buffers can), and when the
  module is a raster-mask source the uniform mask is published
  host-side by `dt_iop_image_fill` — same values as the device mask,
  no readback. The eager VK hook already uses this path today;
  the graph-node form reuses the same kernels unchanged.
- **Detail (scharr) mask.** `dt_dev_write_scharr_mask_cl` already
  computes on GPU and reads back; as a graph node the mask becomes a
  resident single-channel resource written once after
  demosaic/rawprepare, consumed by `develop_details`-type blend nodes,
  with per-consumer distortion nodes (see next bullet). Host readback
  only if a CPU island consumes it (planner knows the consumers).
- **Raster masks.** Producer blend nodes write the mask resource
  (VRAM). Consumers reached without crossing a CPU island get the
  distortion chain as nodes — which requires `distort_mask` VK ports
  for the geometry modules on the path; until those exist, the planner
  inserts a READBACK, runs today's CPU walk (including the
  `dt_dev_distorted_mask_cache_t` caching), and re-uploads at the
  consumer. This degrades exactly one edge, not the trunk.
- **Taps: histogram / scopes / pickers.** Per-module `request_histogram`
  (preview pipe) and picker requests become DISPATCH reduction nodes
  (subgroup + atomics; the CAS-loop `vk_atomic_add_f` and the
  workgroup patterns from the bilateral splat are the precedent)
  writing tiny HOSTVIS buffers, all read back together after the final
  fence — one readback of a few KB replaces N trunk-sized copies. The
  existing CPU reducers stay as the reference implementation for
  validation.
  **M5 status note (histogram reduction landed).** The first tap
  kernel is `histogram_rgb` (`data/kernels/vulkan/histogram.cl`),
  driven by `dt_histogram_helper_vk` (`src/common/histogram.c`): plain
  `uint` atomic increments (native, no CAS needed — that idiom is only
  for *float* accumulation) binning a float4 image into a
  `bins_count·4` histogram, matching the CPU `dt_histogram_helper` for
  the RGB / Lab / Lab→LCh binnings (RAW and profile-compensated RGB
  refuse to the CPU path). Validated against the CPU reducer to *exact*
  bin counts on rounding-invariant inputs (`test_vulkan_histogram`).
  This is the reduction *primitive*; the readback is not yet deferred
  to the end-of-run fence — that needs the **tap registry** below and
  is the next M5 increment. Until it lands, calling the tap mid-span
  would still flush (its readback is a §5.3 sync tap like any other),
  so it is not yet wired into the pipe.
  - **Tap registry (landed).** `dt_vulkan_tap_register` (vulkan.c) is a
    per-run list of `(device buffer, host destination, size)` pending
    taps: the caller appends the reduction DISPATCH and registers the
    tiny result buffer instead of reading it back; `dt_vulkan_capture_end`
    drains the registry in one batch after the final fence (reads
    eagerly, since capture is already inactive) and frees the buffers.
    The buffers live on their own list — not the deferred-free list —
    so they survive every mid-span sync-tap flush and only retire at
    run end; `capture_abort` and a rollback past the registration
    point free them without reading (producing dispatch gone), the
    capture mark carrying a tap count for the latter. Unit-tested in
    `test_vulkan_capture` (deferred-until-end, survives-a-flush,
    abort/rollback-drop). **Remaining (M5c):** wire the histogram tap
    in at §5.3.1 site 2 — dispatch on the hand-off, register the result
    to `piece->histogram`, and compute `histogram_max`/stats at drain.
    That is preview-pipe behaviour (histograms don't fire in headless
    export), so it wants a darkroom/GPU bench to validate; the two
    mechanisms it composes are each already tested.
- **Pixelpipe-cache taps.** See §5.7.
- **Format conversions** (`bpp` changes along the RAW segment, 1f→4f
  at demosaic) are DISPATCH nodes like any other; the RAW trunk
  segment is just resources with different element sizes.

### 5.5 Memory planning

After capture, per graph:

1. Node order is already topological (capture order = pipe order;
   side edges only ever point forward — dt refuses backward raster
   references, see `dt_dev_get_raster_mask`'s iop-order check).
2. Liveness: one linear sweep fills `first_node`/`last_node` per
   resource from the nodes' read/write masks. Blended modules extend
   their input's interval to the blend-apply node automatically
   (the capture recorded that read — no special casing).
3. Greedy interval-packing into one arena: sort resources by size
   descending, best-fit into a free-list of `[offset, size)` holes à
   la vkdt's `alloc.c`, honouring `minStorageBufferOffsetAlignment`
   (and `nonCoherentAtomSize` for HOSTVIS). Implementation choice:
   one big `VkBuffer` + descriptor offsets (simplest with our
   buffer-only HAL: one allocation, one buffer, descriptors are
   `{buffer, offset, range}`) — falling back to N chunk buffers when
   `maxStorageBufferRange` (≥128 MB min spec, 4 GB on desktop
   drivers) or budget fragmentation demands.
4. Budget: arena size vs `VK_EXT_memory_budget`. If over budget →
   spill planning (§5.10).
5. PINNED resources (residency cache, §5.7) are allocated outside the
   aliasing arena so their contents survive across runs.

Expected peak for a typical stack (24 MP export, float4 trunk ≈
384 MB): 2 trunk ping-pong buffers + 1 blend-diamond extension + the
largest single module's scratch (heavy modules like `diffuse`/`atrous`
run 2–6 image-sized scratch buffers, but those alias *inside* the
module's interval) + mask planes (¼ size each) ≈ **4–7 × trunk ≈
1.5–2.7 GB**, comfortably inside 8 GB cards and workable on 4 GB for
screen-sized ROIs (darkroom trunk at 6 MP viewport ≈ 96 MB → arena
< 700 MB). Today's eager model has a *similar or worse* transient
footprint (pool buffers + staging + CL duplicates) — it's just never
been visible in one number. The win isn't only peak; it's that
`vkAllocateMemory`/`vkCreateBuffer` churn drops to ~zero per run
(plan reuse, §5.8).

### 5.6 Record & submit

- Per segment: allocate descriptor sets from a per-graph pool (reset
  once per run), record nodes in order; between a writer and its
  readers a `vkCmdPipelineBarrier` (COMPUTE→COMPUTE,
  `SHADER_WRITE→SHADER_READ`, per-buffer ranges; batch barriers for
  adjacent nodes). UPLOAD nodes at segment head as one
  TRANSFER batch + single TRANSFER→COMPUTE barrier — the
  `dispatch_n_batched` pattern promoted to whole-segment scope.
- One `vkQueueSubmit` per segment; one fence (or one timeline
  semaphore counting segments) awaited at pipe end or at sync-taps.
- **Cancellation** (`pipe->shutdown`, the `DT_DEV_PIXELPIPE_STOP_*`
  protocol): a submitted command buffer can't be aborted, so segments
  double as cancellation checkpoints. The recorder caps segments at a
  budget (e.g. ~64 dispatches or an ms-estimate once per-kernel
  timings accumulate), checks `_pipe_has_shutdown()` between submits,
  and skips the remainder on stop. Today's granularity is per-module,
  so segment-granular (a few modules) is a mild regression on
  abort latency but a huge win on throughput; the budget knob tunes
  the trade. Timings come from optional per-node timestamp queries
  (`VK_QUERY_TYPE_TIMESTAMP`), which also feed `-d perf` logs and the
  future scheduler.
- **Queues.** Phase 1: one compute queue per device, per-pipe-context
  submission serialised per queue (this replaces the global
  `g_vk_lock` with per-context recording locks + a queue lock —
  subsumes companion Path E). Phase 2 options: distinct queue for the
  preview pipe (preview and full overlap on hardware that exposes ≥2
  compute queues), async-transfer queue for cache-tap readbacks and
  CPU-island staging so they overlap with compute.

### 5.7 Pixelpipe cache integration (the interactivity question)

The host pixelpipe cache exists to make darkroom edits cheap: change
module N → upstream comes from cache, only N..end recompute. A
GPU-resident pipe must preserve that, and can improve on it. Three
layers:

1. **Host cache stays, writes become async.** The cachelines and the
   `dt_dev_pixelpipe_cache_hash` machinery are untouched (they also
   serve CPU/CL runs and snapshot/duplicate-preview flows). What
   changes: the "important" input write-back (inventory row 7) becomes
   a READBACK node into a HOSTVIS staging slice recorded *in the
   graph* (dedicated to the focused module + `IOP_FLAGS_WRITE_PIPECACHE`
   modules, same policy as today, decided at capture from the same
   flags), memcpy'd into the cacheline after the final fence. No
   mid-run stall, no extra submissions.
2. **VRAM residency cache (the real upgrade).** The graph context pins
   a small set of trunk resources across runs — at minimum the
   *input of the focused module* (`module == dt_dev_gui_module()`,
   same predicate the eager path uses for `important_cl_input`),
   tagged with the same cumulative hash the host cache uses
   (`dt_dev_pixelpipe_cache_hash(roi, pipe, pos)`). On the next run,
   if the hash at position N matches a pinned resource, capture
   starts the GPU span *at N* with that resource as trunk head: the
   upstream isn't just cache-served, it's cache-served **in VRAM**
   with zero upload. Budgeted (e.g. 2–4 pinned lines, LRU), dropped
   under memory pressure — correctness never depends on it because
   layer 1 still exists.
3. **Cache hits upstream of the span** (host cachelines from previous
   runs, e.g. after restart or when residency was evicted) work as
   today: the recursion terminates at the hit, and the graph's source
   UPLOAD node takes the cacheline instead of `pipe->input`.

Slider-drag steady state with layers 1+2: history commit → capture
from module N (µs–ms, host only) → submit nodes N..end → fence →
small backbuf readback. The trunk never crosses PCIe at all.

### 5.8 Incremental re-run (the runflags analogue)

vkdt classifies changes; we get the same effect from infrastructure dt
already has:

| Change | Detected via | Work re-done |
|---|---|---|
| params of module N (slider) | piece hash change at N (existing) | re-capture N..end (host code must re-run: params → push constants/LUTs), reuse memory plan if topology+ROIs+sizes unchanged (`topology_hash`), record + submit. With residency (§5.7) upstream is free. |
| ROI (zoom/pan/scale) | roi set differs | full re-capture + re-plan. Arena realloc only if high-water grows (arena is monotonic per context, like the staging buffer today). |
| topology (enable/disable/reorder/new instance) | `DT_DEV_PIPE_REMOVE/SYNCH` events (existing) | full rebuild incl. plan; residency pins whose position-hash died are dropped. |
| nothing (cache hit at pipe end) | backbuf hash (existing) | nothing — unchanged fast path. |

Re-capture is deliberately cheap (append-only arrays, no Vulkan calls;
Vulkan objects touched only at record time: descriptor writes and
command recording, both µs-scale per node — ~100–300 nodes for a real
stack). We do **not** attempt vkdt's "params-only → reuse recorded
command buffer, just update UBO" level, because our params ride in
push constants and captured host code legitimately re-derives LUTs
from params. If profiling ever shows record cost mattering (it
shouldn't at 300 nodes), the C-API modules (§4/approach C) can adopt
UBO-resident params later.

### 5.9 CPU islands

Until `toneequal` & friends are ported, real pipes contain CPU
islands. The planner makes them cost exactly one exit + one entry:

- span tail: READBACK node of the trunk into pinned host staging
  (`HOST_VISIBLE|HOST_COHERENT`, persistent ring per context) — plus
  readbacks of any masks the island consumes;
- fence; run the island's `process()` (OpenMP) on the staged buffer
  exactly as today (colorspace glue on CPU as today);
- next span head: UPLOAD node from staging; continue capture.

Compared with today's VK arm the *count* of transfers around a CPU
module is the same, but they're planned (pinned memory, no
per-transfer allocation, overlappable with the transfer queue) and
they *don't* additionally break the surrounding chain — the spans on
both sides remain single-submission graphs. As Path D ports land,
islands disappear and the planner automatically fuses the spans; no
further pipeline work needed.

### 5.10 When the arena doesn't fit (tiling and huge exports)

Per-module tiling (`process_tiling`) cannot run *inside* a fused
graph — it's an eager-model concept (its `factor_cl`/`overlap`
accounting remains authoritative for the eager fallback). Graph-mode
strategies, in escalation order:

1. **Segment spill:** split the trunk at the liveness high-water
   point(s); earlier segment's tail spills to host staging (or stays
   in VRAM if only scratch was the problem — spilling scratch is
   never needed because scratch dies at module end), later segment
   reloads. Costs 2 transfers per split — still far better than
   per-module staging, and the planner picks split points that
   minimise crossings (e.g. after `crop`/`finalscale` where the trunk
   shrinks).
2. **Graph-level tiling** (vkdt-style): run the whole span per tile
   with accumulated overlap = Σ module overlaps along the span.
   Overlap accumulation across many neighbourhood modules can explode
   the halo, so this pays only for spans of mostly point ops —
   *deferred* until data shows segment spill isn't enough.
3. **Fallback:** run that pipe eagerly (today's path). Always
   available, decided per run in the §5.2 gate (predicted arena vs
   budget).

Rule of thumb from §5.5's estimate: 24 MP export fits 8 GB cards
without spilling; 61 MP on a 4 GB card takes 1–2 spills; the darkroom
screen pipes essentially never spill.

### 5.11 Error handling & the fallback ladder

Mirrors the existing OpenCL discipline (`pipe->opencl_error` → restart
without CL):

- capture-time module failure → span split + CPU piece (§5.3), run
  continues;
- submit/fence failure or `VK_ERROR_DEVICE_LOST` → mark
  `pipe->vulkan_error`, restart the run on the eager path (CL/CPU) —
  identical shape to today's `goto restart` after `opencl_error`;
- budget misprediction (alloc fail at plan time) → spill planning →
  eager fallback;
- every fallback logs a graph dump reference (§5.12).

### 5.12 Observability

Non-negotiable for something this central:

- `-d vkgraph`: per-run one-line-per-node dump (tag, kernel, sizes,
  resource ids with offsets, barriers, segment boundaries, spill
  points, SYNC_TAP causes) — the analogue of `-d pipe` today, and the
  thing that makes "why did my pipe split?" answerable.
  *M1 status: the flag exists (`DT_DEBUG_VKGRAPH`) and logs per-segment
  summaries (node/dispatch/upload/copy counts, staged bytes, submit
  result) plus sync-tap and rollback events; the per-node dump comes
  with the M3 planner where offsets exist to print;*
- `VK_EXT_debug_utils` labels per node (module op + phase) so
  RenderDoc/Nsight captures read like the pipe;
- per-node timestamps behind `-d perf`;
- the lavapipe CI harness gains a graph-mode twin for every existing
  module equality test (same inputs through eager VK vs graph VK must
  be bit-identical — they dispatch the same kernels with the same
  constants, so any diff is a scheduler bug);
- graph invariant checks in debug builds: every read has a prior
  write, no resource used outside its interval, barrier coverage.

---

## 6. Lifecycle walkthroughs

**Darkroom slider drag (module N, all-VK stack, warm caches).**
history commit → piece hash N changes → run: capture skips 0..N−1
(residency pin at N's input matches), captures N..gamma incl. blend +
tap nodes (~ms host) → plan reuse (topology unchanged) → record +
submit 1 segment → fence → picker/histogram KB-readbacks + backbuf
readback. PCIe traffic: backbuf (≈ viewport 8-bit) + taps. Today the
same drag moves the trunk across PCIe 2–6 times.

**Zoom/pan.** ROI set changes → full re-capture & re-plan (µs–ms) →
arena already sized (monotonic) → submit. Residency pins keyed on
position-hash+roi go stale and simply miss (drop to host-cache layer,
which also keys on roi — unchanged semantics).

**Export.** Fresh context, no residency, minimal pixelpipe cache
(`DT_PIPECACHE_MIN`) as today. One capture over the full stack;
islands per §5.9; segments sized for budget, not cancellation
(exports check shutdown between segments as a bonus). Expected
motion: source upload + final readback + islands.

**Preview + full concurrently.** Two contexts, two graphs, own arenas
and staging rings; submissions interleave on the queue (or ride two
queues). No shared mutable state except the HAL kernel table
(read-only after init) — the residency/bcache-style races the eager
path had to hand-tune disappear with per-context ownership.

---

## 7. Coexistence with OpenCL during the transition

- The graph executor is **Vulkan-only**. No CL nodes: interleaving two
  device runtimes inside one scheduled graph multiplies sync/interop
  complexity for a shrinking payoff (DMA-BUF interop — companion Path
  C — remains a possible *island bridge* optimisation if profiling
  ever justifies it; the planner treats a CL-only module simply as a
  CPU-visible island endpoint).
- Per pipe run, the §5.2 gate picks graph-VK vs eager (CL/CPU) — a
  pipe whose stack is largely un-ported keeps its current CL
  performance untouched. The existing per-module `process_cl` arm is
  not modified by this design at all.
- End state (companion milestone 13, OpenCL retirement) is what makes
  the maintenance story actually *simpler than today*: one backend,
  one scheduler, and `pixelpipe_hb.c` sheds the CL arm, the VK-hook
  arm, the hand-off cache, and their interaction rules.

## 8. Performance model

Against the §10.2 export breakdown (15 s, RX 9060 XT, 18 MP):

| Bucket (today) | Today | Graph mode |
|---|---:|---|
| CPU-only modules (toneequal ×2) | ~5.6 s | unchanged until ported (island), then ~0.1 s |
| CPU blending after VK | ~3–5 s | GPU blend nodes: ~0.05–0.1 s |
| CL↔VK transitions | ~1–2 s | gone (single backend per run) |
| Per-module staging/sync residue | ~1–3 s | source upload + sink readback: ~0.15 s |
| Kernel work | <0.5 s | <0.5 s (unchanged math) |
| **Total** | **~15 s** | **~6 s with islands; ~1 s once toneequal ports** |

Darkroom: a mid-stack slider on a 6 MP viewport today costs (upload +
process + readback + CPU blend) ≈ 150–500 ms wall on the traced
hardware; graph mode with residency ≈ kernel time + one fence ≈
**10–40 ms** — the difference between "adjust, wait, look" and live
feedback. The preview pipe benefits identically, which compounds:
scopes and thumbnails stop competing with the full pipe for PCIe.

These are estimates, not measurements; M1 (below) exists to replace
them with numbers before the big pieces land.

### 8.1 First measurement: what actually bounds span length (post-M2)

The M2 deep-pipe run (§11, ~60-module benchmark XMP on lavapipe) gives
the first empirical picture, and it matches the model's premise. The
run collapsed into 11 submissions, and **every interior break was a
`sync-at-need (cpu process)`** — a CPU-island module (no `process_vk`,
or a runtime fallback) reading the trunk. Not one break came from the
DAG machinery itself: no capture faults, and the one non-`cpu process`
sync was a profile-without-matrix colorspace transform the glue node
correctly declined. Spans grew until they hit a CPU module and no
further — the largest reached 565 nodes / 525 dispatches in a single
submit.

Two consequences for milestone priority:

- **The span core is at its ceiling for a given VK coverage.** More
  fusion now comes from *shrinking the CPU islands*, i.e. Path B
  module ports (the independent lane) — not from more DAG-core work.
  Each ported mid-stack module merges the two spans it currently
  separates.
- **Among the remaining DAG milestones, M5 (GPU taps) outranks M3
  (memory aliasing) for interactivity.** The breaks that will matter
  most in darkroom/preview are the picker/histogram/scope readbacks;
  those are M5. M3's win is peak VRAM, which (a) this software-Vulkan
  test bench can't measure and (b) doesn't change submission count.
  M3 stays queued but is no longer the obvious next step it looked
  like on paper.

## 9. Migration plan

Each milestone lands green and user-invisible-by-default
(`pixelpipe_vulkan_graph` pref, default off until M5).

- **M0 — blendop VK kernels** (= companion Path A, already planned).
  Needed by eager *and* graph paths. ✅ **landed** (see §11): all
  blend modes of the Lab / RGB display / RGB scene colorspaces plus
  the mask fill, wired into the eager VK hook for uniform-opacity
  masks via `dt_develop_blend_process_vk`; drawn/parametric/raster
  masks and RAW keep the CPU blend until the M2+ node set.
- **M1 — capture HAL + single-segment executor.** ✅ **landed** (see
  the §11 implementation log). Thread-local capture context; IR; naive
  planner (no aliasing: real pool buffers + deferred frees); one
  command buffer / one submit / one fence per segment; sync-tap flush;
  mark/rollback for module faults; `-d vkgraph`; unit tests on
  lavapipe. Success criterion met: a captured multi-kernel chain runs
  as one submission, bit-identical to eager (asserted in
  `test_vulkan_capture`). Pixelpipe integration shipped
  **module-scoped** behind `pixelpipe_vulkan_graph` (default off):
  each module's uploads + dispatches + copies reach the queue as one
  submission with every inter-module contract unchanged;
  darktable-cli exports verified bit-identical pref on vs off.
- **M2 — span capture + glue nodes.** Widen the capture window from
  module to GPU span. **Landed:** the sync-point audit (§5.3.1), the
  span machinery (run-scoped capture, deferred interior readbacks,
  sync-at-need sites, root final sync — bit-identical eager-vs-graph),
  and the **module-input colorspace glue node** (§5.4 status note:
  Lab↔RGB matrix transforms on-device, so a `colorin → … → colorout`
  span now captures as one submission; float-precision equivalent to
  eager, kernels validated against the CPU transform). Remaining:
  blend-space transforms (they gate on CPU blend today, so they wait
  for blendif/mask nodes) and format conversions; then the RGB↔RGB
  output transform wired into the export tail.
- **M3 — memory planner.** Liveness + arena + aliasing + budget +
  spill-by-segmentation. Retire per-dispatch pool churn in graph mode.
- **M4 — cache integration.** Async cache-tap readbacks; VRAM
  residency pins; plan reuse via topology hash.
- **M5 — taps on GPU.** Histogram/picker/scope reduction kernels +
  end-of-run KB readbacks; scharr as resident resource. Flip pref
  default on for darkroom pipes; export next release. *In progress:*
  two tested mechanisms landed (§5.4 status note, §11) — the
  **histogram reduction kernel** (validated against the CPU reducer)
  and the **tap registry** (`dt_vulkan_tap_register`, deferring a
  tap's readback to the end-of-run fence so it stops splitting the
  span). Remaining: **M5c** — compose them in the pipe (dispatch the
  histogram on the hand-off at site 2, register it to
  `piece->histogram`, compute max/stats at drain), plus the
  picker/scope reduction kernels. M5c is preview-pipe behaviour, so it
  wants a darkroom/GPU bench to validate end to end.
- **M6 — islands polish.** Pinned staging ring, transfer-queue
  overlap, planned island bridging; `distort_mask` VK ports for
  raster-mask edges as they come.
- **M7 — scheduler extras (opportunistic).** Per-node timestamps
  feeding segment budgets; preview/full on separate queues; C-API
  (`create_nodes_vk`) for `diffuse`-class modules; graph-level tiling
  if spill data says it's worth it.

Independent lanes that keep paying either way: Path B module ports
(each un-islands a stack), Path D CPU-module ports (toneequal), §8.5
images/samplers (unlocks the last OpenCL-only kernels).

## 10. Risks and open questions

- **Capture assumes modules don't inspect device buffer contents
  synchronously.** Audited: the known offenders use the priming
  pattern and degrade to sync-taps; but third-party/lua-adjacent
  assumptions should be re-checked when the pref defaults on.
- **Segment sizing vs cancellation latency** needs a real-world knob;
  darkroom wants ~50 ms segments, exports want maximal fusion.
- **Arena fragmentation across runs** (monotonic arenas + varying ROI
  sets): mitigate with high-water reuse and periodic trim on idle,
  like the staging buffer today.
- **MoltenVK**: single big VkBuffer + offsets and timeline semaphores
  are portability-subset-safe, but `maxStorageBufferRange` and
  barrier granularity on Metal deserve an early smoke test (the M1
  executor is the right vehicle).
- **Descriptor volume**: ~300 nodes × ≤20 bindings per run is well
  within pool limits, but per-run reset strategy vs
  `VK_KHR_push_descriptor` is worth a micro-benchmark.
- **Preview-pipe priming inversion**: graph mode makes the *full* pipe
  so much faster that the preview pipe may become the laggard; may be
  worth running preview at graph priority or deriving primes from the
  full run instead.
- **Where does `pipe->dsc` mutation end up** if a captured module's
  host code depends on a *pixel-dependent* dsc field written by an
  upstream module in the same span? Survey says current fields
  (`processed_maximum`, temperature coeffs) are param-derived, not
  pixel-derived, so capture order suffices — but this invariant should
  be asserted in debug builds.
- **Naming**: "graph mode" vs "fused pipe" vs "resident pipe" — pick
  once before the pref ships.

## 11. Implementation log

Living section, newest first. Every landing that touches the design
gets an entry here; where the implementation deviates from the
proposal, the affected section carries an inline **status note** and
the rationale lives here. Keep this in lockstep with the code.

### 2026-07-13 — M5 tap registry landed: deferred reduction readbacks

Commit: `4505357` (registry + capture-context integration + tests).

`dt_vulkan_tap_register(devid, buf, host_dst, size)` records a pending
tap on the capture context; `dt_vulkan_capture_end` drains them all
after the final fence. The design points that made it correct:

- **Own list, not the deferred-free list.** Tap buffers must outlive
  every *mid-span* sync-tap flush (their result is read only at run
  end), whereas the deferred-free list is drained on each flush. So
  taps sit in a separate array and retire only at `capture_end`.
- **Drain reads eagerly.** `capture_end` sets `active = FALSE` before
  draining, so each tap readback takes the plain (non-capturing) read
  path instead of trying to re-flush an already-ended capture.
- **Rollback/abort drop, don't read.** A module fault removes the
  producing dispatch, so the tap result is meaningless — the mark
  gained a `taps` field and rollback frees the buffers registered
  after it without reading; abort does the same for the whole list.
- **Ownership.** The registry owns the buffer and frees it at drain,
  so the caller registers and walks away (must not also free it).

Tested in `test_vulkan_capture` (now 12): the deferred value is absent
mid-span and present only after `capture_end`; a tap survives a
mid-span sync-tap flush of a different buffer; abort and rollback drop
taps without reading. This is the mechanism half of the histogram
tap; composing it with the kernel in the pipe (M5c) is preview-pipe
work for a darkroom bench.

### 2026-07-13 — M5 first tap landed: GPU histogram reduction kernel

Commit: `e191085` (kernel + host dispatcher + test).

What shipped:

- **Kernel.** `histogram_rgb` (`data/kernels/vulkan/histogram.{cl,comp}`)
  bins a float4 image into a `bins_count·4` uint histogram with
  **native `uint` atomic increments** — the `vk_atomic_add_f` CAS idiom
  is only needed for *float* accumulation, integer counters use
  `atomic_add`/`atomicAdd` directly. One kernel, a mode switch over the
  three 4-channel-float binnings (RGB, Lab, Lab→LCh) with the same
  scale/shift/clamp-and-truncate as the CPU reducers.
- **Host dispatcher.** `dt_histogram_helper_vk` (`src/common/histogram.c`)
  zeroes the small device histogram (a few-KB upload), dispatches, and
  reads the result back. Subset gates mirror `dt_histogram_helper`'s
  switch: RAW and middle-grey-compensated RGB return FALSE so the
  caller keeps the CPU reducer.
- **Validation shift, again.** Bin counts *can* differ between CPU and
  GPU when a pixel's scaled value lands next to a bin boundary and the
  two float paths round it across. `test_vulkan_histogram` handles that
  with two regimes: rounding-invariant *bin-centre* inputs must match
  the CPU histogram **exactly** (proves the bin indices and atomic
  counting), and random inputs are bounded by a <1% total-variation
  budget (proves no gross error), across RGB/Lab/LCh. Both paths must
  also bin every sampled pixel exactly once per channel.
- **Deliberately not wired yet.** This is the reduction *primitive*.
  Wired into the pipe mid-span today its readback would still flush
  (it's a §5.3 sync tap), so there is no span benefit until the tap
  registry defers the readback to the end-of-run fence (§5.4). That
  registry + the pipe wiring is the next M5 increment and wants a
  darkroom/GPU bench: taps fire in the preview pipe, which this
  headless export harness doesn't exercise.

### 2026-07-13 — M2 glue node landed: module-input colorspace transform on-device

Commit: `6622949` (Lab↔RGB kernels + host dispatcher + pipe wiring +
tests).

What shipped:

- **Kernels.** `colorspaces_transform_rgb_matrix_to_lab` and
  `..._lab_to_rgb_matrix` ported to `colorspaces.cl` (multi-entry for
  clspv) plus dedicated single-entry `.cl`/`.comp` siblings so the
  glslang fallback exposes each — the same pattern gaussian/blendop
  use. Straight ports of the CL twins: input TRC LUT, matrix from the
  profile struct, XYZ↔Lab.
- **Host dispatcher.** `dt_ioppr_transform_image_colorspace_vk`
  mirrors the CL function's structure and pass-through policy — matrix
  profiles only, RAW/unsupported pairs return FALSE with
  `*converted_cst == cst_from` so the caller can sync + CPU-convert.
  Reuses the existing `build/free_iccprofile_params_vk` plumbing
  (capture-safe: profile uploads snapshot into the ring).
- **Pipe wiring.** `_vk_span_cst_glue` at §5.3.1 site 1 transforms the
  live hand-off into a new buffer and swaps it in, advancing the cst
  and leaving the host line deferred — the span survives the hop.
  Gated to fire only when the input is a deferred hand-off and the
  module will consume it on-device (else a CPU module would sync the
  transformed pixels straight back). The fault path already restores
  the host input from the transformed hand-off.
- **Testing shift.** The glue node moves math CPU→GPU, so graph output
  is no longer bit-identical to eager — the M1/M2b bit-identity signal
  doesn't apply to it. Replaced with `test_vulkan_colorspace`: feed
  one constructed sRGB profile through darktable's *own* CPU
  `dt_ioppr_transform_image_colorspace` and through the VK path,
  require agreement to <1e-4 (linear TRC) / <3e-3 (sRGB gamma) both
  directions, plus a round-trip inversion <1e-4. This caught a real
  setup bug (the CPU path reads `matrix_*_transposed`, which a naive
  fixture leaves invalid → NaN) — a good sign the comparison is
  actually exercising the CPU reference.
- **Evidence.** lavapipe: `-d vkgraph` on the test export shows the
  three VK modules and two cst glue nodes collapse from three
  submissions (the M2b span-capture number) to **one** — 18 nodes,
  1 submit. The eager path stays byte-for-byte the pre-M2 baseline
  (PFM md5 unchanged); graph output differs from eager by max ~2e-5 /
  mean ~2e-7, nothing over 1e-3. Full ctest green (5 suites).
- **Deep-pipe check.** The same PPM re-exported through a ~60-module
  benchmark XMP (`src/tests/benchmark/darktable-bench-4.2.xmp`) at
  702×465 to stress real spans: 46 modules defer their readback, and
  the biggest span fuses **565 nodes (525 dispatches) into one
  submit**. Graph-vs-eager agreement holds at scale — max abs diff
  5.2e-4 on a 0.85 value (0.06% relative), mean 6.4e-6, zero floats
  past 1e-2 — confirming the span/defer/glue machinery composes
  correctly across a realistic module stack, not just the 3-module
  case.

### 2026-07-13 — M2 span capture landed: run-scoped context, deferred readbacks

Commit: `4ac3479` (pixelpipe span state + sync sites + root scope).

What shipped, against the §5.3.1 contract:

- **Run-scoped capture.** One context per pipe run: opened lazily at
  the first VK module's hook, flushed by sync taps and sync-at-need
  reads, closed (or aborted, on error/shutdown) at the recursion
  root. `pipe->vk_graph_run` is decided once at the root so the mode
  can't flip mid-span; the pixel-reading debug modes (NaN scan, pfm
  dumps, module benchmarks) get their per-module host visibility by
  running eager.
- **Deferred readbacks.** A module with no host consumer of its
  output in its own scope (no pending CPU blend, no fast-blend cache,
  no picker, no mask display) skips the device→host readback;
  the cacheline is invalidated (later-run cache hits impossible) and
  marked stale via a single `vk_span_host_stale` pointer — one
  suffices because only the newest skipped line is ever consumable
  again, and it's cleared once its successor module completes so
  cacheline reuse can't false-match. Sync-at-need
  (`_vk_span_sync_host`) is wired at audit sites 1, 2, 3, 9, 10, 11,
  15, 16; the root serves site 19 (final output) and 20 (context
  close).
- **Snapshot uploads under capture** (audit decision 1): the M1
  borrowed-pointer trunk upload is retired in graph mode — the flush
  may now happen modules later than the borrowing module, outside
  the pipe cache's two protected lines. Costs one ring memcpy per
  span entry; interior modules ride the hand-off and upload nothing.
- **Fault handling.** Per-module capture marks + rollback replace the
  M1 per-module end/abort, so a fault drops only the faulting
  module's nodes. If the fault had already consumed the hand-off as
  `vin` while the host input was never materialized, the input is
  restored from the still-intact buffer before the CPU fallback —
  closing the one path where the M1-era semantics would have read
  garbage.
- **Cross-run hygiene**: the hand-off is dropped at run start (it is
  keyed by size alone; parameters may have changed), and the span
  state never survives the run.
- **Evidence.** lavapipe A/B: PFM exports byte-identical pref off vs
  on; PNG exports pixel-stream-identical (file md5 differs only in
  metadata timestamp chunks — verified by decompressing IDAT). `-d
  vkgraph` shows all three VK modules of the test pipe deferring
  ("run done: 3 readbacks deferred"), the two mid-span
  cst-transform syncs, and the final root sync. Full ctest green.
- **Reality check, kept honest in §9:** submissions per run didn't
  drop yet — the default pipe has a host colorspace transform between
  every VK module pair, so each span flushes at site 1 before the
  next module starts. That is exactly the remaining M2 work (glue
  nodes); the defer/sync machinery this landing proves is what makes
  those transforms *worth* porting.

### 2026-07-13 — M2a done: the span-capture sync-point audit

Doc-only landing; no behaviour change. §5.3.1 is the deliverable: the
table of every host consumer of the trunk between `process_vk` and the
next module's dispatch, and the span contract derived from it. The
audit turned up two cacheline-lifetime facts that decided the design
before any code was written:

- `dt_dev_pixelpipe_cache_get` protects only the current module's
  input and output lines (victim selection excludes just `lastline`
  and age ≤ 1, `pixelpipe_cache.c:203`), and reclaim may free the
  buffer outright on size change (:330). A borrowed span-entry upload
  would dangle as soon as the span crosses its second module.
- Export/thumbnail pipes alternate exactly two cachelines (:245) — a
  line is *guaranteed* reused two modules later, in exactly the pipes
  where spans pay most.

Consequences (now §5.3.1's decisions 1+2): span-entry uploads are
snapshotted into the capture ring (the M1 borrow stays for the
module-scoped mode, where the flush happens inside the borrowing
module); interior readbacks are skipped by invalidating the output
cacheline rather than deferring a host write into memory the cache may
hand to someone else; every mid-span host consumer syncs at need into
the current module's still-protected lines. The M2 milestone in §9 now
tracks only the implementation work.

### 2026-07-12 — M0 landed: VK blend kernels, uniform-mask blending in the eager hook

Commit: `d98a0b0` (kernels + host path + pipe wiring + tests).

What shipped, against the plan:

- **Kernels** (`data/kernels/vulkan/blendop_{lab,rgb_hsl,rgb_jzczhz,
  set_mask}.{cl,comp}` + `dt_vulkan_blendop.h`): straight ports of the
  three `blendop_*` kernels in `data/kernels/blendop.cl`, each carrying
  the *complete* mode switch — arithmetic and lighten/darken families,
  HSL/HSV/LCH recombinations, `DEVELOP_BLEND_REVERSE` operand swap,
  `blend_parameter`, `mask_display` alpha transfer. Buffer-based like
  the rest of the VK kernel set: `iwidth` + `offx/offy` express the
  roi_in offset the CL image path got for free from `read_imagef`
  coordinates.
- **Host path** (`dt_develop_blend_process_vk`, `src/develop/blend.c`):
  mirrors the CL blend driver's structure with the M0 gates up front —
  returns FALSE (nothing touched, caller keeps the CPU blend) unless
  the mask is uniform (`mask_mode == DEVELOP_MASK_ENABLED`), the blend
  colorspace equals the module output colorspace, 4-channel, non-RAW.
- **Pipe wiring** (`pixelpipe_hb.c` VK hook): after a successful
  module `process`, the hook tries the device blend before the
  readback; success sets `PIXELPIPE_FLOW_BLENDED_ON_GPU` and skips the
  CPU `_transform_for_blend` + blend-apply. Gated off when a picker or
  mask display needs host data, and when the fast-blend cache wants to
  store the pre-blend input (`want_bcache`) — the eager cache contract
  stays intact.
- **Design deltas, with reasons.** (1) *Wider on modes, narrower on
  masks* than the planned "normal-mode subset": the mode switch is one
  kernel either way, and sweeping every mode against independent C
  references cost less than curating a subset; drawn/parametric masks
  need the blendif/feathering/blur node set that belongs with M2's
  glue work. (2) The blend applies **in place on the module's output
  buffer** — the CL path's `dev_tmp` round-trip exists only because
  image objects can't be read and written by the same kernel; buffers
  can, so the copy and its allocation are gone. (3) A raster-source
  module publishes the uniform mask host-side via `dt_iop_image_fill`
  (bit-identical values, no readback). §5.4 carries the matching
  status note.
- **Testing lessons recorded.** cmocka assertions longjmp out of the
  test; the first cut asserted while holding the device lock, so one
  comparison failure deadlocked every later test on `g_vk_lock`. The
  suite now runs each device sequence to completion, unlocks, then
  asserts. Comparisons use a scale-aware error (absolute below
  |ref| = 1, relative above): HSV round-trips on out-of-gamut inputs
  legitimately produce channels of magnitude ~1e2, where GPU fma
  reassociation exceeds any absolute epsilon calibrated near 1.
- **Evidence.** `test_vulkan_blendop` (7 tests, lavapipe): per-mode
  device-vs-reference sweeps in all three colorspaces (REVERSE,
  blend_parameter, mask_display, roi offsets covered), the mask fill,
  an end-to-end uniform blend through `dt_develop_blend_process_vk`,
  and the subset-gate refusals. Full ctest green; `darktable-cli`
  exports remain bit-identical with `pixelpipe_vulkan_graph` off vs
  on (`-d vkgraph` confirms capture active on the graph run).

Next up: M2 span capture, starting with the sync-point audit the
milestone lists; the M0 blend gates (colorspace equality, uniform
mask) are exactly the seams M2's glue nodes widen.

### 2026-07-12 — M1 landed: capture HAL, segment executor, module-scoped pipe integration

Commits: `b5f0e23` (HAL + executor + tests), `dcf5c66` (hand-off
staleness fix), `6cf9b18` (pixelpipe integration + pref).

What shipped, against the plan:

- **Capture HAL + executor** (`src/common/vulkan.{c,h}`): thread-local
  `dt_vk_capture_ctx_t`; all dispatch shapes, both copies, and both
  upload flavours captured at the entry points the modules already
  use; `read_from_device` doubles as the §5.3 sync tap;
  `mark`/`rollback` implement the module-fault path; the executor
  plans staging offsets in one pass, stages every upload with a
  single map/memcpy, allocates one descriptor pool per segment, and
  replays the nodes into one command buffer with conservative
  adjacent-node memory barriers — one `vkQueueSubmit`, one fence.
  A per-device `submit_count` makes the collapse assertable.
- **Design deltas, with reasons.** (1) The M1 "planner" uses *real*
  pool buffers plus flush-deferred frees instead of virtual resource
  ids — deferral is what actually protects correctness here (an eager
  free can destroy the `VkBuffer` under pending nodes or hand it to a
  concurrent eager thread); virtual ids without interval-aliasing
  would be ceremony. §5.1 carries the matching status note. (2) A
  `dt_vulkan_write_to_device_borrowed` variant was added for
  trunk-sized caller-owned uploads; the snapshot default stands for
  module-owned LUTs (they really do die before the flush — the unit
  test clobbers one to prove it matters). (3) `-d vkgraph` logs
  segment summaries now; the per-node dump waits for M3 offsets.
- **Pixelpipe integration is module-scoped, not span-scoped.** The
  capture opens after the device lock in the VK hook and closes
  before the buffer frees; the module's output readback is the sync
  tap that flushes the segment. Chosen deliberately: it collapses
  per-call submissions (an nlmeans-class module is ~140 today) while
  keeping every inter-module contract — hand-off, cachelines,
  blending, taps, CPU fallback — bit-for-bit the eager behaviour,
  which made it honestly testable this session. Span capture needs a
  deferred-readback audit across every host consumer of the trunk
  (enumerated in the M2 milestone) and lands separately.
- **Pre-existing bug found and fixed** while auditing hand-off
  invariants for the integration: the CPU-tiling and blend-cache
  paths in `_pixelpipe_process_on_CPU` never invalidated the §5.14
  hand-off, so a following VK module with a matching size could
  consume the *previous* module's pixels. Fixed at one choke point
  (invalidate whenever a module's output was produced off the GPU
  hook). Exactly the §4A "whack-a-mole" class this design retires
  structurally.
- **Evidence.** Unit suite `test_vulkan_capture` (9 tests, lavapipe):
  captured chain bit-identical to eager and exactly 1 submission
  where eager takes 5; sync-tap; snapshot-vs-clobber; borrowed
  uploads; deferred-free non-aliasing; rollback; COPY/COPY_ROWS;
  abort; nested-begin refusal. Integration: `darktable-cli` PFM
  exports bit-identical with `pixelpipe_vulkan_graph` off vs on;
  `-d vkgraph` shows e.g. colorin as one 7-node segment (1 dispatch +
  6 uploads, 4.9 MB staged) in 1 submit. Full ctest green.

Next up (M2): the span-capture sync-point audit, then glue nodes;
blendop kernels (M0) can proceed in parallel.

## 12. Summary

- The eager, module-at-a-time execution model — not kernel speed — is
  what keeps darktable's GPU pipeline shuttling pixels over PCIe;
  ten distinct host touch points are enumerated in §1.1, and most of
  them cannot be fixed inside that model.
- darktable's pipe is already a DAG in disguise (blend diamonds,
  mask edges, taps, glue transforms). The proposal reifies it: keep
  the linear pipe, history, and UI untouched; behind them, *capture*
  each run into an explicit node graph, plan its memory like vkdt
  (liveness-aliased arena), and execute it as a handful of barriered
  Vulkan submissions with pixels resident in VRAM from source upload
  to sink readback.
- The capture trick — hooking the six HAL dispatch/copy entry points
  all 73 ported modules already use — means no module rewrites, an
  incremental landing path with a permanent eager fallback, and a
  scheduler that degrades gracefully (segment splits) instead of
  producing wrong pixels when a module needs the CPU. M1 (§11) proved
  this end-to-end: the executor landed with zero module changes and
  bit-identical output.
- Interactivity improves rather than survives: the pixelpipe cache
  gains a VRAM residency layer, so the steady-state slider drag
  becomes "record N..end, one submit, one fence" with zero trunk
  traffic.
- vkdt is the proof that the destination exists; this document is the
  route that gets darktable there without giving up its UI, its
  history stack, its module ecosystem, or its OpenCL-C kernel
  heritage.
