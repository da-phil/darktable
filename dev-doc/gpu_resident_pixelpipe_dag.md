# A GPU-resident pixelpipe: DAG execution behind the linear pipe

**Status:** in implementation — M1 (capture HAL + segment executor),
M0 (VK blend kernels, uniform-mask blending on-device), and M2 (run-
scoped span capture with deferred readbacks + colorspace/blend-space
glue nodes + verified format conversions) landed; a `colorin → … →
colorout` span captures as a single GPU submission. M5: the GPU
histogram and full-colorspace color-picker reduction kernels and the
deferred-readback tap registry are landed and validated, and the
histogram tap is wired into the pipe (validated headless through
levels-automatic; picker/scope wiring still wants a darkroom bench).
M3 started: the liveness peak-memory measurement is landed (the
aliasing planner wants a real GPU). Both the eager per-module VK path
and the graph span-capture path are validated on real AMD hardware
(RADV PHOENIX, §11): zero errors, and graph mode fuses ~90 % of GPU
dispatch work into large single-submit spans, bounded by the RAW/CFA
front-end (chiefly the still-missing real demosaic kernels) and a few
other CPU-island modules — a source audit (§11) sorts these into three
structurally distinct buckets. **Safety:** graph mode now carries the
§5.2 memory-budget gate it always specified — a full-res export on an
integrated GPU was accumulating an un-tileable multi-buffer span that
exhausted system RAM and froze the machine; runs whose trunk buffer
exceeds the single-buffer budget now decline graph mode and use the
eager path (§11, 2026-07-15 export-freeze entry). See §11 for the living
implementation log and §9 for per-milestone state. §1–§10 are the
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
                 && trunk buffer fits the single-buffer budget   // §5.10 #3
```

> **Status (implemented).** The budget term is live in
> `_vk_graph_fits_budget()` (`pixelpipe_hb.c`): a run whose estimated
> trunk buffer (`w·h·4·float`) exceeds `dt_get_singlebuffer_mem()` — the
> very threshold the OpenCL tiler uses to decide a buffer is too big to
> hold whole — declines graph mode and keeps the eager per-module path.
> Graph capture holds several such buffers live and cannot tile, so a
> single tiling-sized buffer already makes a span unsafe. This is what
> stops full-res export from exhausting RAM on an integrated GPU (§11,
> 2026-07-15 export-freeze entry). The other two terms are still the
> pref plus the debug-mode exclusions in `dt_dev_pixelpipe_process`.
>
> The **VK routing heuristic follows this decision**: the
> `vk_chain_live` / `vk_chain_ahead` look-ahead (~:2758) only routes a
> CL-capable module to VK when `pipe->vk_graph_run` is set, because the
> chain trade pays off only if the executor fuses the span. When the gate
> declines (oversized export), CL-capable modules stay on OpenCL — GPU
> compute + full GPU blend, as upstream — instead of staging every module
> through the host and CPU-blending (§11, "VK routing must follow the
> graph decision").

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
  **M2 status note (blend-space transforms landed).** The
  "optional blend-space transform nodes … into temporaries" sketched
  above are implemented (`bf4c024`): a blend colorspace differing
  from the buffers' colorspace no longer refuses — the buffers are
  converted into temporaries with the Lab↔RGB transform primitive,
  blended there, and the result copied into the output buffer, which
  ends up in the *blend* colorspace exactly as the CPU
  `_transform_for_blend` leaves `*output`; the hook mirrors the
  `pipe->dsc.cst` bookkeeping via a `blended_cst` out-parameter.
  Never in place, so any failure leaves the module output intact for
  the CPU fallback. Mismatches outside Lab↔RGB or without a matrix
  work profile still refuse. Validated function-level against
  darktable's own CPU transform + the independent blend references
  in both directions (`test_vulkan_blendop`).
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
  **M5 status note (two reduction kernels + registry landed).** Two
  tap kernels are ported, each validated against its CPU reducer:
  - `histogram_rgb` (`data/kernels/vulkan/histogram.cl`), driven by
    `dt_histogram_helper_vk` (`src/common/histogram.c`): plain `uint`
    atomic increments (native, no CAS — that idiom is only for *float*
    accumulation) binning a float4 image into a `bins_count·4`
    histogram, matching `dt_histogram_helper` for the RGB / Lab /
    Lab→LCh binnings (RAW and profile-compensated RGB refuse to the CPU
    path). Exact bin counts on rounding-invariant inputs
    (`test_vulkan_histogram`).
  - `picker_rgb` (`data/kernels/vulkan/picker.cl`), driven by
    `dt_color_picker_helper_vk` (`src/common/color_picker.c`): reduces
    the picker box to per-channel sum/min/max with *float* atomics
    (`vk_atomic_{add,min,max}_f` — the min/max siblings added here,
    reinterpret-CAS with an early return), host dividing the sum by the
    box size for the mean. Covers **all four picker colorspaces**: a
    mode switch in `picker_rgb` handles no-conversion
    (`_color_picker_rgb_or_lab`), Lab→LCH and RGB→HSL
    (`_color_picker_lch/hsl`, rotated-4th-channel hue handling), and a
    separate profile-bearing `picker_jzczhz` runs the full
    RGB→XYZ_D50→XYZ_D65→JzAzBz→JzCzhz chain for the scene-referred
    picker (`_color_picker_jzczhz`). Only denoise (the b-spline blur)
    and the no-matrix/lcms JzCzhz profile refuse to the CPU path.
    Validated against `dt_color_picker_helper` — min/max exact on the
    no-conversion path, within a ulp-tolerance on the converting paths
    (a different pixel may hold the extremum), mean within a
    summation-order tolerance (`test_vulkan_picker`, incl. a JzCzhz
    case with a constructed sRGB profile that exercises the whole
    chain in both toolchains).

  These are the reduction *primitives*; the readback is deferred to the
  end-of-run fence by the **tap registry** below.
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
    abort/rollback-drop). **The histogram tap is wired at §5.3.1
    site 2** (`dt_histogram_helper_vk_collect` in
    `_collect_histogram_on_CPU`): when the module input lives in the
    deferred hand-off, the reduction runs on the device and only the
    few-KB histogram is read back — the trunk stays resident and the
    span continues, where previously the request materialized the
    whole trunk. The readback is a *sync* tap by design, not
    registry-deferred: modules that consume their own input histogram
    inside `process` (levels automatic) need it before they run.
    Deferring through the registry is only valid for GUI-only
    consumers (`DT_REQUEST_ONLY_IN_GUI`) and remains a preview-pipe
    increment, as does wiring the picker at its request sites.
- **Pixelpipe-cache taps.** See §5.7.
- **Format conversions** (`bpp` changes along the RAW segment, 1f→4f
  at demosaic) are DISPATCH nodes like any other; the RAW trunk
  segment is just resources with different element sizes.
  **M2 status note (verified).** The capture HAL keys on byte sizes,
  not element counts, so a dispatch reading a 1-channel buffer and
  writing a 4-channel one captures and replays like any other — no
  special handling. Confirmed by `test_vulkan_capture`'s
  `test_mixed_bpp_format_conversion` (`capt_expand` 1f→4f then a 4f
  dispatch, bit-identical eager-vs-capture, one submission). A full
  raw-pipe capture (`rawprepare`/`demosaic` on device) still wants a
  real GPU + raw file, but the element-size mechanism it relies on is
  covered.

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

**M3 status note (liveness measurement landed).** Step 2 above (the
liveness sweep) and the peak it yields are implemented ahead of the
aliasing planner as a pure measurement: `dt_vulkan_capture_peak_bytes`
+ per-segment `peak-live` under `-d vkgraph` (`vulkan.c`
`_capture_peak_live_bytes`) compute the max simultaneous live device
bytes over each span's `[first, last]` intervals. It is read-only over
the node list — safe under today's fully-serialised executor — and
reports exactly what the deferred-free model keeps resident until
flush, i.e. what the aliasing arena (step 3) must fit. First real
numbers (benchmark XMP, 702×465 preview ROI): **node count doesn't
predict memory** — a 565-node span peaks at 34 MB (long chain, few
buffers live at once) while a 124-node span peaks at **128 MB** (many
image-sized buffers simultaneously live, e.g. a wavelet module's
scales). So the aliasing/spill target is the *high-simultaneity* span,
not the longest — and, scaled to full resolution, the 128 MB peak
alone would exceed VRAM on many cards, which is the concrete
justification for the arena + spill work. Unit-tested against
hand-computed peaks from the buffers' actual sizes
(`test_vulkan_capture::test_liveness_peak`). The aliasing planner
(steps 3–5) stays for a real-GPU context where the VRAM win is
measurable; the safe recycling that could ride today's conservative
barriers is deferred with it (the rollback interaction and the
zero-measurable-benefit here don't justify the correctness risk).

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
   budget). ✅ **implemented** (`_vk_graph_fits_budget`): the coarse
   form of this gate — decline graph mode whenever a single trunk
   buffer already exceeds the single-buffer budget — is what the M1/M2
   executor ships today, since it has neither segment spill (#1) nor
   graph-level tiling (#2). It is what keeps a 24 MP+ export from
   accumulating a multi-buffer span that thrashes an integrated GPU's
   shared RAM (§11, 2026-07-15). Spill/tiling would *raise* this
   ceiling; until they land, the ceiling is the safety net.

Rule of thumb from §5.5's estimate: 24 MP export fits 8 GB cards
without spilling; 61 MP on a 4 GB card takes 1–2 spills; the darkroom
screen pipes essentially never spill.

#### 5.10.1 Whole-pipe output tiling (the `ME` export mechanism)

The form of graph-level tiling (#2 above) chosen for the export
milestone, because it reuses machinery that already exists rather than
building a new intra-executor tiler:

- **darktable already processes an arbitrary output region.**
  `dt_dev_pixelpipe_process(pipe, dev, x, y, w, h, scale, devid)`
  (`pixelpipe_hb.c`) runs the pipe for region `(x,y,w,h)`; darkroom uses
  it for the viewport, export calls it once with the full image
  (`imageio.c`). Call it once **per output tile** instead and each
  tile-run is an independent small pipe.
- **Halos come for free.** Each module's `modify_roi_in` adds its
  overlap as the ROI propagates backward, so a tile's haloed input
  region is pulled automatically — no manual halo bookkeeping in the
  driver.
- **Memory becomes image-size-independent.** A tile-run's graph capture
  operates at *tile*-sized buffers, so it passes the §5.2 budget gate
  and fuses; peak-live is bounded by the tile, not the image. This is
  what makes full-res export *possible* on the graph.
- **Tiling wraps at the pipe-run level, not inside the executor** — the
  capture executor (`dt_vulkan_capture_flush`) is unchanged; it just
  gets handed tile-sized work. Same layering as the OpenCL tiler, which
  wraps `process_cl` per tile.

Grid/overlap/alignment and tile assembly reuse the algorithm in
`_default_process_tiling_cl_roi` (`tiling.c`): the whole-pipe overlap is
`Σ` each enabled piece's `tiling_callback` overlap (clamped), `align` the
lcm of per-module aligns, `factor` the graph peak-live (§5.5). Cost: the
whole pipe (incl. demosaic) recomputes per tile over the halo — the same
tiled cost OpenCL pays, coarser-grained; large tiles keep the overlap
fraction small. Pathological Σoverlap (deep neighbourhood stacks) falls
back to OpenCL via the gate's third branch (§5.2 / #3).

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

**Hardware confirmation (2026-07-15, AMD RADV, §11).** The same shape
holds on real silicon and names the islands. A graph-mode darkroom +
export session fused **90 % of GPU dispatch work (732 / 817 dispatches)
into 30 large spans**, the biggest 91 nodes / 76 dispatches in one
submit — but not into a single end-to-end span, because the seven
unported modules (`highlights`, `demosaic`, `toneequal`,
`denoiseprofile`, `gamma`, `rawprepare`, `channelmixerrgb`) rolled back
30× and broke the chain, exactly as the lavapipe run predicted. A
source audit of *why* each rolls back (§11, 2026-07-15) then showed the
seven are three structurally different problems — the RAW/CFA front-end
(led by the missing real demosaic kernels), two 4-channel gate-widenings,
and two net-new ports — not the flat frequency-ordered list the count
first suggested; the HAL substrate is not the blocker. A caution the
hardware added: the count-weighted view is
misleading — 70 % of *flushes* are trivial 2-node readback servers, so
span health has to be read dispatch-weighted, not flush-weighted.

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
  the **module-input colorspace glue node** (§5.4 status note:
  Lab↔RGB matrix transforms on-device, so a `colorin → … → colorout`
  span now captures as one submission; float-precision equivalent to
  eager, kernels validated against the CPU transform), and the
  **blend-space transforms** (§5.4 blending status note: a blend
  colorspace differing from the buffers no longer forces the CPU
  blend — the `_transform_for_blend` step runs on-device into
  temporaries, validated function-level in both directions).
  Format conversions (1f↔4f along the RAW segment) are verified at
  the HAL level (§5.4 status note: mixed-bpp capture is bit-identical
  to eager) — a full raw-pipe capture still wants a real GPU + raw
  file, but nothing DAG-specific is missing. **M2's glue is complete.**
  The trunk needs no output-tail RGB↔RGB glue node: `colorout` already
  performs the working→output-profile transform *inside* its
  `process_vk` (the `_rgb_vk` primitive is used *within* modules like
  `overexposed`, not as a pipe glue), and the working profile is
  constant between `colorin` and `colorout` so interior site-1
  transforms are same-profile no-ops. A full raw-pipe capture on a
  real GPU is the only M2-adjacent validation left.
- **M3 — memory planner.** Liveness + arena + aliasing + budget +
  spill-by-segmentation. Retire per-dispatch pool churn in graph mode.
  *Started:* the **liveness measurement** is landed (§5.5 status note:
  `dt_vulkan_capture_peak_bytes` + `-d vkgraph` peak-live, unit-tested)
  — the number the arena/spill work is sized from, and its first real
  data already reprioritises which span to alias (high-simultaneity,
  not longest). The arena + aliasing + spill remain a real-GPU job
  (the VRAM win is unmeasurable on this software-Vulkan bench).
- **M4 — cache integration.** Async cache-tap readbacks; VRAM
  residency pins; plan reuse via topology hash.
- **M5 — taps on GPU.** Histogram/picker/scope reduction kernels +
  end-of-run KB readbacks; scharr as resident resource. Flip pref
  default on for darkroom pipes; export next release. *In progress:*
  the **histogram** and **color-picker** reduction kernels (each
  validated against its CPU reducer, picker covering all four picker
  colorspaces), the **tap registry** (`dt_vulkan_tap_register`), and —
  landed after the "needs a darkroom bench" assessment was revisited —
  the **histogram tap wired at §5.3.1 site 2**, validated headless end
  to end through levels-automatic (which self-consumes its input
  histogram in export; §11 entry with the determinism analysis).
  Remaining: the registry-deferred variant for GUI-only histogram
  consumers, the picker wiring at its request sites, and the scope
  reductions — those genuinely are preview/darkroom behaviour — plus
  the denoise picker blur.
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

### Priority milestone — `ME`: GPU-resident export

**Reprioritised 2026-07-15 at user request:** *export*, not interactive
darkroom (which is fine on OpenCL), is the target. This pulls the
graph-level tiling out of M7 (opportunistic) onto the critical path and
folds in Path-B demosaic and M0+ blend as coverage. Two decisions taken
with the user: **success = parity at any resolution** (correct vs the
OpenCL/CPU reference, memory-safe, competitive perf — *not* beating
OpenCL, which is already resident+tiled; the win is backend unification
+ future fusion/aliasing headroom), and **port demosaic to VK now** (no
permanent OpenCL island mid-pipe).

Mechanism (see §5.10.1): **whole-pipe output tiling** — split the export
output into budget-sized tiles and run the pipe per tile via the
existing region path (`dt_dev_pixelpipe_process(x,y,w,h)`); each tile's
graph capture then operates at tile-sized buffers → fits the §5.2 budget
→ fuses. darktable's ROI back-propagation pulls each tile's haloed input
for free. Tiling wraps at the pipe-run level, like the OpenCL tiler —
the capture executor is unchanged.

- **ME.0 — export parity harness + baseline.** Headless: export one edit
  via OpenCL and via the graph path, diff the output (fp tolerance);
  record the OpenCL wall-time as the perf bar. Small-image half runs on
  lavapipe. The gate for every later stage.
- **ME.1 — whole-pipe output tiling** *(critical path)*. Tile driver +
  grid/overlap/align math (reuse `_default_process_tiling_cl_roi`,
  `tiling.c`), whole-pipe overlap/factor accumulated from each piece's
  `tiling_callback`, tile assembly.
- **ME.2 — flip the budget gate to tile.** `_vk_graph_fits_budget` →
  `_vk_graph_tile_plan`: fits → 1 tile; oversized → N tiles ≤ budget;
  un-tileable overlap → decline to OpenCL (the existing safe branch).
- **ME.3 — demosaic VK kernels** (task #22): RCD + Markesteijn first,
  then PPG/VNG; widen the mosaic front-end gates (`rawprepare`,
  `highlights`; task #21). Un-islands sensor → RGBA.
- **ME.4 — M0+ on-device blend** (task #25): parametric/drawn masks +
  common modes, so blend-heavy export modules stay in-span.
- **ME.5 — straggler islands** (`toneequal`, `gamma`): cheap OpenCL/CPU
  boundaries first, promote later.
- **ME.6 — parity gate + enable.** Bit-parity across a corpus + perf ≈
  OpenCL + clean under the validation layers (task #23) → enable
  export-via-graph by default; until then behind `pixelpipe_vulkan_graph`.

Critical path: ME.0 → **ME.1 + ME.2** (this is what makes full-res
export *possible and memory-safe* on the graph — a demosaic'd full-res
span still won't fit RAM without tiling) → ME.3 + ME.4 (coverage, in
parallel) → ME.5/ME.6. Memory aliasing (M3) is a deferred perf lever,
not on the parity path.

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

### 2026-07-15 — plan set: GPU-resident export (`ME`) is the priority

The user reprioritised: make *export* run on the graph Vulkan pipeline;
interactive darkroom is fine on OpenCL and is not the target. Plan
recorded in §9 (`ME` milestone) and §5.10.1 (mechanism). Two decisions
taken with them:

- **Success = parity at any resolution**, not beating OpenCL. Upstream's
  OpenCL export is already GPU-resident and tiled, so the graph's
  near-term value is backend unification + future fusion/aliasing
  headroom. Done = correct (bit-parity vs OpenCL/CPU), memory-safe at any
  resolution, competitive perf, enable-able by default. Aliasing arena
  (M3) is a deferred perf lever, off the parity path.
- **Port demosaic to VK now** — the whole pipe from the sensor should
  form spans; no permanent OpenCL island mid-export.

Architecture: **whole-pipe output tiling** (§5.10.1) — tile the export
output, run the pipe per tile via the existing region path, let ROI
back-prop pull each tile's halo, capture each tile at tile-sized buffers
so it fits the §5.2 budget and fuses. Reuses `dt_dev_pixelpipe_process`
region processing + `_default_process_tiling_cl_roi`'s grid math; the
capture executor is untouched. Critical path: ME.0 harness → ME.1/ME.2
tiling + gate-flip (the memory enabler) → ME.3 demosaic + ME.4 blend
(coverage) → ME.6 parity gate. No code this entry — plan + task
restructure only; implementation lands stage by stage with a runnable
check each, per the standing directive.

### 2026-07-15 — export still 5× slower after the gate: VK routing must follow the graph decision

With the budget gate in place the freeze was gone, but export was still
~5× slower than upstream (63 s vs ~13 s) and littered with "blended on
CPU". The follow-up log shows the gate firing correctly
(`graph capture declined: trunk ~865 MB > budget 385 MB`) — and then a
second, independent bug underneath it.

- **What the log shows.** Every GPU module carries a ~2 s floor even for
  point ops (`crop` 2.5 s, `colorin` 2.3 s, `primaries` 2.3 s,
  `finalscale` 2.1 s), and three modules with non-trivial masks —
  `exposure.1` (8.6 s / 37 CPU-s), `exposure.2` (6.9 s), `colorequal`
  (15.5 s / 25 CPU-s) — report "blended on CPU". Those three alone are
  ~31 s of the 63 s.

- **Root cause: the VK routing heuristic ignored whether the graph would
  actually run.** The §5.2 router (`pixelpipe_hb.c` ~:2758) sends a
  CL-capable module to VK whenever a VK chain is live or ≥2 VK-ready
  modules lie ahead (`vk_chain_live` / `vk_chain_ahead`). That trade is
  only sound **if the graph executor fuses the span and skips the
  per-module readbacks**. The budget gate had just turned graph mode
  *off* for this export — so eager VK still (a) reads the whole 865 MB
  trunk back to host after every module (the ~2 s floor, and it breaks
  the OpenCL `cl_mem` residency chain upstream keeps end-to-end), and
  (b) can only blend the M0 subset on device, so every parametric/drawn-
  mask blend falls back to the CPU (the "blended on CPU" lines, and the
  25–37 CPU-s that dominate the run). Upstream keeps these modules in
  `cl_mem` on the OpenCL arm with **full GPU blend** — hence 5×.

- **The fix (this commit).** Gate the chain heuristics on
  `pipe->vk_graph_run`: `chain_worthwhile = vk_graph_run && (vk_chain_live
  || vk_chain_ahead)`. When the graph won't run, a CL-capable module is
  no longer routed to VK — it stays on OpenCL (GPU compute + full GPU
  blend), exactly like upstream. VK is still taken when it is the only
  GPU option (`vk_is_only_gpu`) or the user forces it
  (`opencl_force_vulkan_routing`). Small-buffer runs where the graph
  *does* run are unchanged: chains still form and fuse. So the routing
  now tracks the executor: chain to VK only when something will fuse the
  chain; otherwise stay resident on CL.

- **On "blended on CPU".** It means: the module computed on the GPU, but
  its blend wasn't in the on-device VK blend subset (M0: normal mode,
  uniform mask; §9/§11 M0), so darktable read the result back and blended
  on the CPU. After this fix the *export* path routes those modules to
  OpenCL and blends on the GPU, so the message disappears there. In
  *darkroom* (graph on, small buffers) a module with a parametric or
  drawn mask can still show it, because VK's blend is still M0-only —
  widening that subset (M0+) is tracked separately; until then, OpenCL is
  the full-featured GPU-blend path.

### 2026-07-15 — export near-froze the machine: the missing budget gate, now implemented

A user ran a full-res export with `pixelpipe_vulkan_graph=TRUE` and it
"almost crashed my laptop" — darktable ate all RAM and every CPU core
and froze the system; an earlier run reported memory corruption; and it
felt "very sluggish, nothing in comparison to upstream master." The
attached export log is unambiguous, and it exposes a real design gap:
**graph mode had no memory budget gate**, so at export resolution the
span accumulates more resident memory than the machine has.

- **What the log shows.** The hardware is an AMD **PHOENIX APU** —
  integrated, so device "VRAM" *is* system RAM. Export trunk buffers are
  **865 MB** each (vs 53 MB for the darkroom preview in the same log);
  the liveness sweep reports **peak-live 1791 MB**, i.e. ~2 whole trunk
  buffers live at once, and the upload ring stages another ~865 MB.
  `exposure.1` alone took **55.4 s wall / 45.6 CPU-s** ("blended on
  CPU"), and there is a ~40 s stretch (t≈111→151 s) with nothing logged
  but histogram redraws. That profile — wall ≫ useful work, a single
  memcpy-scale operation costing 45 CPU-s, a dead 40 s gap — is swap
  thrashing: the resident set (graph device buffers on shared RAM + the
  ring + the host CPU-blend copies + the pipe cache) exceeded physical
  RAM and the box went to swap.

- **Why it's so much worse than upstream.** Upstream's OpenCL path
  **tiles** large buffers and reads back + frees each module before the
  next; it never holds a whole-pipe span of 865 MB buffers. Graph mode
  holds several live (deferred frees + §5.14 hand-off + deferred
  readbacks) and **cannot tile** (§5.10). On a discrete card with spare
  VRAM that is merely aggressive; on an APU it is fatal, because every
  one of those buffers is system RAM competing with the host copies.

- **Root cause = a gate that was designed but never built.** §5.2's
  graph gate always specified a "predicted arena vs budget" term (§5.10
  fallback #3), but the M1/M2 executor shipped the gate as *pref +
  debug-exclusions only* — no size check. So graph mode engaged for
  **any** buffer size, including full-res export.

- **The fix (this commit).** `_vk_graph_fits_budget()` in
  `pixelpipe_hb.c` implements the missing term: decline graph mode when
  the estimated trunk buffer (`w·h·4·float`) exceeds
  `dt_get_singlebuffer_mem()` — the exact threshold darktable's OpenCL
  tiler uses to decide a buffer is too big to hold whole. A span holds
  several such buffers and can't tile, so one tiling-sized buffer
  already makes the span unsafe. Oversized runs fall back to the eager
  per-module path (read back + free each module before the next: bounded
  to ~2 buffers, no ring, no span). The threshold scales with the user's
  resource level, so it needs no magic constant, and it declines to the
  log: `[vkgraph] graph capture declined: trunk ~N MB > single-buffer
  budget M MB`. Darkroom preview (53 MB) is far under any real budget and
  keeps graph mode; export (865 MB) declines. Compiles clean; the guard
  is pure arithmetic over existing budgeting, exercised on every pipe
  run. I could not re-run on the APU (no such hardware here), so this is
  validated by build + the log's own numbers, not by a fresh capture.

- **On the reported corruption.** I audited the capture lifecycle most
  likely to corrupt — `capture_mark` / `capture_rollback` /
  `_capture_truncate` and the deferred-free list — and it holds up on
  inspection: the mark snapshots `nodes`/`dfree`/`ring`/`taps` lengths,
  rollback truncates to them and releases only post-mark frees, and
  vin/vout frees defer correctly past pending nodes. I did not find a
  clear defect by reading, and without a reproducer on the affected
  hardware I will not guess-patch memory-safety code. The budget gate
  removes the memory-pressure regime where a latent overrun is most
  likely to bite (near-OOM allocation failures that the capture paths do
  not all check). Tracked as its own task for a hardware repro under a
  validation layer.

- **What this does *not* fix.** Darkroom interactive is still bounded by
  the CPU-island round-trips (the demosaic/highlights/etc. rollbacks of
  the entries below) — that's the Path-B work, unchanged. And large-
  buffer *support* (segment spill §5.10 #1, graph tiling #2) is still
  future work; today large runs simply decline graph mode rather than
  spilling. The gate is the safety net under that unfinished ceiling.

### 2026-07-15 — root-cause audit of the 30 rollbacks: three buckets, not a frequency list

Follow-up to the graph-path entry below, which named the seven rollback
modules and proposed porting them "in rollback-frequency order." A source
audit of *why* each rolls back shows that ordering is a **count artifact**
and the seven are structurally three different problems. No code change —
this corrects the priority framing (and Task #21) to match the code.

- **Rollback frequency measures how often a module runs, not how hard it is
  to port or what blocks it.** The RAW front-end modules run once per pipe
  (their mask/preview variants add counts), so they top the count; that says
  nothing about the port. Read structurally instead.

- **Bucket A — the RAW/CFA front-end (1-channel mosaic domain):
  `rawprepare`, `highlights`, `demosaic`.** These run before demosaic emits
  RGBA, on single-channel CFA data. Crucially this is **not** a HAL/substrate
  limit:
  - the capture HAL already records 1-channel and 1ch→4ch nodes — the
    `capt_expand` test (Task #20) is literally a "demosaic (1f → 4f)" node
    (`src/tests/unittests/common/kernels/capt_expand.comp`); and
  - `temperature`/whitebalance already runs mosaic on VK in production
    (`vk_1f` / `vk_1f_xtrans`, single-channel float buffers,
    `src/iop/temperature.c:696`), sitting *inside* the front-end but
    islanded because its neighbours roll back.

  The island therefore comes from module-level choices, not the graph core:
  `rawprepare` (`src/iop/rawprepare.c:805`) and `highlights`
  (`src/iop/highlights.c:1088`) gate their `process_vk` OFF for
  `dsc.filters` (mosaic) — feasible to widen, temperature is the template —
  and `demosaic` (`src/iop/demosaic.c:1284`, `:1293`) ships only
  passthrough/mono kernels at full scale, bailing `"bayer/xtrans method not
  ported"` / `"zoom path not ported"` for every colour RAW and every
  reduced-scale (darkroom-preview / thumbnail) pass. The **real demosaic
  algorithm kernels are the single largest missing piece** and the
  highest-payoff lever: they un-island the whole `rawprepare → temperature →
  highlights → demosaic → …` front-end into one span.

- **Bucket B — 4-channel modules with narrow gates: `channelmixerrgb`,
  `denoiseprofile`.** Post-demosaic RGBA, close to portable.
  `channelmixerrgb` (`src/iop/channelmixerrgb.c:3330`, `:3346`) only rolls
  back when the needed adaptation `.spv` didn't load (glslang-only builds
  keep just linear-Bradford) or `colors != 4` — a **build-config** artifact
  a full clspv build removes with *zero code change*. `denoiseprofile`
  (`src/iop/denoiseprofile.c:2892`–`:2905`) supports wavelets + RGB +
  new-VST and bails NLM / Y0U0V0 / legacy-VST — a genuine mode-gate to widen.

- **Bucket C — no GPU path at all: `gamma`, `toneequal`.** Both have no
  `process_cl` (hence no `process_vk`); never GPU-ported by any effort.
  `gamma` is the trivial final 8-bit display encode; `toneequal` is a
  guided-filter module. These are net-new GPU ports unrelated to the DAG
  substrate, and the lowest DAG priority.

- **Corrected lever order (by span-merge payoff, not count):** (1) demosaic
  real-algorithm kernels — biggest single merge, largest effort, needs real
  GPU; (2) `rawprepare` + `highlights` mosaic gate-widening — cheap given
  temperature's template, but only pays once demosaic-VK exists (else the
  span still breaks at demosaic), so sequenced after (1); (3)
  `denoiseprofile` mode-gate widening — independent 4-channel win; (4)
  `channelmixerrgb` — free on a clspv build, no code; (5) `gamma` /
  `toneequal` — separate net-new GPU ports, lowest priority. Task #21
  rewritten to this; Task #22 added for the demosaic kernels.

### 2026-07-15 — graph path validated on AMD RADV; the "short spans" read corrected

The re-run the eager entry asked for: same build (`g3056c0b`,
darktable 5.7.0+193, dirty) and hardware (AMD RADV PHOENIX), now with
`--conf pixelpipe_vulkan_graph=TRUE -d vkgraph -d perf`. This is the
first time the DAG span-capture path — not just the per-module eager
hook — has run on non-lavapipe silicon. Analysed from the attached log;
no code change (the log revealed no bug).

- **Stability + fault path.** 377 `[vkgraph]` lines, **zero** errors /
  asserts / `VK_ERROR`s. 100 flushes, 817 dispatches, 1394 nodes across
  darkroom-full, preview and 16 export jobs. The rollback fault path
  fired **30×** — each a clean single-node drop of a module the capture
  HAL declined to record (`highlights` ×3, `toneequal`/`gamma`/
  `denoiseprofile`/`demosaic` ×2 each, `rawprepare`/`channelmixerrgb`
  ×1) — and every one recovered to eager execution with correct output.
  The conservative-rollback design (§5.11) works on real hardware:
  correctness over fusion, no capture ever corrupted a run.
- **The DAG fuses the bulk of the work — my mid-analysis "spans are
  short" note was wrong, and this corrects it.** That read was
  *count-weighted*: 70 of 100 flushes are trivial 2-node readback
  servers (19 of them the 244 MB full-res input upload staged to serve
  a preview/export readback — a pipe boundary, not an intra-pipe
  stall), so by flush-count the small ones dominate. **Work-weighted**
  tells the real story: **30 large spans (≥10 nodes) carry 732 of 817
  dispatches = 90 % of all GPU dispatch work**, and they occur across
  every pipe type (7 export, 4 darkroom-full, 5 preview at ≥40 nodes).
  The largest single span is **91 nodes / 76 dispatches → 1 submit +
  1 fence** — 76 kernel launches fused into one command buffer, exactly
  the design goal. The 20 cst glue nodes and 20 deferred readbacks
  confirm the M2 span machinery runs on hardware.
- **What still breaks the spans is the CPU islands, now named.** The 30
  rollbacks + the sync-at-need breaks are the seven modules above —
  the same demosaic/highlights/toneequal/denoiseprofile set the eager
  run flagged as unported. They interrupt the top of the pipe, so the
  big fused spans form in the mid/late stack *after* the island, which
  is why 90 % of the work fuses but not into a single end-to-end span.
  This is §8.1's "CPU islands bound span length" finding, now confirmed
  on silicon and with the offending modules enumerated: they are the
  Path-B port targets. (Priority correction: the audit entry above shows
  rollback frequency is a count artifact, not the port order — the real
  order is payoff-ranked across three buckets.)
- **No `peak-live` data.** Build `g3056c0b` predates the liveness commit
  (`3c5c4af`); 0 peak-live lines, as expected. M3 measurement wants a
  build at `c158eb8`+.
- **Perf: no head-to-head claim from these logs.** The eager and graph
  logs are two *different* interactive sessions (different images,
  edits, zoom), so per-module wall sums aren't comparable and I won't
  quote a speedup. The honest signal is structural: on this pipe the
  DAG already fuses 90 % of dispatch work into 30 submits instead of
  817, and the residual 120 sync taps map one-to-one onto the seven
  unported modules. The measurable win is now gated on porting those,
  not on more DAG-core work — the same conclusion §8.1 drew on lavapipe,
  now with the hardware to back it and the module list to act on.

### 2026-07-15 — first real-hardware run: eager VK path validated on AMD RADV

A user darkroom session (`-d opencl -d perf`, build `g3056c0b`) on a
real AMD Radeon (RADV PHOENIX, gfx1103, 10 GB) with
`opencl_force_vulkan_routing=true` — the first time this branch has run
on non-lavapipe hardware. Analysed from the attached log; no code
change.

- **Stability.** Zero errors / asserts / `VK_ERROR`s across 892
  thumbnail + 41 preview + 210 full-pipe module invocations on a
  ~60 MP raw. The VK kernels and HAL that lavapipe validated run clean
  on RADV too — meaningful de-risking of every ported kernel. (The two
  `[opencl_events_flush] invalid profiling data` lines are a benign CL
  event-timer quirk, not VK.)
- **M0 blend on real hardware.** `channelmixerrgb` (uniform mask)
  reported `blended on GPU`; every other blended module used a
  drawn/parametric mask and correctly refused to the CPU path — the M0
  gate behaving exactly as designed, now confirmed on hardware.
- **Coverage.** Only the expected unported paths fell back: demosaic
  bayer/xtrans, denoiseprofile Y0U0V0, retouch HEAL, toneequal, gamma.
- **The graph path was NOT exercised** — `pixelpipe_vulkan_graph` was
  at its default (off), so span capture / glue nodes / histogram tap /
  `peak-live` never fired. The log is the *eager* per-module VK path,
  and its perf makes the DAG's case concretely: parametric-masked
  modules cost full-res readbacks + CPU blends (e.g. `exposure.2`
  1.9 s wall / **21 s CPU**, `colorequal` 11 s CPU), and every module
  pays its own upload/process/readback with no fusion (~3.7 s per
  full reprocess). Validating the DAG itself on this hardware needs a
  re-run with the pref enabled and `-d vkgraph`. Recorded as the
  concrete next step for whoever has the hardware.

### 2026-07-14 — M3 liveness measurement landed; M2 output-tail claim corrected

Commits: `3c5c4af` (liveness peak + test), doc.

- **Liveness peak-memory measurement** (M3 step one). `-d vkgraph` now
  reports per-segment `peak-live` MB, and `dt_vulkan_capture_peak_bytes`
  exposes the run's peak. The deep-pipe numbers immediately paid off:
  the 565-node span holds only 34 MB, but a 124-node span holds
  128 MB — node count doesn't predict memory, simultaneity does, so
  the aliasing/spill target is the high-simultaneity span. Details in
  the §5.5 status note. The test caught a real subtlety worth noting:
  the buffer pool hands back *oversized* buffers, so the peak counts
  actual `->size`, and the test computes its expectation from the
  buffers' real sizes rather than assuming equal-size allocations
  (the first cut asserted `chain == 2 × single` and failed by exactly
  one pool-oversized buffer, 256 B — the measurement was right, the
  test's assumption wasn't).
- **Doc correction.** A prior entry called the RGB↔RGB output-tail
  transform "the only strictly-remaining M2 piece." That was wrong:
  `colorout` already performs the working→output-profile transform
  *inside* its `process_vk`, the `_rgb_vk` primitive is used within
  modules (`overexposed`) not as a pipe glue, and the working profile
  is constant colorin→colorout so interior site-1 transforms are
  same-profile no-ops. M2's glue is complete; §9/§5.4 corrected to
  match. (Keeping the doc honest is the point — the claim was an
  overstatement, not a missing feature.)

### 2026-07-14 — M2 format-conversion mechanism verified; module-port viability noted

Commit: `d418988` (mixed-bpp capture test).

Closes the last buildable M2 glue item. `test_mixed_bpp_format_conversion`
runs a 1-channel→4-channel dispatch (`capt_expand`, the demosaic
pattern) followed by a 4-channel dispatch and asserts the captured
mixed-bpp span is bit-identical to eager in one submission —
confirming the design claim that the RAW segment's element-size
change needs no special handling (the HAL tracks bytes, not
elements). A full raw-pipe capture stays for a real GPU + raw file.

**Module-port viability, recorded for the next lane.** With the DAG
core done, the frontier is CPU-island modules (the §8.1 finding). An
audit of the benchmark pipe's CPU-only enabled modules found none is
a clean next port *on this bench*: the ones with an existing CL path
are already VK-ported; of the rest, `grain` uses double-precision
3D simplex noise (a float GPU port diverges by the grain amplitude,
not float-precision, so the CPU-comparison validation every port has
relied on doesn't apply), `toneequal` is a large guided-filter
pyramid, `dither` is sequential/output-stage, and `diffuse` is
explicitly M7 (`create_nodes_vk`). So further un-islanding is real
work but not *cleanly testable here* — it belongs with the Path B
lane on a real GPU, where statistical/visual validation of
noise-class modules is acceptable.

### 2026-07-14 — M2 blend-space transforms landed: device blend survives cst mismatches

Commit: `bf4c024` (widened gate + hook bookkeeping + tests).

`dt_develop_blend_process_vk` no longer refuses when the blend
colorspace differs from the buffers' colorspace: the CPU
`_transform_for_blend` step runs on the device instead, reusing the
M2 Lab↔RGB transform primitive. Design points:

- **Temporaries, never in place.** Both buffers convert into fresh
  buffers, the blend runs there, and one device copy lands the result
  in `dev_out` — the only write to `dev_out` on this path, so a
  failure at any earlier step leaves the module output intact and the
  CPU fallback stays correct (an in-place variant would have
  corrupted `dev_out` on a mid-sequence failure, poisoning the
  fallback).
- **Colorspace bookkeeping mirrors the CPU path.** The blended output
  is in the *blend* colorspace, like the CPU path's `*output`; a new
  `blended_cst` out-parameter carries that to the hook, which sets
  `pipe->dsc.cst` accordingly, so the next module's input transform
  (site 1 / cst glue) sees the truth. In the matched (M0) case the
  value equals the module output colorspace and the override is a
  no-op.
- **Refusals kept:** mismatches outside Lab↔RGB, and pipes without a
  matrix work profile (the function-level gate test now documents
  that it refuses *for lack of a profile*).
- **Evidence.** Two new function-level tests reference the device
  path against darktable's own CPU
  `dt_ioppr_transform_image_colorspace` on host copies plus the
  independent blend references: RGB buffers blending in Lab, and Lab
  buffers blending in scene-RGB (jzczhz), both asserting the reported
  `blended_cst`. Full ctest green (9 blendop tests); the deep-pipe
  export A/B is unchanged from the pre-change baseline — the
  benchmark pipe's blends all use matching colorspaces, so the glue
  path correctly stays idle there and the unit tests carry its
  coverage.

With this, a module whose user-selected blend space differs from its
output space stops being a guaranteed span boundary (audit sites
10–11 fire only for masks outside the uniform subset now, not for
mere colorspace mismatches).

### 2026-07-14 — M5 histogram tap wired into the pipe, validated headless

Commit: `f81ab14` (collect-level helper + site-2 wiring + tests).

The "M5c needs a darkroom bench" assessment was wrong for the
histogram, and revisiting it found a headless testbed:
**levels in automatic mode** clears `DT_REQUEST_ONLY_IN_GUI` in
non-GUI runs (`levels.c` `commit_params`), collects a 16384-bin Lab
histogram of its input in *export*, and computes its black/grey/white
points from it inside `process` — its pixel output is a direct
function of the histogram. Flipping the benchmark XMP's levels entry
to automatic (params hex: mode 0→1, percentiles 5/50/95) exercises
the whole wiring in `darktable-cli`.

What shipped:

- `dt_histogram_helper_vk_collect` (`src/common/histogram.c`): the
  collect-level sibling of `dt_histogram_helper` — same buffer
  (re)allocation contract, stats fill, and channel-maxima rules, with
  the reduction on the device. Unit-tested against the CPU helper for
  the *whole* contract (bins, maxima, stats), exact on
  rounding-invariant inputs.
- Site-2 wiring in `_collect_histogram_on_CPU`: input in the deferred
  hand-off → device reduction, few-KB readback, trunk stays resident,
  hand-off survives, span continues. Unported cases fall through to
  sync-at-need + CPU. `PIXELPIPE_FLOW_HISTOGRAM_ON_GPU` reports it.
- **Sync by design, not registry-deferred:** the result must land in
  `piece->histogram` *before* `module->process` for self-consumers
  like levels-auto. The registry-deferred variant is only valid for
  GUI-only consumers and stays a preview-pipe increment.

Evidence, including a determinism analysis worth keeping:

- `-d vkgraph` shows `histogram tap: 16384 bins on device, trunk
  stays resident` between two readback-deferred VK modules, and
  levels itself then rides the hand-off (`[vk handoff] … [readback
  deferred]`); `-d perf` reports `collected histogram on GPU`.
- **Repeated runs of the same config differ by up to ~6e-4** in this
  deep pipe — *pre-existing* OMP accumulation noise (present in
  eager runs without any graph code) amplified by levels-auto's
  percentile step function. Graph runs are exactly as repeatable as
  eager runs (max 5.1e-4 vs 6.3e-4 across repeats).
- Eager-vs-graph differs stably by max 4.6e-2 on 13 of 979k pixels
  (0.001%), clustered near the white point with sign flips — the
  signature of the 95th-percentile threshold landing one bin apart
  between the two float paths, not a bias (the unit tests pin the
  kernel to exact bin counts, and per-channel totals always equal the
  pixel count). This is the same divergence class darktable already
  accepts between its CPU and CL paths; any percentile-driven module
  amplifies CPU↔GPU heterogeneity this way regardless of where the
  histogram is computed.

### 2026-07-13 — M5 second tap landed: GPU color-picker reduction kernel

Commit: `cf6be88` (kernel + host dispatcher + float atomic min/max + test).

`picker_rgb` reduces the picker box to per-channel sum/min/max with
float atomics; `dt_color_picker_helper_vk` drives it and divides the
sum by the box size for the mean. Matches the CPU
`_color_picker_rgb_or_lab` — the common no-conversion picker path;
converting pickers (LCH/HSL/JzCzhz), the denoise blur, 1-ch raw, and
empty/out-of-bounds boxes all refuse to the CPU reducer.

The sum reused `vk_atomic_add_f`; min/max needed float atomics too, so
`vk_atomic_min_f` / `vk_atomic_max_f` were added next to it in
`dt_vulkan_common.h` (reinterpret-CAS with an early return, so an
uncontended lane touches memory only through the initial atomic read).
The GLSL twin carries the `atomicCompSwap` equivalents.

Validated in `test_vulkan_picker` against `dt_color_picker_helper`:
min/max are order-independent → exact; the mean is compared with a
scale-aware tolerance because the atomic accumulation order differs
from the OMP reduction. Full box, sub-box, and a 1×1 box
(mean == min == max) covered, plus the gate refusals.

Two follow-ups complete the picker's colorspace coverage. `14843f0`
adds a mode switch for the Lab→LCH and RGB→HSL pickers
(`_color_picker_lch/hsl`): convert, set the 4th channel to the rotated
3rd (hue-wraparound handling, matching `_update_stats_4ch`), then
reduce. `236477e` adds the scene-referred JzCzhz picker as a separate
profile-bearing kernel (`picker_jzczhz`, 4 bindings) running the full
RGB→XYZ_D50→XYZ_D65→JzAzBz→JzCzhz chain, with `dt_color_picker_helper_vk`
gaining a profile argument and the `dt_ioppr_build_iccprofile_params_vk`
plumbing; the no-matrix/lcms profile refuses to the CPU path. On all
converting paths the GPU/CPU conversions differ by a few ulp so a
different pixel can hold the extremum — the test compares min/max
within a tolerance there while keeping the exact check on the
no-conversion path. Only the denoise blur remains unported.

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
