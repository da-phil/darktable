/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "common/vulkan.h"

#ifdef HAVE_VULKAN

#include "common/darktable.h"
#include "common/dtpthread.h"
#include "common/file_location.h"
#include "control/conf.h"

#include <limits.h>  // PATH_MAX

#include <glib/gstdio.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// One global lock; mirrors the per-device mutex pattern in opencl.c but
// since our backend currently exposes a single physical device we keep
// it simple and lock the whole subsystem on the dispatch path.
static dt_pthread_mutex_t g_vk_lock = PTHREAD_MUTEX_INITIALIZER;

// ---- helpers ---------------------------------------------------------

static const char *_vkerr(VkResult r)
{
  switch(r)
  {
    case VK_SUCCESS:                       return "VK_SUCCESS";
    case VK_ERROR_OUT_OF_HOST_MEMORY:      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:    return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:   return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:       return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:   return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:     return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:     return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_DEVICE_LOST:             return "VK_ERROR_DEVICE_LOST";
    default: break;
  }
  return "VK_<unknown>";
}

#define VKCHECK(call) do {                                       \
    VkResult _r = (call);                                        \
    if(_r != VK_SUCCESS) {                                       \
      dt_print(DT_DEBUG_OPENCL,                                  \
               "[vulkan] %s failed: %s", #call, _vkerr(_r));     \
      goto error;                                                \
    }                                                            \
  } while(0)

// Compare a SPIR-V literal-string operand against a C string.
// SPIR-V strings are null-terminated bytes packed 4-per-word.
// max_words bounds the read so we never run off the end of the
// module if a string is malformed.
static gboolean _spirv_string_match(const uint32_t *words, size_t max_words, const char *needle)
{
  const char *bytes = (const char *)words;
  const size_t max_bytes = max_words * 4;
  for(size_t k = 0; k < max_bytes; k++)
  {
    const char c = bytes[k];
    if(c == 0)
      return needle[k] == 0;
    if(c != needle[k])
      return FALSE;
  }
  return FALSE;
}

// Scan a SPIR-V module for OpEntryPoint (opcode 15) and return TRUE
// iff one of them names `entry_name`. Used by dt_vulkan_create_kernel
// to refuse pipeline creation when the requested entry isn't in the
// module — Mesa RADV segfaults inside vkCreateComputePipelines in
// that case (observed on multi-entry modules built with glslang,
// which only emits one entry per .spv).
static gboolean _spirv_has_entry(const uint32_t *spirv, size_t words, const char *entry_name)
{
  if(!spirv || words < 5) return FALSE;
  size_t i = 5;  // SPIR-V module header is 5 words
  while(i < words)
  {
    const uint32_t inst = spirv[i];
    const uint16_t opcode = (uint16_t)(inst & 0xffff);
    const uint16_t wc     = (uint16_t)(inst >> 16);
    if(wc == 0 || i + wc > words) return FALSE;  // malformed
    if(opcode == 15 /* OpEntryPoint */ && wc >= 4)
    {
      // Layout: word i+0 = inst, i+1 = ExecutionModel, i+2 = entry-point %id,
      // i+3.. = LiteralString name (null-terminated), then Interface IDs.
      if(_spirv_string_match(&spirv[i + 3], wc - 3, entry_name))
        return TRUE;
    }
    i += wc;
  }
  return FALSE;
}

static uint32_t _find_memtype(const dt_vk_device_t *d,
                              uint32_t type_filter,
                              VkMemoryPropertyFlags props)
{
  for(uint32_t i = 0; i < d->mem_props.memoryTypeCount; ++i)
  {
    if((type_filter & (1u << i)) &&
       (d->mem_props.memoryTypes[i].propertyFlags & props) == props)
      return i;
  }
  return UINT32_MAX;
}

// ---- capture / deferred graph execution (DAG milestone M1) -----------
//
// dev-doc/gpu_resident_pixelpipe_dag.md §5.1-§5.3. The M1 IR is
// deliberately simpler than the design sketch: nodes reference *real*
// dt_vk_mem_t buffers (allocated from the pool exactly as in eager
// mode) instead of virtual resource ids, and correctness of buffer
// reuse is guaranteed by deferring dt_vulkan_free_buffer until the
// segment flush rather than by liveness analysis. The M3 planner
// swaps this for interval-aliased arena offsets without touching the
// node shape.

typedef enum _capture_node_kind_t
{
  _NODE_DISPATCH = 0,
  _NODE_UPLOAD,     // staging slice -> dst buffer (payload snapshotted or borrowed)
  _NODE_COPY,       // src buffer -> dst buffer, one region at offset 0
  _NODE_COPY_ROWS,  // src -> dst, one region per row (subregion copy)
} _capture_node_kind_t;

typedef struct _capture_node_t
{
  _capture_node_kind_t kind;

  // _NODE_DISPATCH
  int          kernel;                        // device kernel slot
  uint32_t     gx, gy, gz;                    // resolved workgroup counts
  uint32_t     nbufs;
  dt_vk_mem_t *bufs[DT_VULKAN_MAX_BINDINGS];
  uint32_t     push_size;
  uint8_t      push[DT_VULKAN_MAX_PUSH_CONSTANTS];

  // _NODE_UPLOAD / _NODE_COPY / _NODE_COPY_ROWS
  dt_vk_mem_t       *dst;
  const dt_vk_mem_t *src;         // COPY / COPY_ROWS
  size_t             size;        // UPLOAD / COPY byte count
  const void        *borrowed;    // UPLOAD: borrowed host pointer, or NULL
  size_t             ring_off;    // UPLOAD: offset into the snapshot ring
  VkDeviceSize       staging_off; // UPLOAD: staging offset, assigned at flush

  // _NODE_COPY_ROWS geometry (see dt_vulkan_copy_subregion)
  size_t sox, soy, dox, doy, rw, rh, srp, drp, bpp;
} _capture_node_t;

// A pending GPU tap (DAG M5, §5.4): a small device result buffer whose
// readback is deferred to the end of the run so the reduction doesn't
// split the span. `buf` is owned by the registry (freed at drain).
typedef struct _capture_tap_t
{
  dt_vk_mem_t *buf;   // device result, written by a captured dispatch
  void        *dst;   // host destination, filled at drain
  size_t       size;  // bytes
} _capture_tap_t;

typedef struct dt_vk_capture_ctx_t
{
  gboolean   active;
  int        devid;
  GArray    *nodes;      // of _capture_node_t
  uint8_t   *ring;       // snapshot payloads (uploads whose host ptr dies)
  size_t     ring_used;
  size_t     ring_cap;
  GPtrArray *dfree;      // dt_vk_mem_t* whose free is deferred to flush
  GArray    *taps;       // of _capture_tap_t, drained at capture_end
  uint32_t   flushes;    // segments submitted since begin (diagnostics)
} dt_vk_capture_ctx_t;

// One context per thread: parallel pipelines (full / preview / export)
// capture independently and only serialise on the device lock at
// flush time. __thread has precedent in src/control/jobs.c and is
// checked by the build system (src/CMakeLists.txt tests it).
static __thread dt_vk_capture_ctx_t *g_capture = NULL;

// Active context of the calling thread, or NULL in eager mode.
static inline dt_vk_capture_ctx_t *_cap(void)
{
  return (g_capture && g_capture->active) ? g_capture : NULL;
}

static _capture_node_t *_capture_append(dt_vk_capture_ctx_t *c)
{
  _capture_node_t z;
  memset(&z, 0, sizeof(z));
  g_array_append_val(c->nodes, z);
  return &g_array_index(c->nodes, _capture_node_t, c->nodes->len - 1);
}

// Copy an upload payload into the snapshot ring (4-byte aligned so the
// flush can reuse the offsets as vkCmdCopyBuffer staging offsets).
static int _capture_ring_push(dt_vk_capture_ctx_t *c,
                              const void *data, size_t size, size_t *off)
{
  const size_t base = (c->ring_used + 3u) & ~(size_t)3u;
  if(base + size > c->ring_cap)
  {
    size_t cap = c->ring_cap ? c->ring_cap : (size_t)1 << 20;
    while(cap < base + size) cap *= 2;
    uint8_t *ring = realloc(c->ring, cap);
    if(!ring) return -1;
    c->ring = ring;
    c->ring_cap = cap;
  }
  memcpy(c->ring + base, data, size);
  *off = base;
  c->ring_used = base + size;
  return 0;
}

// Wind the context back to a checkpoint (failed append inside one HAL
// call, or module-granular rollback via dt_vulkan_capture_rollback).
static void _capture_truncate(dt_vk_capture_ctx_t *c,
                              uint32_t nodes, size_t ring_used)
{
  if(c->nodes->len > nodes) g_array_set_size(c->nodes, nodes);
  if(c->ring_used > ring_used) c->ring_used = ring_used;
}

// ---- init / cleanup --------------------------------------------------

static gboolean _create_device(dt_vk_device_t *d, VkPhysicalDevice phys)
{
  d->phys = phys;
  VkPhysicalDeviceProperties pp;
  vkGetPhysicalDeviceProperties(phys, &pp);
  snprintf(d->name, sizeof(d->name), "%s", pp.deviceName);

  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, NULL);
  if(qn == 0) return FALSE;
  VkQueueFamilyProperties *qprops = calloc(qn, sizeof(*qprops));
  vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qprops);

  d->queue_family_index = UINT32_MAX;
  for(uint32_t i = 0; i < qn; ++i)
  {
    if(qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
    {
      d->queue_family_index = i;
      break;
    }
  }
  free(qprops);
  if(d->queue_family_index == UINT32_MAX) return FALSE;

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = d->queue_family_index,
                                  .queueCount = 1, .pQueuePriorities = &prio };
  VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci };
  VKCHECK(vkCreateDevice(phys, &dci, NULL, &d->device));

  vkGetDeviceQueue(d->device, d->queue_family_index, 0, &d->queue);
  vkGetPhysicalDeviceMemoryProperties(phys, &d->mem_props);

  VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .queueFamilyIndex = d->queue_family_index,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
  VKCHECK(vkCreateCommandPool(d->device, &cpci, NULL, &d->cmd_pool));

  // Pre-size a descriptor pool big enough for all kernels we register
  // up front. This grows lazily if exhausted in future revisions.
  VkDescriptorPoolSize psz = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               .descriptorCount = DT_VULKAN_MAX_BINDINGS * DT_VULKAN_MAX_KERNELS };
  VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                      .maxSets = DT_VULKAN_MAX_KERNELS,
                                      .poolSizeCount = 1, .pPoolSizes = &psz };
  VKCHECK(vkCreateDescriptorPool(d->device, &dpci, NULL, &d->dset_pool));
  return TRUE;

error:
  if(d->cmd_pool) vkDestroyCommandPool(d->device, d->cmd_pool, NULL);
  if(d->device)   vkDestroyDevice(d->device, NULL);
  memset(d, 0, sizeof(*d));
  return FALSE;
}

void dt_vulkan_init(dt_vulkan_t *vk)
{
  if(!vk) return;
  memset(vk, 0, sizeof(*vk));

  // Honour the runtime preference: even if compiled in, allow opt-out.
  vk->enabled = dt_conf_get_bool("opencl_use_vulkan");

  if(!vk->enabled)
  {
    dt_print(DT_DEBUG_OPENCL, "[vulkan] disabled via preferences");
    return;
  }

  VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "darktable",
                            .applicationVersion = VK_MAKE_VERSION(5, 0, 0),
                            .pEngineName = "darktable",
                            .engineVersion = VK_MAKE_VERSION(5, 0, 0),
                            .apiVersion = VK_API_VERSION_1_2 };
  VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app };
  VKCHECK(vkCreateInstance(&ici, NULL, &vk->instance));

  uint32_t n = 0;
  VKCHECK(vkEnumeratePhysicalDevices(vk->instance, &n, NULL));
  if(n == 0)
  {
    dt_print(DT_DEBUG_OPENCL, "[vulkan] no physical devices");
    goto error;
  }
  VkPhysicalDevice *phys = calloc(n, sizeof(*phys));
  VKCHECK(vkEnumeratePhysicalDevices(vk->instance, &n, phys));

  vk->dev = calloc(n, sizeof(*vk->dev));
  for(uint32_t i = 0; i < n; ++i)
  {
    if(_create_device(&vk->dev[vk->num_devs], phys[i]))
    {
      dt_print(DT_DEBUG_OPENCL, "[vulkan] device %d: %s",
               vk->num_devs, vk->dev[vk->num_devs].name);
      vk->num_devs++;
    }
  }
  free(phys);

  if(vk->num_devs == 0)
  {
    dt_print(DT_DEBUG_OPENCL, "[vulkan] no compute-capable devices");
    goto error;
  }

  vk->inited = TRUE;
  return;

error:
  dt_vulkan_cleanup(vk);
}

void dt_vulkan_cleanup(dt_vulkan_t *vk)
{
  if(!vk) return;
  for(int i = 0; i < vk->num_devs; ++i)
  {
    dt_vk_device_t *d = &vk->dev[i];
    if(d->device) vkDeviceWaitIdle(d->device);
    for(int k = 0; k < DT_VULKAN_MAX_KERNELS; ++k) dt_vulkan_free_kernel(k);
    for(int p = 0; p < DT_VULKAN_MAX_PROGRAMS; ++p)
    {
      if(d->programs[p].used) free(d->programs[p].spirv);
    }
    if(d->staging)
    {
      if(d->staging->buffer) vkDestroyBuffer(d->device, d->staging->buffer, NULL);
      if(d->staging->memory) vkFreeMemory(d->device, d->staging->memory, NULL);
      free(d->staging);
      d->staging = NULL;
    }
    for(int b = 0; b < d->buf_pool_count; b++)
    {
      dt_vk_mem_t *m = d->buf_pool[b];
      if(m->buffer) vkDestroyBuffer(d->device, m->buffer, NULL);
      if(m->memory) vkFreeMemory(d->device, m->memory, NULL);
      free(m);
    }
    d->buf_pool_count = 0;
    if(d->oneshot_fence) vkDestroyFence(d->device, d->oneshot_fence, NULL);
    if(d->oneshot_cmd)
      vkFreeCommandBuffers(d->device, d->cmd_pool, 1, &d->oneshot_cmd);
    if(d->dset_pool) vkDestroyDescriptorPool(d->device, d->dset_pool, NULL);
    if(d->cmd_pool)  vkDestroyCommandPool(d->device, d->cmd_pool, NULL);
    if(d->device)    vkDestroyDevice(d->device, NULL);
  }
  free(vk->dev);
  if(vk->instance) vkDestroyInstance(vk->instance, NULL);
  memset(vk, 0, sizeof(*vk));
}

gboolean dt_vulkan_running(void)
{
  if(!darktable.vulkan) return FALSE;
  return darktable.vulkan->inited && darktable.vulkan->enabled
         && darktable.vulkan->num_devs > 0;
}

int dt_vulkan_lock_device(void)
{
  if(!dt_vulkan_running()) return -1;
  dt_pthread_mutex_lock(&g_vk_lock);
  return 0; // single-device for now
}

void dt_vulkan_unlock_device(int devid)
{
  (void)devid;
  dt_pthread_mutex_unlock(&g_vk_lock);
}

// ---- programs --------------------------------------------------------

static uint32_t *_load_spv(const char *path, size_t *out_words)
{
  // g_fopen rather than fopen for Windows wide-path correctness; same
  // pattern as src/common/opencl.c.
  FILE *f = g_fopen(path, "rb");
  if(!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if(sz <= 0 || (sz % 4) != 0) { fclose(f); return NULL; }
  uint32_t *buf = malloc((size_t)sz);
  if(!buf) { fclose(f); return NULL; }
  size_t r = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  if(r != (size_t)sz) { free(buf); return NULL; }
  *out_words = (size_t)sz / 4;
  return buf;
}

int dt_vulkan_load_program_by_name(const char *name)
{
  if(!dt_vulkan_running()) return -1;
  char path[PATH_MAX] = { 0 };
  dt_loc_get_datadir(path, sizeof(path));
  g_strlcat(path, "/kernels/vulkan/", sizeof(path));
  g_strlcat(path, name, sizeof(path));
  g_strlcat(path, ".spv", sizeof(path));
  return dt_vulkan_load_program(name, path);
}

int dt_vulkan_load_program(const char *name, const char *path)
{
  if(!dt_vulkan_running()) return -1;
  dt_vk_device_t *d = &darktable.vulkan->dev[0];

  // Find a free slot.
  int slot = -1;
  for(int i = 0; i < DT_VULKAN_MAX_PROGRAMS; ++i)
    if(!d->programs[i].used) { slot = i; break; }
  if(slot < 0)
  {
    // Slot starvation: every subsequent module/helper will silently
    // lose its Vulkan path and fall back to CPU. This must never
    // happen in a release build — raise DT_VULKAN_MAX_PROGRAMS.
    dt_print(DT_DEBUG_ALWAYS,
             "[vulkan] FATAL: out of program slots (max %d) loading '%s' — "
             "raise DT_VULKAN_MAX_PROGRAMS; all later modules will fall back to CPU",
             DT_VULKAN_MAX_PROGRAMS, name ? name : "<?>");
    return -1;
  }

  size_t words = 0;
  uint32_t *spv = _load_spv(path, &words);
  if(!spv)
  {
    dt_print(DT_DEBUG_OPENCL, "[vulkan] cannot load SPIR-V from %s", path);
    return -1;
  }

  d->programs[slot].used        = TRUE;
  d->programs[slot].spirv       = spv;
  d->programs[slot].spirv_words = words;
  snprintf(d->programs[slot].name, sizeof(d->programs[slot].name), "%s", name);
  return slot;
}

// ---- kernels ---------------------------------------------------------

int dt_vulkan_create_kernel(int program,
                            const char *entry,
                            uint32_t num_storage_buffer_bindings,
                            uint32_t push_constant_size,
                            uint32_t local_size_x,
                            uint32_t local_size_y,
                            uint32_t local_size_z)
{
  if(!dt_vulkan_running()) return -1;
  if(num_storage_buffer_bindings > DT_VULKAN_MAX_BINDINGS) return -1;
  if(push_constant_size > DT_VULKAN_MAX_PUSH_CONSTANTS)    return -1;
  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  if(program < 0 || program >= DT_VULKAN_MAX_PROGRAMS || !d->programs[program].used)
    return -1;

  // Multi-entry modules (channelmixerrgb, borders) load several entry
  // names from the same program. clspv emits all of them; glslang's
  // -e flag emits only one. Mesa RADV crashes inside
  // vkCreateComputePipelines if asked for a missing entry, so scan
  // the SPIR-V header here and bail cleanly before touching Vulkan.
  if(!_spirv_has_entry(d->programs[program].spirv,
                       d->programs[program].spirv_words, entry))
  {
    // Expected on glslang-fallback builds — the multi-entry modules
    // (channelmixerrgb, borders) probe each entry; the ones absent
    // here just stay on OpenCL. Logged once per probe so a developer
    // chasing real failures has a breadcrumb, but the wording is
    // intentionally informational rather than alarmist.
    dt_print(DT_DEBUG_OPENCL,
             "[vulkan] entry '%s' not in program %d (likely glslang fallback build; "
             "module will use OpenCL for this entry)", entry, program);
    return -1;
  }

  int slot = -1;
  for(int i = 0; i < DT_VULKAN_MAX_KERNELS; ++i)
    if(!d->kernels[i].used) { slot = i; break; }
  if(slot < 0)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[vulkan] FATAL: out of kernel slots (max %d) creating '%s' — "
             "raise DT_VULKAN_MAX_KERNELS; this module will fall back to CPU",
             DT_VULKAN_MAX_KERNELS, entry ? entry : "<?>");
    return -1;
  }
  dt_vk_kernel_t *k = &d->kernels[slot];
  memset(k, 0, sizeof(*k));

  VkShaderModuleCreateInfo smci = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = d->programs[program].spirv_words * sizeof(uint32_t),
                                    .pCode = d->programs[program].spirv };
  VKCHECK(vkCreateShaderModule(d->device, &smci, NULL, &k->shader_module));

  VkDescriptorSetLayoutBinding bindings[DT_VULKAN_MAX_BINDINGS];
  for(uint32_t i = 0; i < num_storage_buffer_bindings; ++i)
  {
    bindings[i] = (VkDescriptorSetLayoutBinding){ .binding = i,
                                                  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                  .descriptorCount = 1,
                                                  .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
  }
  VkDescriptorSetLayoutCreateInfo dslci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = num_storage_buffer_bindings,
                                            .pBindings = bindings };
  VKCHECK(vkCreateDescriptorSetLayout(d->device, &dslci, NULL, &k->dset_layout));

  VkPushConstantRange pc = { .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                             .offset = 0, .size = push_constant_size };
  VkPipelineLayoutCreateInfo plci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                      .setLayoutCount = 1, .pSetLayouts = &k->dset_layout,
                                      .pushConstantRangeCount = push_constant_size ? 1 : 0,
                                      .pPushConstantRanges    = push_constant_size ? &pc : NULL };
  VKCHECK(vkCreatePipelineLayout(d->device, &plci, NULL, &k->pipeline_layout));

  VkPipelineShaderStageCreateInfo ssci = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                           .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                                           .module = k->shader_module,
                                           .pName = entry };
  VkComputePipelineCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                       .stage = ssci, .layout = k->pipeline_layout };
  VKCHECK(vkCreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cpci, NULL, &k->pipeline));

  k->used = TRUE;
  k->program = program;
  snprintf(k->name, sizeof(k->name), "%s", entry);
  k->num_storage_buffer_bindings = num_storage_buffer_bindings;
  k->push_constant_size = push_constant_size;
  k->local_size_x = local_size_x;
  k->local_size_y = local_size_y;
  k->local_size_z = local_size_z;
  return slot;

error:
  if(k->pipeline)        vkDestroyPipeline(d->device, k->pipeline, NULL);
  if(k->pipeline_layout) vkDestroyPipelineLayout(d->device, k->pipeline_layout, NULL);
  if(k->dset_layout)     vkDestroyDescriptorSetLayout(d->device, k->dset_layout, NULL);
  if(k->shader_module)   vkDestroyShaderModule(d->device, k->shader_module, NULL);
  memset(k, 0, sizeof(*k));
  return -1;
}

void dt_vulkan_free_kernel(int kernel)
{
  if(!dt_vulkan_running()) return;
  if(kernel < 0 || kernel >= DT_VULKAN_MAX_KERNELS) return;
  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  dt_vk_kernel_t *k = &d->kernels[kernel];
  if(!k->used) return;
  if(k->pipeline)        vkDestroyPipeline(d->device, k->pipeline, NULL);
  if(k->pipeline_layout) vkDestroyPipelineLayout(d->device, k->pipeline_layout, NULL);
  if(k->dset_layout)     vkDestroyDescriptorSetLayout(d->device, k->dset_layout, NULL);
  if(k->shader_module)   vkDestroyShaderModule(d->device, k->shader_module, NULL);
  memset(k, 0, sizeof(*k));
}

// ---- memory ----------------------------------------------------------

static dt_vk_mem_t *_alloc(dt_vk_device_t *d, VkDeviceSize size,
                           VkBufferUsageFlags usage, bool host_visible)
{
  dt_vk_mem_t *m = calloc(1, sizeof(*m));
  if(!m) return NULL;
  m->size = size;
  m->host_visible = host_visible;

  VkBufferCreateInfo bci = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE };
  VKCHECK(vkCreateBuffer(d->device, &bci, NULL, &m->buffer));

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(d->device, m->buffer, &req);

  VkMemoryPropertyFlags props = host_visible
      ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
      : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  uint32_t mt = _find_memtype(d, req.memoryTypeBits, props);
  if(mt == UINT32_MAX) goto error;

  VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = req.size, .memoryTypeIndex = mt };
  VKCHECK(vkAllocateMemory(d->device, &ai, NULL, &m->memory));
  VKCHECK(vkBindBufferMemory(d->device, m->buffer, m->memory, 0));
  return m;

error:
  if(m->buffer) vkDestroyBuffer(d->device, m->buffer, NULL);
  if(m->memory) vkFreeMemory(d->device, m->memory, NULL);
  free(m);
  return NULL;
}

// Device-buffer pool. The HAL's `_alloc` path is dominated by
// `vkAllocateMemory` (5-30 ms per large buffer on RADV / lavapipe).
// Each VK module dispatch allocates two buffers (vin/vout); without
// a pool that's 10-60 ms of pure driver overhead per module before
// any work happens. The pool serves alloc requests from a free-list
// of previously-released buffers; only when no buffer ≥ requested
// size is available do we fall back to a fresh _alloc. Buffers
// remain device-local + storage + transfer-src/dst, matching the
// fresh-allocation flags, so any caller can use a pooled buffer
// interchangeably.
static dt_vk_mem_t *_pool_take(dt_vk_device_t *d, VkDeviceSize size)
{
  // Best fit: pick the smallest pooled buffer that still satisfies
  // `size`. Avoids handing out a 1 GB staging-leftover for a 4 MB
  // dispatch.
  int best = -1;
  VkDeviceSize best_size = (VkDeviceSize)-1;
  for(int i = 0; i < d->buf_pool_count; i++)
  {
    const VkDeviceSize s = d->buf_pool[i]->size;
    if(s >= size && s < best_size)
    {
      best = i;
      best_size = s;
    }
  }
  if(best < 0) return NULL;
  dt_vk_mem_t *m = d->buf_pool[best];
  // O(1) removal — order in the pool isn't meaningful.
  d->buf_pool[best] = d->buf_pool[--d->buf_pool_count];
  return m;
}

static gboolean _pool_put(dt_vk_device_t *d, dt_vk_mem_t *m)
{
  // Don't pool host-visible buffers — the staging buffer is the only
  // host-visible allocation we make and it has its own caching path.
  if(m->host_visible) return FALSE;
  if(d->buf_pool_count >= DT_VULKAN_BUF_POOL_CAP)
  {
    // Pool full: evict the smallest buffer (most likely to be
    // re-allocated cheaply if its size class comes up again) and
    // make room for this one.
    int victim = 0;
    for(int i = 1; i < d->buf_pool_count; i++)
      if(d->buf_pool[i]->size < d->buf_pool[victim]->size) victim = i;
    dt_vk_mem_t *ev = d->buf_pool[victim];
    if(ev->buffer) vkDestroyBuffer(d->device, ev->buffer, NULL);
    if(ev->memory) vkFreeMemory(d->device, ev->memory, NULL);
    free(ev);
    d->buf_pool[victim] = d->buf_pool[--d->buf_pool_count];
  }
  d->buf_pool[d->buf_pool_count++] = m;
  return TRUE;
}

dt_vk_mem_t *dt_vulkan_alloc_buffer(int devid, size_t size)
{
  if(!dt_vulkan_running()) return NULL;
  (void)devid;
  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  dt_vk_mem_t *pooled = _pool_take(d, (VkDeviceSize)size);
  if(pooled) return pooled;
  return _alloc(d, (VkDeviceSize)size,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
              | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                false);
}

void dt_vulkan_free_buffer(int devid, dt_vk_mem_t *mem)
{
  if(!mem || !dt_vulkan_running()) return;
  (void)devid;

  // Under capture, pending nodes may still reference this buffer —
  // the GPU work hasn't run yet. Defer the release to the segment
  // flush (or rollback), which happens after the fence wait.
  dt_vk_capture_ctx_t *cap = _cap();
  if(cap)
  {
    g_ptr_array_add(cap->dfree, mem);
    return;
  }

  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  if(_pool_put(d, mem)) return;
  if(mem->buffer) vkDestroyBuffer(d->device, mem->buffer, NULL);
  if(mem->memory) vkFreeMemory(d->device, mem->memory, NULL);
  free(mem);
}

// Really release a buffer: pool if possible, destroy otherwise.
// The no-capture branch of dt_vulkan_free_buffer, callable from the
// flush/rollback paths that drain the deferred-free list.
static void _release_buffer(dt_vk_device_t *d, dt_vk_mem_t *mem)
{
  if(!mem) return;
  if(_pool_put(d, mem)) return;
  if(mem->buffer) vkDestroyBuffer(d->device, mem->buffer, NULL);
  if(mem->memory) vkFreeMemory(d->device, mem->memory, NULL);
  free(mem);
}

// Drain the M5 tap registry down to `keep` entries. `do_read` reads
// each tap's device buffer into its host destination (used at
// capture_end, once the final fence has retired the producing
// dispatches); otherwise the results are abandoned (abort / rollback)
// and the buffers are just released. Caller holds the device lock and
// must have already left capture (active == FALSE) when do_read is
// TRUE, so the readback runs eagerly instead of re-flushing.
static void _drain_taps(dt_vk_capture_ctx_t *cap, dt_vk_device_t *d,
                        int devid, guint keep, gboolean do_read)
{
  for(guint i = keep; i < cap->taps->len; i++)
  {
    _capture_tap_t *t = &g_array_index(cap->taps, _capture_tap_t, i);
    if(do_read && t->dst && t->buf)
      dt_vulkan_read_from_device(devid, t->dst, t->buf, t->size);
    if(t->buf) _release_buffer(d, t->buf);
  }
  g_array_set_size(cap->taps, keep);
}

gboolean dt_vulkan_tap_register(int devid, dt_vk_mem_t *buf,
                                void *host_dst, size_t size)
{
  (void)devid;
  dt_vk_capture_ctx_t *cap = _cap();
  if(!cap || !buf) return FALSE;
  const _capture_tap_t t = { buf, host_dst, size };
  g_array_append_val(cap->taps, t);
  return TRUE;
}

// Generic one-shot command-buffer helper: record fn into the
// persistent command buffer, submit on the queue, wait on the
// persistent fence, return. Reuses d->oneshot_cmd and
// d->oneshot_fence rather than creating/destroying them per call —
// each module dispatch invokes this 3+ times (upload, kernel,
// readback) and the create/destroy pair was costing 5-20 µs per
// call on RADV. Callers must hold g_vk_lock; the persistent
// resources are device-wide.
typedef int (*_record_cb)(VkCommandBuffer, void *);

static int _submit_one_shot(dt_vk_device_t *d, _record_cb fn, void *user)
{
  // Lazily create the persistent cmd-buffer + fence on first use.
  // The command pool is created with RESET_COMMAND_BUFFER_BIT so
  // vkResetCommandBuffer is legal.
  if(d->oneshot_cmd == VK_NULL_HANDLE)
  {
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = d->cmd_pool,
                                         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1 };
    if(vkAllocateCommandBuffers(d->device, &cbai, &d->oneshot_cmd) != VK_SUCCESS)
      return -1;
  }
  if(d->oneshot_fence == VK_NULL_HANDLE)
  {
    VkFenceCreateInfo fci = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if(vkCreateFence(d->device, &fci, NULL, &d->oneshot_fence) != VK_SUCCESS)
      return -1;
  }

  VkCommandBuffer cmd = d->oneshot_cmd;
  int rc = -1;

  // Reset rather than recreate. The previous fence wait guarantees
  // the GPU is no longer using the cmd buffer (this helper is
  // called synchronously, lock-held).
  VKCHECK(vkResetCommandBuffer(cmd, 0));

  VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                  .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
  VKCHECK(vkBeginCommandBuffer(cmd, &bi));
  if(fn(cmd, user) != 0) goto error;
  VKCHECK(vkEndCommandBuffer(cmd));

  VKCHECK(vkResetFences(d->device, 1, &d->oneshot_fence));
  VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 1, .pCommandBuffers = &cmd };
  VKCHECK(vkQueueSubmit(d->queue, 1, &si, d->oneshot_fence));
  d->submit_count++;
  VKCHECK(vkWaitForFences(d->device, 1, &d->oneshot_fence, VK_TRUE, UINT64_MAX));
  rc = 0;

error:
  return rc;
}

typedef struct { VkBuffer src, dst; VkDeviceSize sz; } _copy_args_t;

static int _record_copy(VkCommandBuffer cmd, void *u)
{
  _copy_args_t *a = u;
  VkBufferCopy r = { 0, 0, a->sz };
  vkCmdCopyBuffer(cmd, a->src, a->dst, 1, &r);
  return 0;
}

typedef struct
{
  VkBuffer        src, dst;
  const VkBufferCopy *regions;
  uint32_t        n_regions;
} _copy_multi_args_t;

static int _record_copy_multi(VkCommandBuffer cmd, void *u)
{
  _copy_multi_args_t *a = u;
  vkCmdCopyBuffer(cmd, a->src, a->dst, a->n_regions, a->regions);
  return 0;
}

// Return d->staging sized at least `size`. Reallocates if the
// cached buffer is smaller. Always host-visible + both transfer
// directions so the same buffer serves uploads and downloads.
// Callers must hold the global VK lock (g_vk_lock).
static dt_vk_mem_t *_ensure_staging(dt_vk_device_t *d, size_t size)
{
  if(d->staging && d->staging->size >= (VkDeviceSize)size)
    return d->staging;
  if(d->staging)
  {
    if(d->staging->buffer) vkDestroyBuffer(d->device, d->staging->buffer, NULL);
    if(d->staging->memory) vkFreeMemory(d->device, d->staging->memory, NULL);
    free(d->staging);
    d->staging = NULL;
  }
  d->staging = _alloc(d, (VkDeviceSize)size,
                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
  return d->staging;
}

// Shared body for the snapshot / borrowed upload flavours.
static int _write_to_device(int devid, dt_vk_mem_t *dst,
                            const void *host, size_t size,
                            gboolean borrow)
{
  if(!dt_vulkan_running() || !dst || !host) return -1;
  (void)devid;

  dt_vk_capture_ctx_t *cap = _cap();
  if(cap)
  {
    size_t ring_off = 0;
    if(!borrow && _capture_ring_push(cap, host, size, &ring_off) != 0)
      return -1;
    _capture_node_t *n = _capture_append(cap);
    n->kind = _NODE_UPLOAD;
    n->dst = dst;
    n->size = size;
    n->borrowed = borrow ? host : NULL;
    n->ring_off = ring_off;
    return 0;
  }

  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  dt_vk_mem_t *staging = _ensure_staging(d, size);
  if(!staging) return -1;

  void *p = NULL;
  if(vkMapMemory(d->device, staging->memory, 0, size, 0, &p) != VK_SUCCESS)
    return -1;
  memcpy(p, host, size);
  vkUnmapMemory(d->device, staging->memory);

  _copy_args_t a = { staging->buffer, dst->buffer, (VkDeviceSize)size };
  return _submit_one_shot(d, _record_copy, &a);
}

int dt_vulkan_write_to_device(int devid, dt_vk_mem_t *dst,
                              const void *host, size_t size)
{
  return _write_to_device(devid, dst, host, size, FALSE);
}

int dt_vulkan_write_to_device_borrowed(int devid, dt_vk_mem_t *dst,
                                       const void *host, size_t size)
{
  return _write_to_device(devid, dst, host, size, TRUE);
}

int dt_vulkan_read_from_device(int devid, void *host,
                               const dt_vk_mem_t *src, size_t size)
{
  if(!dt_vulkan_running() || !src) return -1;
  (void)devid;

  // Sync tap (dev-doc/gpu_resident_pixelpipe_dag.md §5.3): a read of a
  // buffer whose producing nodes haven't executed can't be served
  // lazily. Flush the pending segment (submit + fence), then serve
  // the read eagerly. Capture stays active for the next segment.
  dt_vk_capture_ctx_t *cap = _cap();
  if(cap)
  {
    dt_print(DT_DEBUG_VKGRAPH,
             "[vkgraph] sync tap: flushing %u node(s) to serve a %zu byte readback",
             cap->nodes->len, size);
    if(dt_vulkan_capture_flush(cap->devid) != 0) return -1;
  }

  dt_vk_device_t *d = &darktable.vulkan->dev[0];

  dt_vk_mem_t *staging = _ensure_staging(d, size);
  if(!staging) return -1;

  _copy_args_t a = { src->buffer, staging->buffer, (VkDeviceSize)size };
  int rc = _submit_one_shot(d, _record_copy, &a);
  if(rc != 0) return rc;

  void *p = NULL;
  if(vkMapMemory(d->device, staging->memory, 0, size, 0, &p) != VK_SUCCESS)
    return -1;
  memcpy(host, p, size);
  vkUnmapMemory(d->device, staging->memory);
  return 0;
}

int dt_vulkan_copy_device_to_device(int devid, dt_vk_mem_t *dst,
                                    const dt_vk_mem_t *src, size_t size)
{
  if(!dt_vulkan_running() || !dst || !src) return -1;
  (void)devid;

  dt_vk_capture_ctx_t *cap = _cap();
  if(cap)
  {
    _capture_node_t *n = _capture_append(cap);
    n->kind = _NODE_COPY;
    n->dst = dst;
    n->src = src;
    n->size = size;
    return 0;
  }

  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  _copy_args_t a = { src->buffer, dst->buffer, (VkDeviceSize)size };
  return _submit_one_shot(d, _record_copy, &a);
}

int dt_vulkan_copy_subregion(int devid,
                             dt_vk_mem_t *dst,
                             const dt_vk_mem_t *src,
                             size_t src_offset_x, size_t src_offset_y,
                             size_t dst_offset_x, size_t dst_offset_y,
                             size_t region_w, size_t region_h,
                             size_t src_row_pixels, size_t dst_row_pixels,
                             size_t bytes_per_pixel)
{
  if(!dt_vulkan_running() || !dst || !src) return -1;
  if(region_w == 0 || region_h == 0) return 0;
  (void)devid;

  dt_vk_capture_ctx_t *cap = _cap();
  if(cap)
  {
    _capture_node_t *n = _capture_append(cap);
    n->kind = _NODE_COPY_ROWS;
    n->dst = dst;
    n->src = src;
    n->sox = src_offset_x; n->soy = src_offset_y;
    n->dox = dst_offset_x; n->doy = dst_offset_y;
    n->rw = region_w; n->rh = region_h;
    n->srp = src_row_pixels; n->drp = dst_row_pixels;
    n->bpp = bytes_per_pixel;
    return 0;
  }

  dt_vk_device_t *d = &darktable.vulkan->dev[0];

  // One VkBufferCopy region per row. For small region_h this is a
  // single command; for large region_h the driver-side cost is
  // proportional to row count but the per-row copy itself runs at
  // device-memory bandwidth.
  VkBufferCopy *regions = malloc(sizeof(VkBufferCopy) * region_h);
  if(!regions) return -1;
  const VkDeviceSize row_bytes = (VkDeviceSize)region_w * bytes_per_pixel;
  const VkDeviceSize src_stride = (VkDeviceSize)src_row_pixels * bytes_per_pixel;
  const VkDeviceSize dst_stride = (VkDeviceSize)dst_row_pixels * bytes_per_pixel;
  for(size_t r = 0; r < region_h; r++)
  {
    regions[r].srcOffset = (VkDeviceSize)(src_offset_y + r) * src_stride
                         + (VkDeviceSize)src_offset_x * bytes_per_pixel;
    regions[r].dstOffset = (VkDeviceSize)(dst_offset_y + r) * dst_stride
                         + (VkDeviceSize)dst_offset_x * bytes_per_pixel;
    regions[r].size = row_bytes;
  }
  _copy_multi_args_t a = {
    .src = src->buffer, .dst = dst->buffer,
    .regions = regions, .n_regions = (uint32_t)region_h
  };
  const int rc = _submit_one_shot(d, _record_copy_multi, &a);
  free(regions);
  return rc;
}

// ---- dispatch --------------------------------------------------------

typedef struct
{
  dt_vk_kernel_t *k;
  VkDescriptorSet dset;
  uint32_t        gx, gy, gz;
  const void     *push;
  size_t          push_size;
} _dispatch_args_t;

static int _record_dispatch(VkCommandBuffer cmd, void *u)
{
  _dispatch_args_t *a = u;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, a->k->pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          a->k->pipeline_layout, 0, 1, &a->dset, 0, NULL);
  if(a->push_size && a->push)
  {
    vkCmdPushConstants(cmd, a->k->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, (uint32_t)a->push_size, a->push);
  }
  vkCmdDispatch(cmd, a->gx, a->gy, a->gz);
  return 0;
}

// Batched: upload copies + memory barrier + compute dispatch, all
// recorded into one command buffer. Each upload reads from a disjoint
// region of the shared staging buffer (4-byte aligned offsets), so
// no extra staging allocations are needed.
typedef struct
{
  // dispatch fields (same as _dispatch_args_t)
  dt_vk_kernel_t *k;
  VkDescriptorSet dset;
  uint32_t        gx, gy, gz;
  const void     *push;
  size_t          push_size;
  // batched-upload fields
  VkBuffer        staging;
  size_t          upload_count;
  const dt_vk_upload_t *uploads;
  const VkDeviceSize   *offsets;  // staging offset per upload, 4-byte aligned
} _batched_args_t;

static int _record_batched(VkCommandBuffer cmd, void *u)
{
  const _batched_args_t *a = u;

  for(size_t i = 0; i < a->upload_count; i++)
  {
    const VkBufferCopy r = { a->offsets[i], 0, (VkDeviceSize)a->uploads[i].size };
    vkCmdCopyBuffer(cmd, a->staging, a->uploads[i].dst->buffer, 1, &r);
  }

  if(a->upload_count > 0)
  {
    // Make the staged writes visible to the compute shader.
    const VkMemoryBarrier mb = { .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                 .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                 .dstAccessMask = VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(cmd,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, NULL, 0, NULL);
  }

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, a->k->pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          a->k->pipeline_layout, 0, 1, &a->dset, 0, NULL);
  if(a->push_size && a->push)
  {
    vkCmdPushConstants(cmd, a->k->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, (uint32_t)a->push_size, a->push);
  }
  vkCmdDispatch(cmd, a->gx, a->gy, a->gz);
  return 0;
}

// ---- capture executor (plan / record / submit one segment) -----------

typedef struct _graph_record_args_t
{
  dt_vk_capture_ctx_t *cap;
  dt_vk_device_t      *d;
  VkBuffer             staging;  // VK_NULL_HANDLE when no uploads
  VkDescriptorSet     *dsets;    // parallel to nodes; only DISPATCH slots set
} _graph_record_args_t;

// M1 barrier policy: correctness over overlap. A full memory barrier
// between every pair of adjacent nodes, staged by the node kinds —
// except between two uploads with distinct destinations, which cannot
// hazard each other (their common staging source is read-only here,
// and ordering against *earlier* nodes flows transitively through the
// stage-scoped barriers already emitted). The M3 planner narrows this
// to per-resource dependencies once liveness intervals exist.
static gboolean _graph_needs_barrier(const _capture_node_t *prev,
                                     const _capture_node_t *cur)
{
  if(prev->kind == _NODE_UPLOAD && cur->kind == _NODE_UPLOAD
     && prev->dst != cur->dst)
    return FALSE;
  return TRUE;
}

static int _record_graph(VkCommandBuffer cmd, void *user)
{
  const _graph_record_args_t *a = user;
  const dt_vk_capture_ctx_t *cap = a->cap;

  for(guint i = 0; i < cap->nodes->len; i++)
  {
    const _capture_node_t *n = &g_array_index(cap->nodes, _capture_node_t, i);

    if(i > 0)
    {
      const _capture_node_t *p = &g_array_index(cap->nodes, _capture_node_t, i - 1);
      if(_graph_needs_barrier(p, n))
      {
        const gboolean src_compute = (p->kind == _NODE_DISPATCH);
        const gboolean dst_compute = (n->kind == _NODE_DISPATCH);
        const VkMemoryBarrier mb = {
          .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
          .srcAccessMask = src_compute ? VK_ACCESS_SHADER_WRITE_BIT
                                       : VK_ACCESS_TRANSFER_WRITE_BIT,
          .dstAccessMask = dst_compute ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
                                       : (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
        };
        vkCmdPipelineBarrier(cmd,
                             src_compute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                         : VK_PIPELINE_STAGE_TRANSFER_BIT,
                             dst_compute ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                         : VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &mb, 0, NULL, 0, NULL);
      }
    }

    switch(n->kind)
    {
      case _NODE_UPLOAD:
      {
        const VkBufferCopy r = { n->staging_off, 0, (VkDeviceSize)n->size };
        vkCmdCopyBuffer(cmd, a->staging, n->dst->buffer, 1, &r);
        break;
      }
      case _NODE_COPY:
      {
        const VkBufferCopy r = { 0, 0, (VkDeviceSize)n->size };
        vkCmdCopyBuffer(cmd, n->src->buffer, n->dst->buffer, 1, &r);
        break;
      }
      case _NODE_COPY_ROWS:
      {
        // Same per-row region math as the eager dt_vulkan_copy_subregion.
        VkBufferCopy *regions = malloc(sizeof(VkBufferCopy) * n->rh);
        if(!regions) return -1;
        const VkDeviceSize row_bytes  = (VkDeviceSize)n->rw * n->bpp;
        const VkDeviceSize src_stride = (VkDeviceSize)n->srp * n->bpp;
        const VkDeviceSize dst_stride = (VkDeviceSize)n->drp * n->bpp;
        for(size_t r = 0; r < n->rh; r++)
        {
          regions[r].srcOffset = (VkDeviceSize)(n->soy + r) * src_stride
                               + (VkDeviceSize)n->sox * n->bpp;
          regions[r].dstOffset = (VkDeviceSize)(n->doy + r) * dst_stride
                               + (VkDeviceSize)n->dox * n->bpp;
          regions[r].size = row_bytes;
        }
        vkCmdCopyBuffer(cmd, n->src->buffer, n->dst->buffer,
                        (uint32_t)n->rh, regions);
        free(regions);  // vkCmdCopyBuffer consumes pRegions at record time
        break;
      }
      case _NODE_DISPATCH:
      {
        const dt_vk_kernel_t *k = &a->d->kernels[n->kernel];
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                k->pipeline_layout, 0, 1, &a->dsets[i], 0, NULL);
        if(n->push_size)
          vkCmdPushConstants(cmd, k->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                             0, n->push_size, n->push);
        vkCmdDispatch(cmd, n->gx, n->gy, n->gz);
        break;
      }
    }
  }
  return 0;
}

gboolean dt_vulkan_capture_begin(int devid)
{
  if(!dt_vulkan_running() || devid != 0) return FALSE;
  if(g_capture && g_capture->active)
  {
    dt_print(DT_DEBUG_ALWAYS,
             "[vkgraph] capture_begin while a capture is already active on "
             "this thread — the nested caller stays on the eager path");
    return FALSE;
  }
  if(!g_capture)
  {
    g_capture = calloc(1, sizeof(*g_capture));
    if(!g_capture) return FALSE;
    g_capture->nodes = g_array_new(FALSE, FALSE, sizeof(_capture_node_t));
    g_capture->dfree = g_ptr_array_new();
    g_capture->taps  = g_array_new(FALSE, FALSE, sizeof(_capture_tap_t));
  }
  g_capture->active = TRUE;
  g_capture->devid = devid;
  g_capture->ring_used = 0;
  g_capture->flushes = 0;
  return TRUE;
}

gboolean dt_vulkan_capture_active(void)
{
  return _cap() != NULL;
}

uint32_t dt_vulkan_capture_pending(void)
{
  const dt_vk_capture_ctx_t *cap = _cap();
  return cap ? cap->nodes->len : 0;
}

int dt_vulkan_capture_flush(int devid)
{
  (void)devid;
  dt_vk_capture_ctx_t *cap = _cap();
  if(!cap) return 0;

  if(!dt_vulkan_running())
  {
    // Device went away mid-capture (shutdown). Drop everything; the
    // deferred buffers die with the device.
    g_array_set_size(cap->nodes, 0);
    cap->ring_used = 0;
    g_ptr_array_set_size(cap->dfree, 0);
    return -1;
  }

  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  const guint nn = cap->nodes->len;
  int rc = 0;

  if(nn > 0)
  {
    // ---- plan: staging offsets for uploads, descriptor demand ----
    VkDeviceSize staging_total = 0;
    guint n_dispatch = 0, n_upload = 0, n_copy = 0;
    uint32_t total_bindings = 0;
    for(guint i = 0; i < nn; i++)
    {
      _capture_node_t *n = &g_array_index(cap->nodes, _capture_node_t, i);
      if(n->kind == _NODE_UPLOAD)
      {
        n->staging_off = staging_total;
        staging_total += (VkDeviceSize)n->size;
        staging_total = (staging_total + 3u) & ~(VkDeviceSize)3u;
        n_upload++;
      }
      else if(n->kind == _NODE_DISPATCH)
      {
        n_dispatch++;
        total_bindings += n->nbufs;
      }
      else
        n_copy++;
    }

    // ---- stage all upload payloads in one map/memcpy pass ----
    dt_vk_mem_t *staging = NULL;
    if(rc == 0 && staging_total > 0)
    {
      staging = _ensure_staging(d, (size_t)staging_total);
      if(!staging) rc = -1;
      void *p = NULL;
      if(rc == 0
         && vkMapMemory(d->device, staging->memory, 0, staging_total, 0, &p) != VK_SUCCESS)
        rc = -1;
      if(rc == 0)
      {
        for(guint i = 0; i < nn; i++)
        {
          const _capture_node_t *n = &g_array_index(cap->nodes, _capture_node_t, i);
          if(n->kind != _NODE_UPLOAD) continue;
          const void *src = n->borrowed ? n->borrowed : cap->ring + n->ring_off;
          memcpy((char *)p + n->staging_off, src, n->size);
        }
        vkUnmapMemory(d->device, staging->memory);
      }
    }

    // ---- one descriptor pool + set per dispatch for this segment ----
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet *dsets = NULL;
    if(rc == 0 && n_dispatch > 0)
    {
      const VkDescriptorPoolSize psz = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                         .descriptorCount = total_bindings };
      const VkDescriptorPoolCreateInfo dpci = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                .maxSets = n_dispatch,
                                                .poolSizeCount = 1, .pPoolSizes = &psz };
      if(vkCreateDescriptorPool(d->device, &dpci, NULL, &pool) != VK_SUCCESS)
        rc = -1;
      dsets = rc == 0 ? calloc(nn, sizeof(VkDescriptorSet)) : NULL;
      if(rc == 0 && !dsets) rc = -1;
      for(guint i = 0; rc == 0 && i < nn; i++)
      {
        const _capture_node_t *n = &g_array_index(cap->nodes, _capture_node_t, i);
        if(n->kind != _NODE_DISPATCH) continue;
        const dt_vk_kernel_t *k = &d->kernels[n->kernel];
        const VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool = pool,
                                                   .descriptorSetCount = 1,
                                                   .pSetLayouts = &k->dset_layout };
        if(vkAllocateDescriptorSets(d->device, &dsai, &dsets[i]) != VK_SUCCESS)
        {
          rc = -1;
          break;
        }
        VkDescriptorBufferInfo bi[DT_VULKAN_MAX_BINDINGS];
        VkWriteDescriptorSet   ws[DT_VULKAN_MAX_BINDINGS];
        for(uint32_t b = 0; b < n->nbufs; b++)
        {
          bi[b] = (VkDescriptorBufferInfo){ .buffer = n->bufs[b]->buffer,
                                            .offset = 0, .range = n->bufs[b]->size };
          ws[b] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = dsets[i], .dstBinding = b,
                                          .descriptorCount = 1,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                          .pBufferInfo = &bi[b] };
        }
        vkUpdateDescriptorSets(d->device, n->nbufs, ws, 0, NULL);
      }
    }

    // ---- record + submit once, wait once ----
    if(rc == 0)
    {
      _graph_record_args_t ga = { .cap = cap, .d = d,
                                  .staging = staging ? staging->buffer : VK_NULL_HANDLE,
                                  .dsets = dsets };
      rc = _submit_one_shot(d, _record_graph, &ga);
    }

    if(pool != VK_NULL_HANDLE) vkDestroyDescriptorPool(d->device, pool, NULL);
    free(dsets);

    dt_print(DT_DEBUG_VKGRAPH,
             "[vkgraph] flush #%u: %u nodes (%u dispatch, %u upload / %.1f MB staged, "
             "%u copy) -> %s",
             cap->flushes, nn, n_dispatch, n_upload, 1e-6 * (double)staging_total,
             n_copy, rc == 0 ? "1 submit" : "FAILED, segment dropped");
  }

  // ---- segment epilogue: reset capture state, run deferred frees ----
  // After the fence wait (or after dropping the nodes) nothing pending
  // references the deferred buffers any more.
  g_array_set_size(cap->nodes, 0);
  cap->ring_used = 0;
  for(guint i = 0; i < cap->dfree->len; i++)
    _release_buffer(d, g_ptr_array_index(cap->dfree, i));
  g_ptr_array_set_size(cap->dfree, 0);
  cap->flushes++;
  return rc;
}

int dt_vulkan_capture_end(int devid)
{
  dt_vk_capture_ctx_t *cap = _cap();
  if(!cap) return 0;
  const int rc = dt_vulkan_capture_flush(devid);
  cap->active = FALSE;
  // M5 §5.4: the final fence has retired every captured dispatch, so
  // the tap result buffers are ready — read them back in one batch
  // (active is now FALSE, so these reads run eagerly, no re-flush) and
  // free them. Skip the reads if the flush failed; the results are
  // undefined then.
  if(dt_vulkan_running())
  {
    dt_vk_device_t *d = &darktable.vulkan->dev[0];
    if(cap->taps->len)
      dt_print(DT_DEBUG_VKGRAPH, "[vkgraph] draining %u GPU tap(s)", cap->taps->len);
    _drain_taps(cap, d, devid, 0, rc == 0);
  }
  // Don't let a trunk-sized snapshot ring linger on a pooled pipeline
  // thread between spans.
  free(cap->ring);
  cap->ring = NULL;
  cap->ring_cap = cap->ring_used = 0;
  return rc;
}

void dt_vulkan_capture_abort(int devid)
{
  (void)devid;
  dt_vk_capture_ctx_t *cap = _cap();
  if(!cap) return;
  if(cap->nodes->len)
    dt_print(DT_DEBUG_VKGRAPH,
             "[vkgraph] abort: dropping %u pending node(s)", cap->nodes->len);
  g_array_set_size(cap->nodes, 0);
  cap->ring_used = 0;
  if(dt_vulkan_running())
  {
    dt_vk_device_t *d = &darktable.vulkan->dev[0];
    for(guint i = 0; i < cap->dfree->len; i++)
      _release_buffer(d, g_ptr_array_index(cap->dfree, i));
    // abandoned run: free tap buffers without reading (results void)
    _drain_taps(cap, d, devid, 0, FALSE);
  }
  g_ptr_array_set_size(cap->dfree, 0);
  cap->active = FALSE;
  free(cap->ring);
  cap->ring = NULL;
  cap->ring_cap = 0;
}

dt_vk_capture_mark_t dt_vulkan_capture_mark(void)
{
  dt_vk_capture_mark_t m = { 0, 0, 0, 0 };
  const dt_vk_capture_ctx_t *cap = _cap();
  if(cap)
  {
    m.nodes = cap->nodes->len;
    m.dfrees = cap->dfree->len;
    m.ring = cap->ring_used;
    m.taps = cap->taps->len;
  }
  return m;
}

void dt_vulkan_capture_rollback(int devid, const dt_vk_capture_mark_t *mark)
{
  (void)devid;
  dt_vk_capture_ctx_t *cap = _cap();
  if(!cap || !mark) return;
  if(cap->nodes->len > mark->nodes)
    dt_print(DT_DEBUG_VKGRAPH,
             "[vkgraph] rollback: dropping %u node(s)",
             cap->nodes->len - mark->nodes);
  _capture_truncate(cap, mark->nodes, mark->ring);
  // Frees deferred after the mark belong to the rolled-back module's
  // scratch; no surviving node references them, so release them now.
  if(dt_vulkan_running())
  {
    dt_vk_device_t *d = &darktable.vulkan->dev[0];
    for(guint i = mark->dfrees; i < cap->dfree->len; i++)
      _release_buffer(d, g_ptr_array_index(cap->dfree, i));
    // taps registered by the rolled-back module: their producing
    // dispatch is gone, so free the result buffers without reading.
    _drain_taps(cap, d, devid, mark->taps, FALSE);
  }
  g_ptr_array_set_size(cap->dfree, mark->dfrees);
}

uint64_t dt_vulkan_submission_count(int devid)
{
  if(!dt_vulkan_running()) return 0;
  (void)devid;
  return darktable.vulkan->dev[0].submit_count;
}

// ---- module helpers --------------------------------------------------
//
// Reduce the per-module wiring boilerplate to ~3 lines. See the
// header for the contract.

void dt_vulkan_module_kernel_load(dt_vk_module_kernel_t *out,
                                  const char *spv_name,
                                  const char *entry,
                                  uint32_t num_storage_buffers,
                                  uint32_t push_constant_size,
                                  uint32_t local_x,
                                  uint32_t local_y,
                                  uint32_t local_z)
{
  out->program = -1;
  out->kernel  = -1;
  if(!dt_vulkan_running()) return;

  out->program = dt_vulkan_load_program_by_name(spv_name);
  if(out->program < 0) return;

  out->kernel = dt_vulkan_create_kernel(out->program, entry,
                                        num_storage_buffers,
                                        push_constant_size,
                                        local_x, local_y, local_z);
  if(out->kernel < 0)
    dt_print(DT_DEBUG_OPENCL,
             "[vulkan] kernel '%s' from '%s' failed to create", entry, spv_name);
}

void dt_vulkan_module_kernel_create_from(dt_vk_module_kernel_t *out,
                                         int program,
                                         const char *entry,
                                         uint32_t num_storage_buffers,
                                         uint32_t push_constant_size,
                                         uint32_t local_x,
                                         uint32_t local_y,
                                         uint32_t local_z)
{
  out->program = program;
  out->kernel  = -1;
  if(!dt_vulkan_running() || program < 0) return;
  // dt_vulkan_create_kernel already logs the missing-entry case
  // with full context; don't duplicate it here.
  out->kernel = dt_vulkan_create_kernel(program, entry,
                                        num_storage_buffers,
                                        push_constant_size,
                                        local_x, local_y, local_z);
}

void dt_vulkan_module_kernel_unload(dt_vk_module_kernel_t *k)
{
  if(!k) return;
  if(k->kernel >= 0) dt_vulkan_free_kernel(k->kernel);
  // Program memory is reclaimed wholesale at dt_vulkan_cleanup; we
  // don't have a per-program unload yet (and modules typically own
  // exactly one kernel per program, so the program slot is freed at
  // cleanup time too).
  k->program = -1;
  k->kernel  = -1;
}

int dt_vulkan_dispatch_n(const dt_vk_module_kernel_t *k,
                         dt_vk_mem_t *const *buffers,
                         size_t buffer_count,
                         size_t global_w,
                         size_t global_h,
                         const void *push_constants,
                         size_t push_constant_size)
{
  if(!k || k->kernel < 0) return -1;
  return dt_vulkan_enqueue_kernel_2d(0, k->kernel, global_w, global_h,
                                     buffers, buffer_count,
                                     push_constants, push_constant_size);
}

int dt_vulkan_dispatch_n_batched(const dt_vk_module_kernel_t *k,
                                 dt_vk_mem_t *const *buffers,
                                 size_t buffer_count,
                                 const dt_vk_upload_t *uploads,
                                 size_t upload_count,
                                 size_t global_w,
                                 size_t global_h,
                                 const void *push_constants,
                                 size_t push_constant_size)
{
  if(!dt_vulkan_running() || !k || k->kernel < 0) return -1;
  if(!buffers || buffer_count > DT_VULKAN_MAX_BINDINGS) return -1;
  if(upload_count > 0 && !uploads) return -1;
  // No uploads? Fall back to the plain dispatch.
  if(upload_count == 0)
    return dt_vulkan_dispatch_n(k, buffers, buffer_count, global_w, global_h,
                                push_constants, push_constant_size);

  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  dt_vk_kernel_t *kk = &d->kernels[k->kernel];
  if(!kk->used) return -1;
  if(buffer_count != kk->num_storage_buffer_bindings) return -1;
  if(push_constant_size != kk->push_constant_size)    return -1;

  // Capture: the uploads become UPLOAD nodes (payloads snapshotted —
  // they are typically stack-local LUTs that die with the caller) and
  // the kernel becomes a DISPATCH node via the enqueue path's capture
  // branch. Any failure winds the context back so a -1 here leaves
  // the graph exactly as before the call.
  dt_vk_capture_ctx_t *cap = _cap();
  if(cap)
  {
    const uint32_t n0 = cap->nodes->len;
    const size_t   r0 = cap->ring_used;
    for(size_t i = 0; i < upload_count; i++)
    {
      size_t off = 0;
      if(!uploads[i].dst || !uploads[i].host
         || _capture_ring_push(cap, uploads[i].host, uploads[i].size, &off) != 0)
      {
        _capture_truncate(cap, n0, r0);
        return -1;
      }
      _capture_node_t *n = _capture_append(cap);
      n->kind = _NODE_UPLOAD;
      n->dst = uploads[i].dst;
      n->size = uploads[i].size;
      n->ring_off = off;
    }
    const int rc = dt_vulkan_enqueue_kernel_2d(0, k->kernel,
                                               global_w, global_h,
                                               buffers, buffer_count,
                                               push_constants, push_constant_size);
    if(rc != 0) _capture_truncate(cap, n0, r0);
    return rc;
  }

  // Pack all uploads contiguously in the shared staging buffer. Each
  // upload's staging offset is 4-byte aligned (Vulkan requires this
  // for vkCmdCopyBuffer srcOffset on non-image buffers).
  VkDeviceSize offsets[DT_VULKAN_MAX_BINDINGS];
  VkDeviceSize total = 0;
  for(size_t i = 0; i < upload_count; i++)
  {
    if(!uploads[i].dst || !uploads[i].host) return -1;
    offsets[i] = total;
    total += uploads[i].size;
    total = (total + 3u) & ~(VkDeviceSize)3u;
  }

  dt_vk_mem_t *staging = _ensure_staging(d, (size_t)total);
  if(!staging) return -1;

  void *p = NULL;
  if(vkMapMemory(d->device, staging->memory, 0, total, 0, &p) != VK_SUCCESS)
    return -1;
  for(size_t i = 0; i < upload_count; i++)
    memcpy((char *)p + offsets[i], uploads[i].host, uploads[i].size);
  vkUnmapMemory(d->device, staging->memory);

  // Allocate + populate the descriptor set for the kernel.
  VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                       .descriptorPool = d->dset_pool,
                                       .descriptorSetCount = 1,
                                       .pSetLayouts = &kk->dset_layout };
  VkDescriptorSet dset = VK_NULL_HANDLE;
  if(vkAllocateDescriptorSets(d->device, &dsai, &dset) != VK_SUCCESS) return -1;

  VkDescriptorBufferInfo bi[DT_VULKAN_MAX_BINDINGS];
  VkWriteDescriptorSet   ws[DT_VULKAN_MAX_BINDINGS];
  for(size_t i = 0; i < buffer_count; i++)
  {
    bi[i] = (VkDescriptorBufferInfo){ .buffer = buffers[i]->buffer,
                                      .offset = 0, .range = buffers[i]->size };
    ws[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    .dstSet = dset, .dstBinding = (uint32_t)i,
                                    .descriptorCount = 1,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    .pBufferInfo = &bi[i] };
  }
  vkUpdateDescriptorSets(d->device, (uint32_t)buffer_count, ws, 0, NULL);

  _batched_args_t ba = {
    .k = kk, .dset = dset,
    .gx = (uint32_t)((global_w + kk->local_size_x - 1) / kk->local_size_x),
    .gy = (uint32_t)((global_h + kk->local_size_y - 1) / kk->local_size_y),
    .gz = 1,
    .push = push_constants, .push_size = push_constant_size,
    .staging = staging->buffer,
    .upload_count = upload_count,
    .uploads = uploads,
    .offsets = offsets,
  };
  int rc = _submit_one_shot(d, _record_batched, &ba);

  vkFreeDescriptorSets(d->device, d->dset_pool, 1, &dset);
  return rc;
}

// The 2- and 3-buffer flavours are thin wrappers over dispatch_n. We
// keep them as named helpers because the common case reads better in
// module code (`dispatch_inout(k, in, out, ...)` vs an explicit
// 2-element array). Callers that need 4+ bindings use dispatch_n.

int dt_vulkan_dispatch_inout(const dt_vk_module_kernel_t *k,
                             dt_vk_mem_t *dev_in,
                             dt_vk_mem_t *dev_out,
                             size_t global_w,
                             size_t global_h,
                             const void *push_constants,
                             size_t push_constant_size)
{
  dt_vk_mem_t *bufs[2] = { dev_in, dev_out };
  return dt_vulkan_dispatch_n(k, bufs, 2, global_w, global_h,
                              push_constants, push_constant_size);
}

int dt_vulkan_dispatch_inout_lut(const dt_vk_module_kernel_t *k,
                                 dt_vk_mem_t *dev_in,
                                 dt_vk_mem_t *dev_out,
                                 dt_vk_mem_t *dev_lut,
                                 size_t global_w,
                                 size_t global_h,
                                 const void *push_constants,
                                 size_t push_constant_size)
{
  dt_vk_mem_t *bufs[3] = { dev_in, dev_out, dev_lut };
  return dt_vulkan_dispatch_n(k, bufs, 3, global_w, global_h,
                              push_constants, push_constant_size);
}

int dt_vulkan_enqueue_kernel_2d(int devid, int kernel,
                                size_t global_w, size_t global_h,
                                dt_vk_mem_t *const *buffers,
                                size_t buffer_count,
                                const void *push_constants,
                                size_t push_constant_size)
{
  if(!dt_vulkan_running() || kernel < 0 || kernel >= DT_VULKAN_MAX_KERNELS) return -1;
  (void)devid;
  dt_vk_device_t *d = &darktable.vulkan->dev[0];
  dt_vk_kernel_t *k = &d->kernels[kernel];
  if(!k->used) return -1;
  if(buffer_count != k->num_storage_buffer_bindings) return -1;
  if(push_constant_size != k->push_constant_size)    return -1;
  for(size_t i = 0; i < buffer_count; ++i)
    if(!buffers[i]) return -1;

  // Capture: same validation as the eager path just ran, so a module
  // whose dispatch would fail eagerly fails identically here and its
  // fallback logic stays truthful. Workgroup counts are resolved now;
  // push constants are snapshotted (module-local storage).
  dt_vk_capture_ctx_t *cap = _cap();
  if(cap)
  {
    _capture_node_t *n = _capture_append(cap);
    n->kind = _NODE_DISPATCH;
    n->kernel = kernel;
    n->gx = (uint32_t)((global_w + k->local_size_x - 1) / k->local_size_x);
    n->gy = (uint32_t)((global_h + k->local_size_y - 1) / k->local_size_y);
    n->gz = 1;
    n->nbufs = (uint32_t)buffer_count;
    for(size_t i = 0; i < buffer_count; ++i) n->bufs[i] = buffers[i];
    n->push_size = (uint32_t)push_constant_size;
    if(push_constant_size && push_constants)
      memcpy(n->push, push_constants, push_constant_size);
    return 0;
  }

  // Allocate a descriptor set from the device pool.
  VkDescriptorSetAllocateInfo dsai = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                       .descriptorPool = d->dset_pool,
                                       .descriptorSetCount = 1,
                                       .pSetLayouts = &k->dset_layout };
  VkDescriptorSet dset = VK_NULL_HANDLE;
  if(vkAllocateDescriptorSets(d->device, &dsai, &dset) != VK_SUCCESS) return -1;

  // Wire buffers into the descriptor set.
  VkDescriptorBufferInfo bi[DT_VULKAN_MAX_BINDINGS];
  VkWriteDescriptorSet   ws[DT_VULKAN_MAX_BINDINGS];
  for(size_t i = 0; i < buffer_count; ++i)
  {
    bi[i] = (VkDescriptorBufferInfo){ .buffer = buffers[i]->buffer,
                                      .offset = 0, .range = buffers[i]->size };
    ws[i] = (VkWriteDescriptorSet){ .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    .dstSet = dset, .dstBinding = (uint32_t)i,
                                    .descriptorCount = 1,
                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    .pBufferInfo = &bi[i] };
  }
  vkUpdateDescriptorSets(d->device, (uint32_t)buffer_count, ws, 0, NULL);

  _dispatch_args_t da = {
    .k = k, .dset = dset,
    .gx = (uint32_t)((global_w + k->local_size_x - 1) / k->local_size_x),
    .gy = (uint32_t)((global_h + k->local_size_y - 1) / k->local_size_y),
    .gz = 1,
    .push = push_constants, .push_size = push_constant_size,
  };
  int rc = _submit_one_shot(d, _record_dispatch, &da);

  // Return the descriptor set to the pool so the next dispatch can reuse it.
  vkFreeDescriptorSets(d->device, d->dset_pool, 1, &dset);
  return rc;
}

#endif // HAVE_VULKAN
