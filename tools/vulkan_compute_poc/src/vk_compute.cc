/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
*/

#include "vk_compute.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace dt::vkpoc {

namespace {

const char* vkres(VkResult r) {
    switch(r) {
        case VK_SUCCESS:                          return "VK_SUCCESS";
        case VK_ERROR_OUT_OF_HOST_MEMORY:         return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:       return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:      return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT:          return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:      return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:        return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:        return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_DEVICE_LOST:                return "VK_ERROR_DEVICE_LOST";
        default: break;
    }
    return "VK_<unknown>";
}

#define VKCHECK(call)                                                       \
    do {                                                                     \
        VkResult _r = (call);                                                \
        if (_r != VK_SUCCESS) {                                              \
            char _b[256];                                                    \
            std::snprintf(_b, sizeof(_b), "%s failed: %s",                   \
                          #call, vkres(_r));                                 \
            throw std::runtime_error(_b);                                    \
        }                                                                    \
    } while(0)

} // namespace

Device::Device() = default;

Device::~Device() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

bool Device::init(bool enable_validation, std::string& error_out) {
    try {
        VkApplicationInfo app{};
        app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName   = "darktable-vk-poc";
        app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        app.pEngineName        = "darktable";
        app.engineVersion      = VK_MAKE_VERSION(5, 0, 0);
        app.apiVersion         = VK_API_VERSION_1_2;

        std::vector<const char*> layers;
        if (enable_validation) layers.push_back("VK_LAYER_KHRONOS_validation");

        VkInstanceCreateInfo ici{};
        ici.sType                = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo     = &app;
        ici.enabledLayerCount    = (uint32_t)layers.size();
        ici.ppEnabledLayerNames  = layers.data();
        VKCHECK(vkCreateInstance(&ici, nullptr, &instance_));

        uint32_t n = 0;
        VKCHECK(vkEnumeratePhysicalDevices(instance_, &n, nullptr));
        if (n == 0) {
            error_out = "no Vulkan physical devices";
            return false;
        }
        std::vector<VkPhysicalDevice> phys(n);
        VKCHECK(vkEnumeratePhysicalDevices(instance_, &n, phys.data()));

        // Pick the first discrete GPU if any, else the first device.
        phys_device_ = phys[0];
        for (auto p : phys) {
            VkPhysicalDeviceProperties pp;
            vkGetPhysicalDeviceProperties(p, &pp);
            if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                phys_device_ = p;
                break;
            }
        }
        VkPhysicalDeviceProperties pp;
        vkGetPhysicalDeviceProperties(phys_device_, &pp);
        device_name_ = pp.deviceName;

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_device_, &qn, qprops.data());

        compute_queue_family_ = UINT32_MAX;
        for (uint32_t i = 0; i < qn; ++i) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                compute_queue_family_ = i;
                break;
            }
        }
        if (compute_queue_family_ == UINT32_MAX) {
            error_out = "no compute-capable queue family";
            return false;
        }

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = compute_queue_family_;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{};
        dci.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos    = &qci;
        VKCHECK(vkCreateDevice(phys_device_, &dci, nullptr, &device_));

        vkGetDeviceQueue(device_, compute_queue_family_, 0, &compute_queue_);
        vkGetPhysicalDeviceMemoryProperties(phys_device_, &mem_props_);

        VkCommandPoolCreateInfo cpci{};
        cpci.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = compute_queue_family_;
        cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VKCHECK(vkCreateCommandPool(device_, &cpci, nullptr, &command_pool_));

        return true;
    } catch (const std::exception& e) {
        error_out = e.what();
        return false;
    }
}

uint32_t Device::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const {
    for (uint32_t i = 0; i < mem_props_.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (mem_props_.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    throw std::runtime_error("no suitable memory type");
}

Buffer Device::alloc_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible) {
    Buffer b{};
    b.size = size;
    b.host_visible = host_visible;

    VkBufferCreateInfo bci{};
    bci.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKCHECK(vkCreateBuffer(device_, &bci, nullptr, &b.handle));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device_, b.handle, &req);

    VkMemoryPropertyFlags props = host_visible
        ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkMemoryAllocateInfo ai{};
    ai.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, props);
    VKCHECK(vkAllocateMemory(device_, &ai, nullptr, &b.memory));
    VKCHECK(vkBindBufferMemory(device_, b.handle, b.memory, 0));
    return b;
}

void Device::free_buffer(Buffer& buf) {
    if (buf.handle) vkDestroyBuffer(device_, buf.handle, nullptr);
    if (buf.memory) vkFreeMemory(device_, buf.memory, nullptr);
    buf = {};
}

void Device::upload(Buffer& dst, const void* host_src, VkDeviceSize size) {
    if (dst.host_visible) {
        void* p = nullptr;
        VKCHECK(vkMapMemory(device_, dst.memory, 0, size, 0, &p));
        std::memcpy(p, host_src, size);
        vkUnmapMemory(device_, dst.memory);
        return;
    }

    // For device-local: staging through a host-visible buffer.
    Buffer staging = alloc_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    upload(staging, host_src, size);

    VkCommandBufferAllocateInfo cbi{};
    cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool        = command_pool_;
    cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VKCHECK(vkAllocateCommandBuffers(device_, &cbi, &cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cmd, &bi));
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, staging.handle, dst.handle, 1, &region);
    VKCHECK(vkEndCommandBuffer(cmd));
    submit_and_wait(cmd);

    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
    free_buffer(staging);
}

void Device::download(const Buffer& src, void* host_dst, VkDeviceSize size) {
    if (src.host_visible) {
        void* p = nullptr;
        VKCHECK(vkMapMemory(device_, src.memory, 0, size, 0, &p));
        std::memcpy(host_dst, p, size);
        vkUnmapMemory(device_, src.memory);
        return;
    }

    Buffer staging = alloc_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);

    VkCommandBufferAllocateInfo cbi{};
    cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool        = command_pool_;
    cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VKCHECK(vkAllocateCommandBuffers(device_, &cbi, &cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cmd, &bi));
    VkBufferCopy region{0, 0, size};
    vkCmdCopyBuffer(cmd, src.handle, staging.handle, 1, &region);
    VKCHECK(vkEndCommandBuffer(cmd));
    submit_and_wait(cmd);

    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);

    void* p = nullptr;
    VKCHECK(vkMapMemory(device_, staging.memory, 0, size, 0, &p));
    std::memcpy(host_dst, p, size);
    vkUnmapMemory(device_, staging.memory);
    free_buffer(staging);
}

Pipeline Device::create_pipeline(const std::vector<uint32_t>& spirv,
                                 const std::string& entry_point,
                                 uint32_t           num_storage_buffer_bindings,
                                 uint32_t           push_constant_size,
                                 uint32_t           local_size_x,
                                 uint32_t           local_size_y,
                                 uint32_t           local_size_z) {
    Pipeline p{};
    p.entry_point        = entry_point;
    p.local_size_x       = local_size_x;
    p.local_size_y       = local_size_y;
    p.local_size_z       = local_size_z;
    p.push_constant_size = push_constant_size;

    VkShaderModuleCreateInfo smci{};
    smci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = spirv.size() * sizeof(uint32_t);
    smci.pCode    = spirv.data();
    VKCHECK(vkCreateShaderModule(device_, &smci, nullptr, &p.shader_module));

    std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_buffer_bindings);
    for (uint32_t i = 0; i < num_storage_buffer_bindings; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = (uint32_t)bindings.size();
    dslci.pBindings    = bindings.data();
    VKCHECK(vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &p.dset_layout));

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc.offset     = 0;
    pc.size       = push_constant_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &p.dset_layout;
    plci.pushConstantRangeCount = push_constant_size ? 1 : 0;
    plci.pPushConstantRanges    = push_constant_size ? &pc : nullptr;
    VKCHECK(vkCreatePipelineLayout(device_, &plci, nullptr, &p.pipeline_layout));

    VkPipelineShaderStageCreateInfo ssci{};
    ssci.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = p.shader_module;
    ssci.pName  = p.entry_point.c_str();

    VkComputePipelineCreateInfo cpci{};
    cpci.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage  = ssci;
    cpci.layout = p.pipeline_layout;
    VKCHECK(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline));

    VkDescriptorPoolSize psz{};
    psz.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    psz.descriptorCount = num_storage_buffer_bindings;

    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes    = &psz;
    VKCHECK(vkCreateDescriptorPool(device_, &dpci, nullptr, &p.dset_pool));

    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool     = p.dset_pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &p.dset_layout;
    VKCHECK(vkAllocateDescriptorSets(device_, &dsai, &p.dset));

    return p;
}

void Device::destroy_pipeline(Pipeline& p) {
    if (p.dset_pool)       vkDestroyDescriptorPool(device_, p.dset_pool, nullptr);
    if (p.pipeline)        vkDestroyPipeline(device_, p.pipeline, nullptr);
    if (p.pipeline_layout) vkDestroyPipelineLayout(device_, p.pipeline_layout, nullptr);
    if (p.dset_layout)     vkDestroyDescriptorSetLayout(device_, p.dset_layout, nullptr);
    if (p.shader_module)   vkDestroyShaderModule(device_, p.shader_module, nullptr);
    p = {};
}

void Device::bind_buffers(Pipeline& p, const std::vector<Buffer*>& bindings) {
    std::vector<VkDescriptorBufferInfo> infos(bindings.size());
    std::vector<VkWriteDescriptorSet>   writes(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i) {
        infos[i].buffer = bindings[i]->handle;
        infos[i].offset = 0;
        infos[i].range  = bindings[i]->size;

        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = p.dset;
        writes[i].dstBinding      = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &infos[i];
    }
    vkUpdateDescriptorSets(device_, (uint32_t)writes.size(), writes.data(), 0, nullptr);
}

void Device::dispatch(const Pipeline& p,
                      const void*     push_constants,
                      uint32_t        global_x,
                      uint32_t        global_y,
                      uint32_t        global_z) {
    VkCommandBufferAllocateInfo cbi{};
    cbi.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool        = command_pool_;
    cbi.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VKCHECK(vkAllocateCommandBuffers(device_, &cbi, &cmd));

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cmd, &bi));

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            p.pipeline_layout, 0, 1, &p.dset, 0, nullptr);
    if (p.push_constant_size && push_constants) {
        vkCmdPushConstants(cmd, p.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, p.push_constant_size, push_constants);
    }

    // Same global-vs-local sizing dance as OpenCL: round up.
    uint32_t gx = (global_x + p.local_size_x - 1) / p.local_size_x;
    uint32_t gy = (global_y + p.local_size_y - 1) / p.local_size_y;
    uint32_t gz = (global_z + p.local_size_z - 1) / p.local_size_z;
    vkCmdDispatch(cmd, gx, gy, gz);

    VKCHECK(vkEndCommandBuffer(cmd));
    submit_and_wait(cmd);
    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}

void Device::submit_and_wait(VkCommandBuffer cmd) {
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VKCHECK(vkCreateFence(device_, &fci, nullptr, &fence));
    VKCHECK(vkQueueSubmit(compute_queue_, 1, &si, fence));
    VKCHECK(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device_, fence, nullptr);
}

bool load_spirv(const std::string& path, std::vector<uint32_t>& out, std::string& error_out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        error_out = "cannot open " + path;
        return false;
    }
    std::streamsize sz = f.tellg();
    if (sz <= 0 || (sz % 4) != 0) {
        error_out = "invalid SPIR-V size at " + path;
        return false;
    }
    f.seekg(0);
    out.resize((size_t)sz / 4);
    if (!f.read(reinterpret_cast<char*>(out.data()), sz)) {
        error_out = "short read on " + path;
        return false;
    }
    return true;
}

} // namespace dt::vkpoc
