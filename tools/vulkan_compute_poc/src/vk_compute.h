/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Vulkan compute runtime for the clspv proof-of-concept.

    This is intentionally a thin slice of what a future dt_gpu HAL would
    have to provide. It mirrors the parts of src/common/opencl.{c,h} that
    a real port would replace first:

      - device/queue selection,
      - SPIR-V module load + compute pipeline build,
      - buffer alloc / host-to-device upload / device-to-host readback,
      - descriptor-set bind + dispatch (the analogue of
        dt_opencl_enqueue_kernel_2d_args).
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

namespace dt::vkpoc {

struct Buffer {
    VkBuffer        handle      = VK_NULL_HANDLE;
    VkDeviceMemory  memory      = VK_NULL_HANDLE;
    VkDeviceSize    size        = 0;
    bool            host_visible = false;
};

struct Pipeline {
    VkShaderModule        shader_module     = VK_NULL_HANDLE;
    VkDescriptorSetLayout dset_layout       = VK_NULL_HANDLE;
    VkPipelineLayout      pipeline_layout   = VK_NULL_HANDLE;
    VkPipeline            pipeline          = VK_NULL_HANDLE;
    VkDescriptorPool      dset_pool         = VK_NULL_HANDLE;
    VkDescriptorSet       dset              = VK_NULL_HANDLE;
    uint32_t              local_size_x      = 1;
    uint32_t              local_size_y      = 1;
    uint32_t              local_size_z      = 1;
    uint32_t              push_constant_size = 0;
    std::string           entry_point;
};

class Device {
public:
    Device();
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    bool init(bool enable_validation, std::string& error_out);

    const std::string& device_name() const { return device_name_; }
    uint32_t           queue_family_index() const { return compute_queue_family_; }
    VkDevice           vk()  const { return device_; }
    VkPhysicalDevice   phys() const { return phys_device_; }

    Buffer alloc_buffer(VkDeviceSize size,
                        VkBufferUsageFlags usage,
                        bool host_visible);
    void   free_buffer(Buffer& buf);

    void upload(Buffer& dst, const void* host_src, VkDeviceSize size);
    void download(const Buffer& src, void* host_dst, VkDeviceSize size);

    Pipeline create_pipeline(const std::vector<uint32_t>& spirv,
                             const std::string& entry_point,
                             uint32_t           num_storage_buffer_bindings,
                             uint32_t           push_constant_size,
                             uint32_t           local_size_x,
                             uint32_t           local_size_y,
                             uint32_t           local_size_z);
    void destroy_pipeline(Pipeline& p);

    void bind_buffers(Pipeline& p, const std::vector<Buffer*>& bindings);

    // Equivalent of dt_opencl_enqueue_kernel_2d_args: dispatch + wait.
    void dispatch(const Pipeline& p,
                  const void*     push_constants,
                  uint32_t        global_x,
                  uint32_t        global_y,
                  uint32_t        global_z = 1);

private:
    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) const;
    void     submit_and_wait(VkCommandBuffer cmd);

    VkInstance          instance_           = VK_NULL_HANDLE;
    VkPhysicalDevice    phys_device_        = VK_NULL_HANDLE;
    VkDevice            device_             = VK_NULL_HANDLE;
    VkQueue             compute_queue_      = VK_NULL_HANDLE;
    uint32_t            compute_queue_family_ = 0;
    VkCommandPool       command_pool_       = VK_NULL_HANDLE;
    std::string         device_name_;

    VkPhysicalDeviceMemoryProperties mem_props_{};
};

// Read a .spv file into a std::vector<uint32_t> ready for vkCreateShaderModule.
bool load_spirv(const std::string& path, std::vector<uint32_t>& out, std::string& error_out);

} // namespace dt::vkpoc
