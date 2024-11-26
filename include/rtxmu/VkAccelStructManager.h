/*
* Copyright (c) 2024 NVIDIA CORPORATION. All rights reserved
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#pragma once

#include "AccelStructManager.h"
#include "VulkanSuballocator.h"
#include <memory>

namespace rtxmu
{
    struct VkAccelerationStructure : AccelerationStructure
    {
        Suballocator<Allocator, VkScratchBlock>::SubAllocation updateGpuMemory;
        Suballocator<Allocator, VkScratchBlock>::SubAllocation scratchGpuMemory;
        Suballocator<Allocator, VkAccelStructBlock>::SubAllocation resultGpuMemory;
        Suballocator<Allocator, VkAccelStructBlock>::SubAllocation compactionGpuMemory;
        Suballocator<Allocator, VkQueryBlock>::SubAllocation queryCompactionSizeMemory;
    };

    class VkAccelStructManager : public AccelStructManager<VkAccelerationStructure>
    {
    public:

        VkAccelStructManager(const vk::Instance&       instance,
                             const vk::Device&         device,
                             const vk::PhysicalDevice& physicalDevice,
                             Level                     verbosity = Level::DISABLED);

        // Initializes suballocator block size
        void Initialize(uint32_t suballocatorBlockSize = DefaultSuballocatorBlockSize);

        // Resets all queues and frees all memory in suballocators
        void Reset();

        void PopulateUpdateCommandList(vk::CommandBuffer                                  commandList,
                                       vk::AccelerationStructureBuildGeometryInfoKHR*     geomInfos,
                                       const vk::AccelerationStructureBuildRangeInfoKHR** rangeInfos,
                                       const uint32_t**                                   maxPrimitiveCounts,
                                       const uint32_t                                     buildCount,
                                       std::vector<uint64_t>&                             accelStructIds);

        // Receives acceleration structure inputs and returns a command list with build commands
        void PopulateBuildCommandList(vk::CommandBuffer                                  commandList,
                                      vk::AccelerationStructureBuildGeometryInfoKHR*     geomInfos,
                                      const vk::AccelerationStructureBuildRangeInfoKHR** rangeInfos,
                                      const uint32_t**                                   maxPrimitiveCounts,
                                      const uint32_t                                     buildCount,
                                      std::vector<uint64_t>&                             accelStructIds);

        // Returns a command list with compaction copies if the acceleration structures are ready to be compacted
        void PopulateCompactionCommandList(vk::CommandBuffer commandList,
                                           const std::vector<uint64_t>& accelStructIds);

        // Receives acceleration structure inputs and places UAV barriers for them
        void PopulateUAVBarriersCommandList(vk::CommandBuffer commandList,
                                            const std::vector<uint64_t>& accelStructIds);

        // Performs copies to bring over any compaction size data
        void PopulateCompactionSizeCopiesCommandList(vk::CommandBuffer commandList,
                                                     const std::vector<uint64_t>& accelStructIds);

        // Remove all memory that an Acceleration Structure might use
        void RemoveAccelerationStructures(const std::vector<uint64_t>& accelStructIds);

        // Remove all memory used in build process, while only leaving the acceleration structure buffer itself in memory
        void GarbageCollection(const std::vector<uint64_t>& accelStructIds);

        // Returns GPUVA of the acceleration structure
        vk::DeviceMemory GetMemory(const uint64_t accelStructId);

        vk::DeviceSize GetMemoryOffset(const uint64_t accelStructId);

        vk::DeviceAddress GetDeviceAddress(const uint64_t accelStructId);

        vk::AccelerationStructureKHR GetAccelerationStruct(const uint64_t accelStructId);

        vk::AccelerationStructureKHR GetAccelerationStructCompacted(const uint64_t accelStructId);

        vk::Buffer GetBuffer(const uint64_t accelStructId);

        // Returns prebuild size of allocation
        uint64_t GetInitialAccelStructSize(const uint64_t accelStructId);

        // Returns size of compacted allocation
        uint64_t GetCompactedAccelStructSize(const uint64_t accelStructId);

        // Returns whether the acceleration structure requested compaction
        bool GetRequestedCompaction(const uint64_t accelStructId);

        // Returns whether the acceleration structure is ready in compaction state
        bool GetCompactionComplete(const uint64_t accelStructId);

        // Returns if acceleration structure is being tracked
        bool IsValid(const uint64_t accelStructId);

        // Returns a log containing memory consumption information
        const char* GetLog();

        Stats GetResultPoolMemoryStats();

        Stats GetTransientResultPoolMemoryStats();

        Stats GetCompactionPoolMemoryStats();

        static void logCallbackFunction(const char* msg);

    private:

        void PostBuildRelease(const uint64_t accelStructId);

        void ReleaseAccelerationStructures(const uint64_t accelStructId);

        Allocator m_allocator;

        // Suballocation buffers
        std::unique_ptr<Suballocator<Allocator, VkScratchBlock>>     m_scratchPool;
        std::unique_ptr<Suballocator<Allocator, VkScratchBlock>>     m_updatePool;
        std::unique_ptr<Suballocator<Allocator, VkAccelStructBlock>> m_resultPool;
        std::unique_ptr<Suballocator<Allocator, VkAccelStructBlock>> m_transientResultPool;
        std::unique_ptr<Suballocator<Allocator, VkAccelStructBlock>> m_compactionPool;
        std::unique_ptr<Suballocator<Allocator, VkQueryBlock>>       m_queryCompactionSizePool;
    };
}
