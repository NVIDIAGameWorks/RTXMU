/*
* Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved
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

#include "Suballocator.h"
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

namespace rtxmu
{
    constexpr uint32_t DefaultBlockAlignment = 65536;

    class VkBlock
    {
    public:
        static vk::DispatchLoaderDynamic  m_dispatchLoader;
        static vk::Instance               m_instance;
        static vk::Device                 m_device;
        static vk::PhysicalDevice         m_physicalDevice;
        vk::DeviceMemory                  m_memory = nullptr;
        vk::Buffer                        m_buffer = nullptr;

        static uint32_t getMemoryIndex(uint32_t memoryTypeBits, vk::MemoryPropertyFlags propFlags, vk::MemoryHeapFlags heapFlags)
        {
            vk::PhysicalDeviceMemoryProperties memProperties;
            m_physicalDevice.getMemoryProperties(&memProperties, m_dispatchLoader);

            for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
            {
                vk::MemoryType* memoryType = &memProperties.memoryTypes[i];
                if ((memoryTypeBits & (1 << i)) &&
                    (memoryType->propertyFlags & propFlags) &&
                    (memProperties.memoryHeaps[memoryType->heapIndex].flags & heapFlags))
                {
                    return i;

                }
            }
            return 0;
        }

        static vk::DeviceMemory getMemory(VkBlock block)
        {
            return block.m_memory;
        }

        static vk::DeviceAddress getDeviceAddress(VkBlock block, uint64_t offset)
        {
            auto addrInfo = vk::BufferDeviceAddressInfo()
                .setBuffer(block.m_buffer);

            return VkBlock::m_device.getBufferAddress(addrInfo) + offset;
        }

        void allocate(vk::DeviceSize             size,
            vk::BufferUsageFlags       usageFlags,
            vk::MemoryPropertyFlags    propFlags,
            vk::MemoryHeapFlags        heapflags)
        {
            auto bufferInfo = vk::BufferCreateInfo()
                .setSize(size)
                .setUsage(usageFlags)
                .setSharingMode(vk::SharingMode::eExclusive);

            m_buffer = m_device.createBuffer(bufferInfo);

            vk::MemoryRequirements memoryRequirements = m_device.getBufferMemoryRequirements(m_buffer, m_dispatchLoader);
            uint32_t memoryTypeIndex = getMemoryIndex(memoryRequirements.memoryTypeBits, propFlags, heapflags);


            auto memoryInfo = vk::MemoryAllocateInfo()
                .setAllocationSize(size)
                .setMemoryTypeIndex(memoryTypeIndex);

            m_memory = m_device.allocateMemory(memoryInfo);
            m_device.bindBufferMemory(m_buffer, m_memory, 0);
        }

        void free()
        {
            m_device.destroyBuffer(m_buffer);
            m_device.freeMemory(m_memory);
        }

        uint64_t getVMA() { return (uint64_t)(VkDeviceMemory)(m_memory); }
    };

    class VkScratchBlock : public VkBlock
    {
    public:
        static constexpr vk::BufferUsageFlags    usageFlags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        static constexpr vk::MemoryPropertyFlags propertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        static constexpr vk::MemoryHeapFlags     heapFlags = vk::MemoryHeapFlagBits::eDeviceLocal;
        static constexpr uint32_t                alignment = DefaultBlockAlignment;

        uint32_t getAlignment() { return alignment; }

        void allocate(vk::DeviceSize size)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags);
        }
    };

    class VkAccelStructBlock : public VkBlock
    {
    public:
        static constexpr vk::BufferUsageFlags    usageFlags = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        static constexpr vk::MemoryPropertyFlags propertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        static constexpr vk::MemoryHeapFlags     heapFlags = vk::MemoryHeapFlagBits::eDeviceLocal;
        static constexpr uint32_t                alignment = DefaultBlockAlignment;

        vk::AccelerationStructureKHR             m_asHandle;

        uint32_t getAlignment() { return alignment; }

        void allocate(vk::DeviceSize size)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags);
        }
    };

    class VkReadBackBlock : public VkBlock
    {
    public:
        static constexpr vk::BufferUsageFlags    usageFlags = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
        static constexpr vk::MemoryPropertyFlags propertyFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached;
        static constexpr vk::MemoryHeapFlags     heapFlags = vk::MemoryHeapFlagBits::eDeviceLocal;
        static constexpr uint32_t                alignment = DefaultBlockAlignment;

        uint32_t getAlignment() { return alignment; }

        void allocate(vk::DeviceSize size)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags);
        }
    };

    class VkCompactionWriteBlock : public VkBlock
    {
    public:
        static constexpr vk::BufferUsageFlags    usageFlags = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
        static constexpr vk::MemoryPropertyFlags propertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        static constexpr vk::MemoryHeapFlags     heapFlags = vk::MemoryHeapFlagBits::eDeviceLocal;
        static constexpr uint32_t                alignment = DefaultBlockAlignment;

        uint32_t getAlignment() { return alignment; }

        void allocate(vk::DeviceSize size)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags);
        }
    };

    class VkQueryBlock : public VkBlock
    {
    public:
        static constexpr uint32_t alignment = 8;
        vk::QueryPool queryPool = nullptr;

        uint32_t getAlignment() { return alignment; }

        void allocate(vk::DeviceSize size)
        {
            auto queryPoolInfo = vk::QueryPoolCreateInfo()
                .setQueryType(vk::QueryType::eAccelerationStructureCompactedSizeKHR)
                .setQueryCount((uint32_t)size);
            queryPool = VkBlock::m_device.createQueryPool(queryPoolInfo, nullptr, m_dispatchLoader);
        }

        void free()
        {
            if (queryPool)
                VkBlock::m_device.destroyQueryPool(queryPool, nullptr, m_dispatchLoader);
        }
    };
}
