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

#include "Suballocator.h"
// #include <assert> #include <string> are included in vulkan.hpp
#include <vulkan/vulkan.hpp>

namespace rtxmu
{
#if VK_HEADER_VERSION >= 301
    typedef vk::detail::DynamicLoader VkDynamicLoader;
    typedef vk::detail::DispatchLoaderDynamic VkDispatchLoaderDynamic;
#else
    typedef vk::DynamicLoader VkDynamicLoader;
    typedef vk::DispatchLoaderDynamic VkDispatchLoaderDynamic;
#endif

    constexpr uint32_t DefaultBlockAlignment = 65536;

    struct Allocator
    {
        vk::Instance       instance;
        vk::Device         device;
        vk::PhysicalDevice physicalDevice;
    };

    class VkBlock
    {
    public:
        static VkDispatchLoaderDynamic& getDispatchLoader();

        static uint32_t getMemoryIndex(const vk::PhysicalDevice& physicalDevice,
                                       uint32_t                  memoryTypeBits,
                                       vk::MemoryPropertyFlags   propFlags,
                                       vk::MemoryHeapFlags       heapFlags);

        static vk::DeviceMemory getMemory(VkBlock block);

        static vk::DeviceAddress getDeviceAddress(const vk::Device& device,
                                                  VkBlock           block,
                                                  uint64_t          offset);

        void allocate(vk::DeviceSize          size,
                      vk::BufferUsageFlags    usageFlags,
                      vk::MemoryPropertyFlags propFlags,
                      vk::MemoryHeapFlags     heapflags,
                      uint32_t                alignment);

        void free();

        uint64_t getVMA();

        vk::Buffer getBuffer();

        static void setAllocator(Allocator* allocator);

    protected:
        static Allocator* m_allocator;

    private:
        static VkDispatchLoaderDynamic m_dispatchLoader;
        vk::DeviceMemory               m_memory = nullptr;
        vk::Buffer                     m_buffer = nullptr;
    };

    class VkScratchBlock : public VkBlock
    {
    public:
        static constexpr vk::BufferUsageFlags    usageFlags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        static constexpr vk::MemoryPropertyFlags propertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        static constexpr vk::MemoryHeapFlags     heapFlags = vk::MemoryHeapFlagBits::eDeviceLocal;
        static constexpr uint32_t                alignment = DefaultBlockAlignment;

        uint32_t getAlignment() { return alignment; }

        void allocate(vk::DeviceSize size, std::string name)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags, alignment);

            if (Logger::isEnabled(Level::DBG))
            {
                char buf[128];
                snprintf(buf, sizeof buf, "RTXMU Scratch Suballocator Block Allocation of size %" PRIu64 "\n", size);
                Logger::log(Level::DBG, buf);
            }
        }

        void free()
        {
            if (Logger::isEnabled(Level::DBG))
            {
                Logger::log(Level::DBG, "RTXMU Scratch Suballocator Block Release\n");
            }
            VkBlock::free();
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

        void allocate(vk::DeviceSize size, std::string name)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags, alignment);

            if (Logger::isEnabled(Level::DBG))
            {
                char buf[128];
                snprintf(buf, sizeof buf, "RTXMU Result BLAS Suballocator Block Allocation of size %" PRIu64 "\n", size);
                Logger::log(Level::DBG, buf);
            }
        }

        void free()
        {
            if (Logger::isEnabled(Level::DBG))
            {
                Logger::log(Level::DBG, "RTXMU Result BLAS Suballocator Block Release\n");
            }
            VkBlock::free();
        }
    };

    class VkCompactedAccelStructBlock : public VkBlock
    {
    public:
        static constexpr vk::BufferUsageFlags    usageFlags = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        static constexpr vk::MemoryPropertyFlags propertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
        static constexpr vk::MemoryHeapFlags     heapFlags = vk::MemoryHeapFlagBits::eDeviceLocal;
        static constexpr uint32_t                alignment = DefaultBlockAlignment;

        vk::AccelerationStructureKHR             m_asHandle;

        uint32_t getAlignment() { return alignment; }

        void allocate(vk::DeviceSize size, std::string name)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags, alignment);

            if (Logger::isEnabled(Level::DBG))
            {
                char buf[128];
                snprintf(buf, sizeof buf, "RTXMU Compacted BLAS Suballocator Block Allocation of size %" PRIu64 "\n", size);
                Logger::log(Level::DBG, buf);
            }
        }

        void free()
        {
            if (Logger::isEnabled(Level::DBG))
            {
                Logger::log(Level::DBG, "RTXMU Compacted BLAS Suballocator Block Release\n");
            }
            VkBlock::free();
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

        void allocate(vk::DeviceSize size, std::string name)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags, alignment);

            if (Logger::isEnabled(Level::DBG))
            {
                char buf[128];
                snprintf(buf, sizeof buf, "RTXMU Readback CPU Suballocator Block Allocation of size %" PRIu64 "\n", size);
                Logger::log(Level::DBG, buf);
            }
        }

        void free()
        {
            if (Logger::isEnabled(Level::DBG))
            {
                Logger::log(Level::DBG, "RTXMU Readback CPU Suballocator Block Release\n");
            }
            VkBlock::free();
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

        void allocate(vk::DeviceSize size, std::string name)
        {
            VkBlock::allocate(size, usageFlags, propertyFlags, heapFlags, alignment);

            if (Logger::isEnabled(Level::DBG))
            {
                char buf[128];
                snprintf(buf, sizeof buf, "RTXMU Compaction Size GPU Suballocator Block Allocation of size %" PRIu64 "\n", size);
                Logger::log(Level::DBG, buf);
            }
        }

        void free()
        {
            if (Logger::isEnabled(Level::DBG))
            {
                Logger::log(Level::DBG, "RTXMU Compaction Size GPU Suballocator Block Release\n");
            }
            VkBlock::free();
        }
    };

    class VkQueryBlock : public VkBlock
    {
    public:
        static constexpr uint32_t alignment = 8;
        vk::QueryPool queryPool = nullptr;

        uint32_t getAlignment() { return alignment; }

        void allocate(vk::DeviceSize size, std::string name)
        {
            auto queryPoolInfo = vk::QueryPoolCreateInfo()
                .setQueryType(vk::QueryType::eAccelerationStructureCompactedSizeKHR)
                .setQueryCount((uint32_t)size);

            queryPool = m_allocator->device.createQueryPool(queryPoolInfo, nullptr, VkBlock::getDispatchLoader());

            if (Logger::isEnabled(Level::DBG))
            {
                char buf[128];
                snprintf(buf, sizeof buf, "RTXMU Compaction Query Suballocator Block Allocation of size %" PRIu64 "\n", size);
                Logger::log(Level::DBG, buf);
            }
        }

        void free()
        {
            if (queryPool)
            {
                m_allocator->device.destroyQueryPool(queryPool, nullptr, VkBlock::getDispatchLoader());

                if (Logger::isEnabled(Level::DBG))
                {
                    Logger::log(Level::DBG, "RTXMU Compaction Query Suballocator Block Release\n");
                }
            }
            VkBlock::free();
        }
    };
}
