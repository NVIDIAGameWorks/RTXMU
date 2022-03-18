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

#include <rtxmu/VulkanSuballocator.h>

namespace rtxmu
{
    vk::DispatchLoaderDynamic VkBlock::m_dispatchLoader;

    vk::DispatchLoaderDynamic& VkBlock::getDispatchLoader()
    {
        return m_dispatchLoader;
    }

    uint32_t VkBlock::getMemoryIndex(const vk::PhysicalDevice& physicalDevice,
                                     uint32_t memoryTypeBits,
                                     vk::MemoryPropertyFlags propFlags,
                                     vk::MemoryHeapFlags heapFlags)
    {
        vk::PhysicalDeviceMemoryProperties memProperties;
        physicalDevice.getMemoryProperties(&memProperties, m_dispatchLoader);

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

    vk::DeviceMemory VkBlock::getMemory(VkBlock block)
    {
        return block.m_memory;
    }

    vk::DeviceAddress VkBlock::getDeviceAddress(const vk::Device& device,
                                               VkBlock block,
                                               uint64_t offset)
    {
        auto addrInfo = vk::BufferDeviceAddressInfo().setBuffer(block.m_buffer);

        return device.getBufferAddress(addrInfo, VkBlock::getDispatchLoader()) + offset;
    }

    vk::Buffer VkBlock::getBuffer()
    {
        return m_buffer;
    }

    void VkBlock::allocate(Allocator*              allocator,
                           vk::DeviceSize          size,
                           vk::BufferUsageFlags    usageFlags,
                           vk::MemoryPropertyFlags propFlags,
                           vk::MemoryHeapFlags     heapflags)
    {
        auto bufferInfo = vk::BufferCreateInfo()
            .setSize(size)
            .setUsage(usageFlags)
            .setSharingMode(vk::SharingMode::eExclusive);

        m_buffer = allocator->device.createBuffer(bufferInfo, nullptr, VkBlock::getDispatchLoader());

        vk::MemoryRequirements memoryRequirements = allocator->device.getBufferMemoryRequirements(m_buffer, m_dispatchLoader);
        uint32_t memoryTypeIndex = getMemoryIndex(allocator->physicalDevice, memoryRequirements.memoryTypeBits, propFlags, heapflags);

        auto memoryInfo = vk::MemoryAllocateInfo()
            .setAllocationSize(size)
            .setMemoryTypeIndex(memoryTypeIndex);

        m_memory = allocator->device.allocateMemory(memoryInfo, nullptr, VkBlock::getDispatchLoader());
        allocator->device.bindBufferMemory(m_buffer, m_memory, 0, VkBlock::getDispatchLoader());
    }

    void VkBlock::free(Allocator* allocator)
    {
        allocator->device.destroyBuffer(m_buffer, nullptr, VkBlock::getDispatchLoader());
        allocator->device.freeMemory(m_memory, nullptr, VkBlock::getDispatchLoader());
    }

    uint64_t VkBlock::getVMA() { return (uint64_t)(VkDeviceMemory)(m_memory); }
}
