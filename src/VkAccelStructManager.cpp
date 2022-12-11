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

#include <rtxmu/VkAccelStructManager.h>

namespace rtxmu
{
    VkAccelStructManager::VkAccelStructManager(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice)
    {
        m_allocator.instance = instance;
        m_allocator.device = device;
        m_allocator.physicalDevice = physicalDevice;
    }
    
    // Initializes suballocator block size
    void VkAccelStructManager::Initialize(uint32_t suballocatorBlockSize)
    {
        m_suballocationBlockSize = suballocatorBlockSize;

        m_scratchPool = std::make_unique<Suballocator<Allocator, VkScratchBlock>>(m_suballocationBlockSize, AccelStructAlignment, &m_allocator);
        m_updatePool = std::make_unique<Suballocator<Allocator, VkScratchBlock>>(m_suballocationBlockSize, AccelStructAlignment, &m_allocator);
        m_resultPool = std::make_unique<Suballocator<Allocator, VkAccelStructBlock>>(m_suballocationBlockSize, AccelStructAlignment, &m_allocator);
        m_compactionPool = std::make_unique<Suballocator<Allocator, VkAccelStructBlock>>(m_suballocationBlockSize, AccelStructAlignment, &m_allocator);
        m_queryCompactionSizePool = std::make_unique<Suballocator<Allocator, VkQueryBlock>>(CompactionSizeSuballocationBlockSize, SizeOfCompactionDescriptor, &m_allocator);

        // Load dispatch table if not loaded
        if (VkBlock::getDispatchLoader().vkGetInstanceProcAddr == nullptr)
        {
            const vk::DynamicLoader dl;
            VkBlock::getDispatchLoader().init(m_allocator.instance, m_allocator.device, dl);
        }
    }

    // Resets all queues and frees all memory in suballocators
    void VkAccelStructManager::Reset()
    {
        m_scratchPool.reset();
        m_updatePool.reset();
        m_resultPool.reset();
        m_compactionPool.reset();
        m_queryCompactionSizePool.reset();
        Initialize(m_suballocationBlockSize);
        AccelStructManager::Reset();
    }

    void VkAccelStructManager::PopulateUpdateCommandList(vk::CommandBuffer                                  commandList,
                                                         vk::AccelerationStructureBuildGeometryInfoKHR*     geomInfos,
                                                         const vk::AccelerationStructureBuildRangeInfoKHR** rangeInfos,
                                                         const uint32_t                                     buildCount,
                                                         std::vector<uint64_t>&                             accelStructIds)
    {
        m_threadSafeLock.lock();

        for (uint32_t buildIndex = 0; buildIndex < buildCount; buildIndex++)
        {
            const uint64_t asId = accelStructIds[buildIndex];
            auto& geomInfo = geomInfos[buildIndex];

            VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[asId];

            if (geomInfo.flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate &&
                geomInfo.mode == vk::BuildAccelerationStructureModeKHR::eUpdate)
            {
                geomInfo.scratchData.deviceAddress = VkBlock::getDeviceAddress(m_allocator.device, accelStruct->updateGpuMemory.block, accelStruct->updateGpuMemory.offset);

                geomInfo.dstAccelerationStructure = GetAccelerationStruct(asId);
                geomInfo.srcAccelerationStructure = GetAccelerationStruct(asId);
            }
            else
            {
                // Do not support rebuilds with compaction, not good practice
                assert((geomInfo.flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction) == vk::BuildAccelerationStructureFlagBitsKHR(0));

                // All scratch is discarded after the build is performed but if a recurring build happens
                // then we need to reallocate
                if ((accelStruct->scratchGpuMemory.subBlock == nullptr) ||
                    (accelStruct->scratchGpuMemory.subBlock->isFree() == true))
                {
                    accelStruct->scratchGpuMemory =
                        m_scratchPool->allocate(accelStruct->scratchGpuMemory.subBlock->getSize());
                }

                geomInfo.scratchData.deviceAddress = VkBlock::getDeviceAddress(m_allocator.device, accelStruct->scratchGpuMemory.block, accelStruct->scratchGpuMemory.offset);
                geomInfo.dstAccelerationStructure = accelStruct->resultGpuMemory.block.m_asHandle;
            }

            commandList.buildAccelerationStructuresKHR(1, &geomInfo, &rangeInfos[buildIndex], VkBlock::getDispatchLoader());
        }
        m_threadSafeLock.unlock();
    }

    // Receives acceleration structure inputs and returns a command list with build commands
    void VkAccelStructManager::PopulateBuildCommandList(vk::CommandBuffer                                  commandList,
                                                        vk::AccelerationStructureBuildGeometryInfoKHR*     geomInfos,
                                                        const vk::AccelerationStructureBuildRangeInfoKHR** rangeInfos,
                                                        const uint32_t**                                   maxPrimitiveCounts,
                                                        const uint32_t                                     buildCount,
                                                        std::vector<uint64_t>&                             accelStructIds)
    {
        m_threadSafeLock.lock();

        accelStructIds.reserve(buildCount);
        for (uint32_t buildIndex = 0; buildIndex < buildCount; buildIndex++)
        {
            uint64_t asId = GetAccelStructId();
            
            VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[asId];

            // Assign an id for the acceleration structure
            accelStructIds.push_back(asId);

            auto buildSizeInfo = vk::AccelerationStructureBuildSizesInfoKHR();
            m_allocator.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, &geomInfos[buildIndex], maxPrimitiveCounts[buildIndex], &buildSizeInfo, VkBlock::getDispatchLoader());

            accelStruct->resultGpuMemory = m_resultPool->allocate(buildSizeInfo.accelerationStructureSize);
            accelStruct->scratchGpuMemory = m_scratchPool->allocate(buildSizeInfo.buildScratchSize);
            accelStruct->scratchSize = accelStruct->scratchGpuMemory.subBlock->getSize();
            m_totalUncompactedMemory += accelStruct->resultGpuMemory.subBlock->getSize();
            accelStruct->resultSize = accelStruct->resultGpuMemory.subBlock->getSize();

            auto asCreateInfo = vk::AccelerationStructureCreateInfoKHR()
                .setType(geomInfos->type)
                .setSize(buildSizeInfo.accelerationStructureSize)
                .setBuffer(accelStruct->resultGpuMemory.block.getBuffer())
                .setOffset(accelStruct->resultGpuMemory.offset);

            auto asHandle = m_allocator.device.createAccelerationStructureKHR(asCreateInfo, nullptr, VkBlock::getDispatchLoader());
            accelStruct->resultGpuMemory.block.m_asHandle = asHandle;

            if (geomInfos[buildIndex].flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate)
            {
                accelStruct->updateGpuMemory = m_updatePool->allocate(buildSizeInfo.updateScratchSize);
            }

            geomInfos[buildIndex].scratchData.deviceAddress = VkBlock::getDeviceAddress(m_allocator.device, accelStruct->scratchGpuMemory.block,
                accelStruct->scratchGpuMemory.offset);
            geomInfos[buildIndex].dstAccelerationStructure = asHandle;

            //Can batch in one call, need to rehandle compact size queries
            commandList.buildAccelerationStructuresKHR(1, &geomInfos[buildIndex], &rangeInfos[buildIndex], VkBlock::getDispatchLoader());

            if (geomInfos[buildIndex].flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction)
            {
                accelStruct->isCompacted = false;
                accelStruct->requestedCompaction = true;

                accelStruct->queryCompactionSizeMemory = m_queryCompactionSizePool->allocate(SizeOfCompactionDescriptor);
            }
        }
        m_threadSafeLock.unlock();
    }
    // Receives acceleration structure inputs and places UAV barriers for them
    void VkAccelStructManager::PopulateUAVBarriersCommandList(vk::CommandBuffer commandList,
                                                              const std::vector<uint64_t>& accelStructIds)
    {
        for (const uint64_t& asId : accelStructIds)
        {
            // Barrier for compaction size query
            auto barrier = vk::BufferMemoryBarrier()
                .setSrcAccessMask(vk::AccessFlagBits::eAccelerationStructureWriteKHR)
                .setDstAccessMask(vk::AccessFlagBits::eAccelerationStructureReadKHR)
                .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .setBuffer(m_asBufferBuildQueue[asId]->isCompacted ?
                    m_asBufferBuildQueue[asId]->compactionGpuMemory.block.getBuffer() :
                    m_asBufferBuildQueue[asId]->resultGpuMemory.block.getBuffer())
                .setOffset(m_asBufferBuildQueue[asId]->isCompacted ?
                    m_asBufferBuildQueue[asId]->compactionGpuMemory.offset :
                    m_asBufferBuildQueue[asId]->resultGpuMemory.offset)
                .setSize(m_asBufferBuildQueue[asId]->resultGpuMemory.subBlock->getSize());

            commandList.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                vk::DependencyFlags(), 0, nullptr, 1, &barrier, 0, nullptr, VkBlock::getDispatchLoader());
        }
    }

    // Performs copies to bring over any compaction size data
    void VkAccelStructManager::PopulateCompactionSizeCopiesCommandList(vk::CommandBuffer commandList,
                                                                       const std::vector<uint64_t>& accelStructIds)
    {
        for (const uint64_t& asId : accelStructIds)
        {
            VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[asId];

            if (accelStruct->requestedCompaction == true &&
                accelStruct->isCompacted == false)
            {
                vk::QueryPool pool = accelStruct->queryCompactionSizeMemory.block.queryPool;
                uint32_t queryIndex = (uint32_t)accelStruct->queryCompactionSizeMemory.offset / SizeOfCompactionDescriptor;
                vk::AccelerationStructureKHR asHandle = accelStruct->resultGpuMemory.block.m_asHandle;

                // Need to batch builds/sychornization/compaction size writes
                commandList.resetQueryPool(pool, queryIndex, 1, VkBlock::getDispatchLoader());
                commandList.writeAccelerationStructuresPropertiesKHR(1, &asHandle, vk::QueryType::eAccelerationStructureCompactedSizeKHR, pool, queryIndex, VkBlock::getDispatchLoader());
            }
        }
    }

    // Returns a command list with compaction copies if the acceleration structures are ready to be compacted
    void VkAccelStructManager::PopulateCompactionCommandList(vk::CommandBuffer commandList,
                                                             const std::vector<uint64_t>& accelStructIds)
    {
        m_threadSafeLock.lock();
        bool compactionCopiesPerformed = false;
        for (const uint64_t& accelStructId : accelStructIds)
        {
            VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

            // Only do compaction on the confirmed completion of the original build execution.
            if (accelStruct->requestedCompaction == true &&
                accelStruct->isCompacted == false)
            {
                uint32_t queryIndex = (uint32_t)accelStruct->queryCompactionSizeMemory.offset / SizeOfCompactionDescriptor;
                vk::QueryPool& pool = accelStruct->queryCompactionSizeMemory.block.queryPool;

                vk::DeviceSize compactionSize;
                auto result = m_allocator.device.getQueryPoolResults(pool, queryIndex, 1, (size_t)sizeof(vk::DeviceSize), (void*)&compactionSize,
                    (vk::DeviceSize)SizeOfCompactionDescriptor, vk::QueryResultFlagBits::eWait | vk::QueryResultFlagBits::e64, VkBlock::getDispatchLoader());
                (void)result;

                accelStruct->compactionGpuMemory = m_compactionPool->allocate(compactionSize);
                accelStruct->compactionSize = accelStruct->compactionGpuMemory.subBlock->getSize();
                m_totalCompactedMemory += accelStruct->compactionGpuMemory.subBlock->getSize();

                auto asCreateInfo = vk::AccelerationStructureCreateInfoKHR()
                    .setType(vk::AccelerationStructureTypeKHR::eBottomLevel)
                    .setSize(compactionSize)
                    .setBuffer(accelStruct->compactionGpuMemory.block.getBuffer())
                    .setOffset(accelStruct->compactionGpuMemory.offset);
                auto asHandle = m_allocator.device.createAccelerationStructureKHR(asCreateInfo, nullptr, VkBlock::getDispatchLoader());
                accelStruct->compactionGpuMemory.block.m_asHandle = asHandle;

                auto copyInfo = vk::CopyAccelerationStructureInfoKHR()
                    .setMode(vk::CopyAccelerationStructureModeKHR::eCompact)
                    .setSrc(accelStruct->resultGpuMemory.block.m_asHandle)
                    .setDst(accelStruct->compactionGpuMemory.block.m_asHandle);
                commandList.copyAccelerationStructureKHR(copyInfo, VkBlock::getDispatchLoader());

                accelStruct->isCompacted = true;
                compactionCopiesPerformed = true;
            }
        }
        if (compactionCopiesPerformed)
        {
            for (const uint64_t& accelStructId : accelStructIds)
            {
                VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

                // Only do compaction on the confirmed completion of the original build execution.
                if (accelStruct->requestedCompaction == true)
                {
                    auto barrier = vk::BufferMemoryBarrier()
                        .setSrcAccessMask(vk::AccessFlagBits::eAccelerationStructureWriteKHR)
                        .setDstAccessMask(vk::AccessFlagBits::eAccelerationStructureReadKHR)
                        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .setBuffer(accelStruct->compactionGpuMemory.block.getBuffer())
                        .setOffset(accelStruct->compactionGpuMemory.offset)
                        .setSize(accelStruct->compactionGpuMemory.subBlock->getSize());

                    commandList.pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                        vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
                        vk::DependencyFlags(), 0, nullptr, 1, &barrier, 0, nullptr, VkBlock::getDispatchLoader());
                }
            }
        }
        m_threadSafeLock.unlock();
    }

    // Remove all memory that an Acceleration Structure might use
    void VkAccelStructManager::RemoveAccelerationStructures(const std::vector<uint64_t>& accelStructIds)
    {
        m_threadSafeLock.lock();
        for (const uint64_t& accelStructId : accelStructIds)
        {
            ReleaseAccelerationStructures(accelStructId);
        }
        m_threadSafeLock.unlock();
    }

    // Remove all memory used in build process, while only leaving the acceleration structure buffer itself in memory
    void VkAccelStructManager::GarbageCollection(const std::vector<uint64_t>& accelStructIds)
    {
        m_threadSafeLock.lock();

        // Complete queue indicates cleanup for acceleration structures
        for (const uint64_t& accelStructId : accelStructIds)
        {
            PostBuildRelease(accelStructId);
            m_asBufferBuildQueue[accelStructId]->readyToFree = true;
        }

        m_threadSafeLock.unlock();
    }

    // Returns GPUVA of the acceleration structure
    vk::DeviceMemory VkAccelStructManager::GetMemory(const uint64_t accelStructId)
    {
        VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        return accelStruct->isCompacted ?
                   VkBlock::getMemory(accelStruct->compactionGpuMemory.block) :
                   VkBlock::getMemory(accelStruct->resultGpuMemory.block);
    }

    vk::DeviceSize VkAccelStructManager::GetMemoryOffset(const uint64_t accelStructId)
    {
        VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        return (VkDeviceSize)(accelStruct->isCompacted ?
                                  accelStruct->compactionGpuMemory.offset :
                                  accelStruct->resultGpuMemory.offset);
    }

    vk::DeviceAddress VkAccelStructManager::GetDeviceAddress(const uint64_t accelStructId)
    {
        VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        return accelStruct->isCompacted ? VkBlock::getDeviceAddress(m_allocator.device,
                                                                    accelStruct->compactionGpuMemory.block,
                                                                    accelStruct->compactionGpuMemory.offset) :
                                          VkBlock::getDeviceAddress(m_allocator.device,
                                                                    accelStruct->resultGpuMemory.block,
                                                                    accelStruct->resultGpuMemory.offset);
    }

    vk::AccelerationStructureKHR VkAccelStructManager::GetAccelerationStruct(const uint64_t accelStructId)
    {
        VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        return accelStruct->isCompacted ?
                   accelStruct->compactionGpuMemory.block.m_asHandle :
                   accelStruct->resultGpuMemory.block.m_asHandle;
    }

    vk::Buffer VkAccelStructManager::GetBuffer(const uint64_t accelStructId)
    {
        VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        return accelStruct->isCompacted ?
                   accelStruct->compactionGpuMemory.block.getBuffer() :
                   accelStruct->resultGpuMemory.block.getBuffer();
    }

    // Returns a const char* containing memory consumption information
    const char* VkAccelStructManager::GetLog()
    {
        // Clear out previous logging
        m_buildLogger.clear();

        double memoryReductionRatio = (static_cast<double>(m_totalCompactedMemory) / (m_totalUncompactedMemory + 1.0));
        double fragmentedRatio = 1.0 - (static_cast<double>(m_totalCompactedMemory) / (m_compactionPool->getSize() + 1.0f));
        m_buildLogger.append(
            "TOTAL Result memory allocated:     " + std::to_string(m_totalUncompactedMemory / 1000000.0f) + " MB\n"
            "TOTAL Compaction memory allocated: " + std::to_string(m_totalCompactedMemory / 1000000.0f) + " MB\n"
            "Compaction memory reduction:       " + std::to_string(memoryReductionRatio * 100.0f) + " %\n"
            "Result suballocator memory:        " + std::to_string(m_resultPool->getSize() / 1000000.0f) + " MB\n"
            "Compaction suballocator memory:    " + std::to_string(m_compactionPool->getSize() / 1000000.0f) + " MB\n"
            "Scratch suballocator memory:       " + std::to_string(m_scratchPool->getSize() / 1000000.0f) + " MB\n"
            "Update suballocator memory:        " + std::to_string(m_updatePool->getSize() / 1000000.0f) + " MB\n"
            "Compaction fragmented percentage:  " + std::to_string(fragmentedRatio * 100.0f) + " %\n"
        );

        return m_buildLogger.c_str();
    }

    void VkAccelStructManager::PostBuildRelease(const uint64_t accelStructId)
    {
        VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        // Only delete compaction size and result if compaction was performed
        if (accelStruct->isCompacted == true)
        {
            // Deallocate all the buffers used to create a compaction AS buffer
            if ((accelStruct->resultGpuMemory.subBlock != nullptr) &&
                (accelStruct->resultGpuMemory.subBlock->isFree() == false))
            {
                m_resultPool->free(accelStruct->resultGpuMemory.subBlock);
            }
            if ((accelStruct->queryCompactionSizeMemory.subBlock != nullptr) &&
                (accelStruct->queryCompactionSizeMemory.subBlock->isFree() == false))
            {
                m_queryCompactionSizePool->free(accelStruct->queryCompactionSizeMemory.subBlock);
            }
            //Destroy the result AccelStruct, because the compaction AccelStruct is used
            auto& resultAS = accelStruct->resultGpuMemory.block.m_asHandle;
            if(resultAS)
            {
                m_allocator.device.destroyAccelerationStructureKHR(resultAS, nullptr, VkBlock::getDispatchLoader());
                resultAS = nullptr;
            }
        }
        
        if ((accelStruct->scratchGpuMemory.subBlock != nullptr) &&
            (accelStruct->scratchGpuMemory.subBlock->isFree() == false))
        {
            m_scratchPool->free(accelStruct->scratchGpuMemory.subBlock);
        }
    }

    void VkAccelStructManager::ReleaseAccelerationStructures(const uint64_t accelStructId)
    {
        VkAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        m_totalCompactedMemory -= accelStruct->compactionSize;
        m_totalUncompactedMemory -= accelStruct->resultSize;

        // Deallocate all the buffers used for acceleration structures
        if ((accelStruct->scratchGpuMemory.subBlock != nullptr) &&
            (accelStruct->scratchGpuMemory.subBlock->isFree() == false))
        {
            m_scratchPool->free(accelStruct->scratchGpuMemory.subBlock);
        }
        if ((accelStruct->updateGpuMemory.subBlock != nullptr) &&
            (accelStruct->updateGpuMemory.subBlock->isFree() == false))
        {
            m_updatePool->free(accelStruct->updateGpuMemory.subBlock);
        }
        if ((accelStruct->resultGpuMemory.subBlock != nullptr) &&
            (accelStruct->resultGpuMemory.subBlock->isFree() == false))
        {
            m_resultPool->free(accelStruct->resultGpuMemory.subBlock);
        }
        if ((accelStruct->compactionGpuMemory.subBlock != nullptr) &&
            (accelStruct->compactionGpuMemory.subBlock->isFree() == false))
        {
            m_compactionPool->free(accelStruct->compactionGpuMemory.subBlock);
        }

        auto&compactionAS = accelStruct->compactionGpuMemory.block.m_asHandle;
        auto& resultAS = accelStruct->resultGpuMemory.block.m_asHandle;
        // Destroy the acceleration structures
        if (accelStruct->isCompacted && compactionAS)
        {
            m_allocator.device.destroyAccelerationStructureKHR(compactionAS, nullptr, VkBlock::getDispatchLoader());
        }
        if (resultAS)
        {
            m_allocator.device.destroyAccelerationStructureKHR(resultAS, nullptr, VkBlock::getDispatchLoader());
        }
        accelStruct->resultGpuMemory.block.m_asHandle = nullptr;
        accelStruct->compactionGpuMemory.block.m_asHandle = nullptr;

        ReleaseAccelStructId( accelStructId);
    }
}