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

#include <rtxmu/D3D12AccelStructManager.h>

namespace rtxmu
{

    DxAccelStructManager::DxAccelStructManager(ID3D12Device5* device)
    {
        D3D12Block::m_device = device;
    }
    
    // Initializes suballocator block size
    void DxAccelStructManager::Initialize(uint32_t suballocatorBlockSize)
    {
        m_suballocationBlockSize = suballocatorBlockSize;

        m_scratchPool = std::make_unique<Suballocator<D3D12ScratchBlock>>(m_suballocationBlockSize, AccelStructAlignment);
        m_resultPool = std::make_unique<Suballocator<D3D12AccelStructBlock>>(m_suballocationBlockSize, AccelStructAlignment);
        m_compactionPool = std::make_unique<Suballocator<D3D12AccelStructBlock>>(m_suballocationBlockSize, AccelStructAlignment);
        m_compactionSizeGpuPool = std::make_unique<Suballocator<D3D12CompactionWriteBlock>>(CompactionSizeSuballocationBlockSize, SizeOfCompactionDescriptor);
        m_compactionSizeCpuPool = std::make_unique<Suballocator<D3D12ReadBackBlock>>(CompactionSizeSuballocationBlockSize, SizeOfCompactionDescriptor);
    }

    // Resets all queues and frees all memory in suballocators
    void DxAccelStructManager::Reset()
    {
        m_scratchPool.reset();
        m_resultPool.reset();
        m_compactionPool.reset();
        m_compactionSizeGpuPool.reset();
        m_compactionSizeCpuPool.reset();
        Initialize(m_suballocationBlockSize);
        AccelStructManager::Reset();
    }

    // Receives acceleration structure inputs and returns a command list with build commands
    void DxAccelStructManager::PopulateUpdateCommandList(ID3D12GraphicsCommandList4*                                 commandList,
                                                         const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* asInputs,
                                                         const uint32_t                                              buildCount,
                                                         const std::vector<uint64_t>&                                accelStructIds)
    {
        m_threadSafeLock.lock();
        for (uint32_t buildIndex = 0; buildIndex < buildCount; buildIndex++)
        {
            const uint64_t accelStructId = accelStructIds[buildIndex];

            if ((asInputs[buildIndex].Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE) &&
                (asInputs[buildIndex].Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE))
            {
                // Setup build desc and allocator scratch and result buffers
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
                buildDesc.Inputs = asInputs[buildIndex];
                buildDesc.ScratchAccelerationStructureData = D3D12Block::getGPUVA(m_asBufferBuildQueue[accelStructId]->updateGpuMemory.block,
                                                                                  m_asBufferBuildQueue[accelStructId]->updateGpuMemory.offset);
                buildDesc.DestAccelerationStructureData   = GetAccelStructGPUVA(accelStructId);
                buildDesc.SourceAccelerationStructureData = GetAccelStructGPUVA(accelStructId);

                commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
            }
            else
            {
                // Do not support rebuilds with compaction, not good practice
                assert((asInputs[buildIndex].Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION) == 0);

                // Setup build desc and allocator scratch and result buffers
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
                buildDesc.Inputs                                             = asInputs[buildIndex];
                buildDesc.ScratchAccelerationStructureData = D3D12Block::getGPUVA(m_asBufferBuildQueue[accelStructId]->scratchGpuMemory.block,
                                                                                  m_asBufferBuildQueue[accelStructId]->scratchGpuMemory.offset);
                buildDesc.DestAccelerationStructureData    = D3D12Block::getGPUVA(m_asBufferBuildQueue[accelStructId]->resultGpuMemory.block,
                                                                                  m_asBufferBuildQueue[accelStructId]->resultGpuMemory.offset);

                commandList->BuildRaytracingAccelerationStructure(&buildDesc, 0, nullptr);
            }
        }
        m_threadSafeLock.unlock();
    }

    // Receives acceleration structure inputs and returns a command list with build commands
    void DxAccelStructManager::PopulateBuildCommandList(ID3D12GraphicsCommandList4*                                 commandList,
                                                        const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* asInputs,
                                                        const uint64_t                                              buildCount,
                                                        std::vector<uint64_t>&                                      accelStructIds)
    {
        m_threadSafeLock.lock();

        accelStructIds.reserve(buildCount);
        bool compactionRequested = false;
        DxAccelerationStructure* lastBuild = nullptr;
        for (uint32_t buildIndex = 0; buildIndex < buildCount; buildIndex++)
        {
            uint64_t asId = GetAccelStructId();
               
            // Assign an id for the acceleration structure
            accelStructIds.push_back(asId);

            // Request build size information and suballocate the scratch and result buffers
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuildInfo = {};
            D3D12Block::m_device->GetRaytracingAccelerationStructurePrebuildInfo(&asInputs[buildIndex], &prebuildInfo);

            DxAccelerationStructure* accelStruct = m_asBufferBuildQueue[asId];

            accelStruct->resultGpuMemory = m_resultPool->allocate(prebuildInfo.ResultDataMaxSizeInBytes);

            if (asInputs[buildIndex].Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE)
            {
                accelStruct->updateGpuMemory = m_scratchPool->allocate(prebuildInfo.UpdateScratchDataSizeInBytes);
            }

            accelStruct->scratchGpuMemory = m_scratchPool->allocate(prebuildInfo.ScratchDataSizeInBytes);

            m_totalUncompactedMemory += accelStruct->resultGpuMemory.subBlock->getSize();
            accelStruct->resultSize = accelStruct->resultGpuMemory.subBlock->getSize();

            // Setup build desc and allocator scratch and result buffers
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildDesc = {};
            buildDesc.Inputs                                             = asInputs[buildIndex];
            buildDesc.ScratchAccelerationStructureData = D3D12Block::getGPUVA(accelStruct->scratchGpuMemory.block,
                                                                              accelStruct->scratchGpuMemory.offset);
            buildDesc.DestAccelerationStructureData    = D3D12Block::getGPUVA(accelStruct->resultGpuMemory.block,
                                                                              accelStruct->resultGpuMemory.offset);

            // Only perform compaction of the build inputs that include compaction
            if (asInputs[buildIndex].Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION)
            {
                // Tag as not yet compacted
                accelStruct->isCompacted = false;
                accelStruct->requestedCompaction = true;

                // Suballocate the gpu memory that the builder will use to write the compaction size post build
                accelStruct->compactionSizeGpuMemory = m_compactionSizeGpuPool->allocate(SizeOfCompactionDescriptor);

                // Request to get compaction size post build
                auto gpuVA = D3D12Block::getGPUVA(accelStruct->compactionSizeGpuMemory.block,
                                                  accelStruct->compactionSizeGpuMemory.offset);

                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC postBuildInfo [] = 
                { gpuVA, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE };

                commandList->BuildRaytracingAccelerationStructure(&buildDesc,
                                                                  sizeof(postBuildInfo) / sizeof(postBuildInfo[0]),
                                                                  postBuildInfo);

                // Suballocate the readback memory
                accelStruct->compactionSizeCpuMemory = m_compactionSizeCpuPool->allocate(SizeOfCompactionDescriptor);

                compactionRequested = true;
            }
            else
            {
                // This build doesn't request compaction
                accelStruct->isCompacted = false;
                accelStruct->requestedCompaction = false;
                commandList->BuildRaytracingAccelerationStructure(&buildDesc,
                                                                  0,
                                                                  nullptr);
            }

            // Use the last build in the sequence for the barrier
            lastBuild = accelStruct;
        }
        m_threadSafeLock.unlock();
    }

    // Compaction size copies
    void DxAccelStructManager::PopulateCompactionSizeCopiesCommandList(ID3D12GraphicsCommandList4* commandList,
                                                                       const std::vector<uint64_t>& accelStructIds)
    {
        (void)accelStructIds;

        auto gpuSizeBlocks = m_compactionSizeGpuPool->getBlocks();
        auto cpuSizeBlocks = m_compactionSizeCpuPool->getBlocks();
        for (int i = 0; i < gpuSizeBlocks.size(); i++)
        {
            auto gpuSizeBlock = gpuSizeBlocks[i];
            auto cpuSizeBlock = cpuSizeBlocks[i];

            // Transition the gpu compaction size suballocator block to copy over to mappable cpu memory
            D3D12_RESOURCE_BARRIER rb = {};
            rb.Transition.pResource = gpuSizeBlock->block.m_resource;
            rb.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            rb.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;

            commandList->ResourceBarrier(1, &rb);

            // Copy the entire resource to avoid individually copying over each compaction size in strides of 8 bytes
            commandList->CopyResource(cpuSizeBlock->block.m_resource,
                gpuSizeBlock->block.m_resource);

            // Transition the gpu written compaction size suballocator block back over to unordered for later use
            rb = {};
            rb.Transition.pResource = gpuSizeBlock->block.m_resource;
            rb.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            rb.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            rb.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;

            commandList->ResourceBarrier(1, &rb);
        }
    }

    // Receives acceleration structure inputs and places UAV barriers for them
    void DxAccelStructManager::PopulateUAVBarriersCommandList(ID3D12GraphicsCommandList4*  commandList,
                                                              const std::vector<uint64_t>& accelStructIds)
    {
        for (uint64_t accelStructId : accelStructIds)
        {
            D3D12_RESOURCE_BARRIER rb = {};
            rb.UAV.pResource = m_asBufferBuildQueue[accelStructId]->resultGpuMemory.block.m_resource;
            rb.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            commandList->ResourceBarrier(1, &rb);
        }
    }

    // Returns a command list with compaction copies if the acceleration structures are ready to be compacted
    void DxAccelStructManager::PopulateCompactionCommandList(ID3D12GraphicsCommandList4*  commandList,
                                                             const std::vector<uint64_t>& accelStructIds)
    {
        m_threadSafeLock.lock();

        // Keep track of last compacted resource to include barrier if the
        // app requires a subsequent TLAS build or other read operation of the compacted version
        ID3D12Resource* compactionResourceBarrier = nullptr;

        for (const uint64_t& accelStructId : accelStructIds)
        {
            // Only do compaction on the confirmed completion of the original build execution.
            if (m_asBufferBuildQueue[accelStructId]->requestedCompaction == true)
            {
                CopyCompaction(commandList, accelStructId);

                compactionResourceBarrier = m_asBufferBuildQueue[accelStructId]->compactionGpuMemory.block.m_resource;
            }
        }

        // Include resource barrier after final compaction
        if (compactionResourceBarrier != nullptr)
        {
            D3D12_RESOURCE_BARRIER rb = {};
            rb.UAV.pResource          = compactionResourceBarrier;
            rb.Type                   = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            commandList->ResourceBarrier(1, &rb);
        }

        m_threadSafeLock.unlock();
    }

    // Remove all memory that an Acceleration Structure might use
    void DxAccelStructManager::RemoveAccelerationStructures(const std::vector<uint64_t>& accelStructIds)
    {
        m_threadSafeLock.lock();
        for (const uint64_t& accelStructId : accelStructIds)
        {
            ReleaseAccelerationStructures(accelStructId);
        }
        m_threadSafeLock.unlock();
    }

    // Remove all memory used in build process, while only leaving the acceleration structure buffer itself in memory
    void DxAccelStructManager::GarbageCollection(const std::vector<uint64_t>& accelStructIds)
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
    D3D12_GPU_VIRTUAL_ADDRESS DxAccelStructManager::GetAccelStructGPUVA(const uint64_t accelStructId)
    {
        DxAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        return accelStruct->isCompacted ?
                   D3D12Block::getGPUVA(accelStruct->compactionGpuMemory.block,
                                        accelStruct->compactionGpuMemory.offset) :
                   D3D12Block::getGPUVA(accelStruct->resultGpuMemory.block,
                                        accelStruct->resultGpuMemory.offset);
    }

    // Returns a "string" containing memory consumption information
    const char* DxAccelStructManager::GetLog()
    {
        //Clear out previous command lists compaction logging
        m_buildLogger.clear();

        double memoryReductionRatio = (static_cast<double>(m_totalCompactedMemory) / (m_totalUncompactedMemory + 1.0));
        double fragmentedRatio = 1.0 - (static_cast<double>(m_totalCompactedMemory) / (m_compactionPool->getSize() + 1.0f));
        m_buildLogger.append(
            "Theoretical uncompacted  memory:     " + std::to_string(m_totalUncompactedMemory    / 1000000.0f) + " MB\n"
            "Compaction               memory:     " + std::to_string(m_totalCompactedMemory      / 1000000.0f) + " MB\n"
            "Compaction  memory       reduction:  " + std::to_string(memoryReductionRatio        * 100.0f)     + " %%\n"
            "Uncompacted suballocator memory:     " + std::to_string(m_resultPool->getSize()     / 1000000.0f) + " MB\n"
            "Compaction  suballocator memory:     " + std::to_string(m_compactionPool->getSize() / 1000000.0f) + " MB\n"
            "Scratch     suballocator memory:     " + std::to_string(m_scratchPool->getSize()    / 1000000.0f) + " MB\n"
            "Compaction  fragmented   percentage: " + std::to_string(fragmentedRatio             * 100.0f)     + " %%\n"
        );

        return m_buildLogger.c_str();
    }

    void DxAccelStructManager::CopyCompaction(ID3D12GraphicsCommandList4* commandList,
                                              const uint64_t              accelStructId)
    {
        DxAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        // Don't compact if not requested or already complete
        if ((accelStruct->isCompacted         == false) &&
            (accelStruct->requestedCompaction == true))
        {
            unsigned char* data           = nullptr;
            uint64_t       compactionSize = 0;
            uint64_t       offset         = accelStruct->compactionSizeCpuMemory.offset;

            // Map the readback gpu memory to system memory to fetch compaction size
            D3D12_RANGE readbackBufferRange{offset, offset + SizeOfCompactionDescriptor};
            accelStruct->compactionSizeCpuMemory.block.m_resource->Map(0, &readbackBufferRange, (void**)&data);
            memcpy(&compactionSize, &data[offset], SizeOfCompactionDescriptor);

            // Suballocate the gpu memory needed for compaction copy
            accelStruct->compactionGpuMemory = m_compactionPool->allocate(compactionSize);

            accelStruct->compactionSize = accelStruct->compactionGpuMemory.subBlock->getSize();
            m_totalCompactedMemory += accelStruct->compactionGpuMemory.subBlock->getSize();

            // Copy the result buffer into the compacted buffer
            commandList->CopyRaytracingAccelerationStructure(D3D12Block::getGPUVA(accelStruct->compactionGpuMemory.block, accelStruct->compactionGpuMemory.offset),
                                                                D3D12Block::getGPUVA(accelStruct->resultGpuMemory.block, accelStruct->resultGpuMemory.offset),
                                                                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);

            // Tag as compaction complete
            accelStruct->isCompacted = true;
        }
    }

    void DxAccelStructManager::PostBuildRelease(const uint64_t accelStructId)
    {
        DxAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        // Only delete compaction size, result and scratch buffers if compaction was done
        // The assumption is that only static acceleration structures will be compacted
        if (accelStruct->isCompacted == true)
        {
            // Deallocate all the buffers used to create a compaction AS buffer
            if ((accelStruct->resultGpuMemory.subBlock != nullptr) &&
                (accelStruct->resultGpuMemory.subBlock->isFree() == false))
            {
                m_resultPool->free(accelStruct->resultGpuMemory.subBlock);
            }
            if ((accelStruct->compactionSizeGpuMemory.subBlock != nullptr) &&
                (accelStruct->compactionSizeGpuMemory.subBlock->isFree() == false))
            {
                m_compactionSizeGpuPool->free(accelStruct->compactionSizeGpuMemory.subBlock);
            }
            if ((accelStruct->compactionSizeCpuMemory.subBlock != nullptr) &&
                (accelStruct->compactionSizeCpuMemory.subBlock->isFree() == false))
            {
                m_compactionSizeCpuPool->free(accelStruct->compactionSizeCpuMemory.subBlock);
            }
            if ((accelStruct->scratchGpuMemory.subBlock != nullptr) &&
                (accelStruct->scratchGpuMemory.subBlock->isFree() == false))
            {
                m_scratchPool->free(accelStruct->scratchGpuMemory.subBlock);
            }
        }
    }

    void DxAccelStructManager::ReleaseAccelerationStructures(const uint64_t accelStructId)
    {
        DxAccelerationStructure* accelStruct = m_asBufferBuildQueue[accelStructId];

        m_totalCompactedMemory -= accelStruct->compactionSize;
        m_totalUncompactedMemory -= accelStruct->resultSize;

        // Deallocate all the buffers used for acceleration structures
        if ((accelStruct->scratchGpuMemory.subBlock != nullptr) &&
            (accelStruct->scratchGpuMemory.subBlock->isFree() == false))
        {
            m_scratchPool->free(accelStruct->scratchGpuMemory.subBlock);
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

        ReleaseAccelStructId(accelStructId);
    }
}