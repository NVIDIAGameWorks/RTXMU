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

#include <vector>
#include <mutex>
#include "Logger.h"
#include <cmath>
#include <cinttypes>

namespace rtxmu
{
    // All sizes are expressed in bytes
    struct Stats
    {
        uint64_t alignmentSavings = 0;
        uint64_t totalResidentMemorySize = 0;
        uint64_t unusedSize = 0;
        double   fragmentation = 0.0;
    };

    // Block type default implementation to force client to implement
    template<typename AllocatorType, typename Block>
    class Suballocator
    {
        // Forward decls
        struct BlockDesc;
        struct SubBlock;

    public:

        // Empty base struct that abstracts away the definition of the derived SubBlock struct
        // which contains information hidden from the client
        struct SubBlockRef
        {
            uint64_t getSize()  
            {
                SubBlock* subBlock = reinterpret_cast<SubBlock*>(this);
                return subBlock->size;
            }

            uint64_t getUnusedSize()
            {
                SubBlock* subBlock = reinterpret_cast<SubBlock*>(this);
                return subBlock->unusedSize;
            }

            bool isFree()
            {
                SubBlock* subBlock = reinterpret_cast<SubBlock*>(this);
                return subBlock->isFree;
            }
        };

        // Contains the memory block, an offset and an opaque reference to a SubBlock
        struct SubAllocation
        {
            Block block;
            uint64_t offset = 0;
            SubBlockRef* subBlock = nullptr;
        };

        Suballocator(uint64_t blockSize,
                     uint64_t allocationAlignment,
                     AllocatorType* allocator)
        {
            m_blockSize = blockSize;
            m_allocationAlignment = allocationAlignment;

            Block::setAllocator(allocator);
        }

        virtual ~Suballocator()
        {
            std::lock_guard<std::mutex> guard(m_threadSafeLock);

            size_t blockCount = m_blocks.size();

            for (uint32_t blockIndex = 0; blockIndex < blockCount; blockIndex++)
            {
                m_blocks[blockIndex]->block.free();
            }
            m_blocks.clear();
        }
        SubAllocation allocate(uint64_t unalignedSize)
        {
            std::lock_guard<std::mutex> guard(m_threadSafeLock);

            // Align allocation
            const uint64_t sizeInBytes = align(unalignedSize, m_allocationAlignment);

            // If no previous blocks exist and if the suballocation size isn't larger than block size
            if (m_blocks.size() == 0 && sizeInBytes <= m_blockSize)
            {
                uint64_t newBlockSize = sizeInBytes > m_blockSize ? sizeInBytes : m_blockSize;
                createBlock(newBlockSize);
            }

            uint64_t numBlocks = m_blocks.size();

            BlockDesc blockDesc = {};
            SubBlock* subBlock = new SubBlock{ SubBlockRef(), 0, 0, 0, 0 };

            // Do not suballocate if the memory request is larger than the block size
            if (sizeInBytes > m_blockSize)
            {
                createBlock(sizeInBytes);
                BlockDesc* block     = m_blocks[m_blocks.size() - 1];
                subBlock->blockDesc  = block;
                subBlock->size       = sizeInBytes;
                subBlock->offset     = block->currentOffset;

                // Capture alignment padding waste
                subBlock->unusedSize = sizeInBytes - unalignedSize;

                const uint64_t offsetInBytes = block->currentOffset + sizeInBytes;
                block->currentOffset         = offsetInBytes;
                block->numSubBlocks++;

                if (Logger::isEnabled(Level::DBG))
                {
                    Logger::log(Level::DBG, "RTXMU Allocation Too Large and Can't Suballocate\n");
                }
            }
            else
            {
                for (uint32_t blockIndex = 0; blockIndex < numBlocks; blockIndex++)
                {
                    BlockDesc* block = m_blocks[blockIndex];

                    // Search within a block to find space for a new allocation
                    // Modifies subBlock if able to suballocate in the block
                    bool foundFreeSubBlock = findFreeSubBlock(block, subBlock, sizeInBytes);

                    bool continueBlockSearch = false;

                    // No memory reuse opportunities available so add a new suballocation
                    if (foundFreeSubBlock == false)
                    {
                        const uint64_t offsetInBytes = block->currentOffset + sizeInBytes;

                        // Add a suballocation to the current offset of an existing block
                        if (offsetInBytes <= block->size)
                        {
                            // Only ever change the memory size if this is a new allocation
                            subBlock->size = sizeInBytes;
                            subBlock->offset = block->currentOffset;
                            // Capture alignment padding waste
                            subBlock->unusedSize = sizeInBytes - unalignedSize;

                            const uint64_t memoryAlignedSize = align(subBlock->size, m_blocks[blockIndex]->block.getAlignment());
                            m_stats.alignmentSavings += (memoryAlignedSize - subBlock->size);

                            block->currentOffset = offsetInBytes;
                            block->numSubBlocks++;
                        }
                        // If this block can't support this allocation
                        else
                        {
                            // If all blocks traversed and suballocation doesn't fit then create a new block
                            if (blockIndex == numBlocks - 1)
                            {
                                // If suballocation block size is too small then do custom allocation of
                                // individual blocks that match the resource's size
                                uint64_t newBlockSize = sizeInBytes > m_blockSize ? sizeInBytes : m_blockSize;
                                createBlock(newBlockSize);
                                numBlocks++;
                            }
                            continueBlockSearch = true;
                        }
                    }
                    // Assign SubBlock to the Block and discontinue suballocation search
                    if (continueBlockSearch == false)
                    {
                        subBlock->blockDesc = m_blocks[blockIndex];
                        break;
                    }
                }
            }

            // Pass a generic SubAllocation struct back to client
            return {subBlock->blockDesc->block, subBlock->offset, subBlock};
        }

        void free(SubBlockRef* subBlockRef)
        {
            std::lock_guard<std::mutex> guard(m_threadSafeLock);

            SubBlock* subBlock = reinterpret_cast<SubBlock*>(subBlockRef);
            uint64_t blockIndex = 0;
            for (BlockDesc* blockDesc : m_blocks)
            {
                if (blockDesc->block.getVMA() == subBlock->blockDesc->block.getVMA())
                {
                    const uint64_t memoryAlignedSize = align(subBlock->size, blockDesc->block.getAlignment());

                    subBlock->isFree = true;

                    // Release the big chunks that are a single resource
                    if (subBlock->size == blockDesc->size)
                    {
                        blockDesc->block.free();
                        m_blocks.erase(m_blocks.begin() + blockIndex);

                        if (Logger::isEnabled(Level::DBG))
                        {
                            Logger::log(Level::DBG, "RTXMU Deallocation of oversized block\n");
                        }
                    }
                    else
                    {
                        struct SubBlock freeSubBlock = { SubBlockRef(),
                                                         subBlock->blockDesc,
                                                         subBlock->offset,
                                                         subBlock->size };

                        blockDesc->freeSubBlocks.push_back(freeSubBlock);

                        m_stats.alignmentSavings -= (memoryAlignedSize - subBlock->size);

                        blockDesc->numSubBlocks--;

                        // If this suballocation was the final remaining allocation then release the suballocator block
                        // but only if there is more than one block
                        if ((blockDesc->numSubBlocks == 0) &&
                            (m_blocks.size() > 1))
                        {
                            blockDesc->block.free();
                            m_blocks.erase(m_blocks.begin() + blockIndex);
                        }
                    }
                    break;
                }
                blockIndex++;
            }
        }

        uint64_t getSize()
        {
            uint64_t size = 0;
            for (auto blockDesc : m_blocks)
            {
                size += blockDesc->size;
            }
            return size;
        }

        //https://asawicki.info/news_1757_a_metric_for_memory_fragmentation
        double getFragmentation(uint64_t& totalUnusedMemory)
        {
            uint64_t quality = 0;
            totalUnusedMemory = 0;
            for (auto blockDesc : m_blocks)
            {
                for (SubBlock& freeSubBlock : blockDesc->freeSubBlocks)
                {
                    quality       += freeSubBlock.size * freeSubBlock.size;
                    totalUnusedMemory += freeSubBlock.size;
                }

                // Current offset free block at the tail end
                uint64_t tailFreeSubBlock = blockDesc->size - blockDesc->currentOffset;

                quality += tailFreeSubBlock * tailFreeSubBlock;
                totalUnusedMemory += tailFreeSubBlock;
            }

            if (quality == 0 || totalUnusedMemory == 0)
            {
                return 0.0;
            }

            double qualityPercent = sqrt(static_cast<double>(quality)) / static_cast<double>(totalUnusedMemory);
            return (1.0 - (qualityPercent * qualityPercent)) * 100.0;
        }

        Stats const getStats()
        {
            Stats stats;
            stats.totalResidentMemorySize = getSize();
            stats.alignmentSavings = m_stats.alignmentSavings;
            uint64_t totalUnusedSize;
            // Calculate fragmentation and total free blocks at the same time
            stats.fragmentation = getFragmentation(totalUnusedSize);
            stats.unusedSize = totalUnusedSize;
            return stats;
        }

        const std::vector<BlockDesc*>& getBlocks()
        {
            return m_blocks;
        }

    private:

        uint64_t align(uint64_t size, uint64_t alignment)
        {
            return ((size + (alignment - 1)) & ~(alignment - 1));
        }

        void createBlock(uint64_t blockAllocationSize)
        {
            BlockDesc* newBlock = new BlockDesc{};
            newBlock->block.allocate(blockAllocationSize, std::to_string(m_blocks.size()));
            newBlock->size = blockAllocationSize;
            m_blocks.push_back(newBlock);
        }

        bool findFreeSubBlock(BlockDesc* suballocatorBlock,
                              SubBlock* subBlock,
                              uint64_t sizeInBytes)
        {
            bool foundFreeSubBlock   = false;
            auto minUnusedMemoryIter = suballocatorBlock->freeSubBlocks.end();
            auto freeSubBlockIter    = suballocatorBlock->freeSubBlocks.begin();

            uint64_t minUnusedMemorySubBlock = ~0ull;

            while (freeSubBlockIter != suballocatorBlock->freeSubBlocks.end())
            {
                if (sizeInBytes <= freeSubBlockIter->size)
                {
                    // Attempt to find the exact fit and if not fallback to the least wasted unused memory
                    if (freeSubBlockIter->size - sizeInBytes == 0)
                    {
                        // Keep previous allocation size
                        subBlock->size = freeSubBlockIter->size;
                        subBlock->offset = freeSubBlockIter->offset;
                        foundFreeSubBlock = true;

                        // Remove from the list
                        suballocatorBlock->freeSubBlocks.erase(freeSubBlockIter);
                        suballocatorBlock->numSubBlocks++;

                        if (Logger::isEnabled(Level::DBG))
                        {
                            Logger::log(Level::DBG, "RTXMU Suballocator Perfect Match\n");
                        }

                        break;
                    }
                    else
                    {
                        // Keep track of the available SubBlock with least fragmentation
                        const uint64_t unusedMemory = freeSubBlockIter->size - sizeInBytes;
                        if (unusedMemory < minUnusedMemorySubBlock)
                        {
                            minUnusedMemoryIter = freeSubBlockIter;
                            minUnusedMemorySubBlock = unusedMemory;
                        }
                    }
                }
                freeSubBlockIter++;
            }

            // Did not find a perfect match so take the closest and get hit with fragmentation
            // Reject SubBlock if the closest available SubBlock is twice the required size
            if ((foundFreeSubBlock   == false) &&
                (minUnusedMemoryIter != suballocatorBlock->freeSubBlocks.end()) &&
                (minUnusedMemorySubBlock < 2 * sizeInBytes))
            {

                const uint64_t memoryAlignedSize = align(minUnusedMemoryIter->size, suballocatorBlock->block.getAlignment());
                m_stats.alignmentSavings -= (memoryAlignedSize - minUnusedMemoryIter->size);

                // Keep previous allocation size
                subBlock->size = minUnusedMemoryIter->size;
                subBlock->offset = minUnusedMemoryIter->offset;
                foundFreeSubBlock = true;

                // Remove from the list
                suballocatorBlock->freeSubBlocks.erase(minUnusedMemoryIter);
                suballocatorBlock->numSubBlocks++;

                const uint64_t newMemoryAlignedSize = align(sizeInBytes, suballocatorBlock->block.getAlignment());
                // Capture the wasted memory from the original allocation because suballoc size does not change
                subBlock->unusedSize = subBlock->size - sizeInBytes;
                m_stats.alignmentSavings += (newMemoryAlignedSize - sizeInBytes);

                if (Logger::isEnabled(Level::DBG))
                {
                    Logger::log(Level::DBG, "RTXMU Suballocator Suboptimal Match with wasted memory\n");
                }
            }

#if MERGE_FREE_BLOCKS
            // If no free blocks work then lets merge free blocks to help if we have more than 1
            if (foundFreeSubBlock == false && suballocatorBlock->freeSubBlocks.size() > 1)
            {
                std::vector<SubBlock>& freeBlocks = suballocatorBlock->freeSubBlocks;
                int currentSubBlock = 0;
                int neighboringSubBlock = 0;
                while (neighboringSubBlock < freeBlocks.size())
                {
                    uint64_t currentOffset = freeBlocks[currentSubBlock].offset + (freeBlocks[currentSubBlock].size + freeBlocks[currentSubBlock].unusedSize);
                    if (currentOffset == freeBlocks[neighboringSubBlock + 1].offset)
                    {
                        // Add size from the next neighboring block but keep same offset
                        freeBlocks[currentSubBlock].size += freeBlocks[neighboringSubBlock + 1].size;
                        freeBlocks[currentSubBlock].unusedSize += freeBlocks[neighboringSubBlock + 1].unusedSize;

                        freeBlocks[neighboringSubBlock + 1].offset = -1;
                        freeBlocks[neighboringSubBlock + 1].size = 0;

                        if (Logger::isEnabled(Level::DBG))
                        {
                            Logger::log(Level::DBG, "RTXMU Suballocator Merging Free Blocks\n");
                        }
                    }
                    else
                    {
                        currentSubBlock++;
                    }
                    neighboringSubBlock++;
                }
            }
#endif
            return foundFreeSubBlock;
        }

        struct SubBlock : public SubBlockRef
        {
            BlockDesc* blockDesc = nullptr;
            uint64_t offset      = 0;
            uint64_t size        = 0;
            uint64_t unusedSize  = 0;
            bool     isFree      = false;
        };

        struct BlockDesc
        {
            Block block;
            std::vector<SubBlock> freeSubBlocks;
            uint64_t currentOffset = 0;
            uint64_t size          = 0;
            uint64_t numSubBlocks  = 0;
        };

        uint64_t                m_blockSize;
        uint64_t                m_allocationAlignment;
        std::vector<BlockDesc*> m_blocks;
        Stats                   m_stats;
        std::mutex              m_threadSafeLock;
    };

}// end rtxmu namespace
