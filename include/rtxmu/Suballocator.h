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

#include <vector>
#include <cstdint>

namespace rtxmu
{
    enum class ResultCode : uint8_t
    {
        Ok = 0,
        OutOfMemoryCPU,
        OutOfMemoryGPU,
        InvalidSuballocationAlignment,
        InvalidBlockAlignment
    };

    // Defines the functions that need to be implemented by the templated Block type
    class BlockInterface
    {
    public:
        ResultCode allocate(uint64_t size) = delete;
        uint32_t   getAlignment()          = delete;
        void       free()                  = delete;
        uint32_t   getVMA()                = delete;
    };

    // Block type default implementation to force client to implement
    template<typename Block = BlockInterface>
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
                     uint64_t allocationAlignment)
        {
            m_blockSize = blockSize;
            m_allocationAlignment = allocationAlignment;
        }

        SubAllocation allocate(uint64_t size)
        {
            // Align allocation
            const uint64_t sizeInBytes = align(size, m_allocationAlignment);

            // If no previous blocks exist 
            if (m_blocks.size() == 0)
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
                BlockDesc* block    = m_blocks[m_blocks.size() - 1];
                subBlock->blockDesc = block;
                subBlock->size      = sizeInBytes;
                subBlock->offset    = block->currentOffset;

                const uint64_t offsetInBytes = block->currentOffset + sizeInBytes;
                block->currentOffset         = offsetInBytes;
                block->numSubBlocks++;
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

                        const uint64_t memoryAlignedSize = align(subBlock->size, m_blocks[blockIndex]->block.getAlignment());
                        m_stats.alignmentSavings += (memoryAlignedSize - subBlock->size);

                        break;
                    }
                }
            }

            // Pass a generic SubAllocation struct back to client
            return {subBlock->blockDesc->block, subBlock->offset, subBlock};
        }

        void free(SubBlockRef* subBlockRef)
        {
            SubBlock* subBlock = reinterpret_cast<SubBlock*>(subBlockRef);
            uint64_t blockIndex = 0;
            for (BlockDesc* blockDesc : m_blocks)
            {
                if (blockDesc->block.getVMA() == subBlock->blockDesc->block.getVMA())
                {
                    const uint64_t memoryAlignedSize = align(subBlock->size, blockDesc->block.getAlignment());
                    m_stats.alignmentSavings        -= (memoryAlignedSize - subBlock->size);
                    m_stats.fragmentedSize          -= (subBlock->unusedSize);

                    subBlock->isFree = true;

                    // Release the big chunks that are a single resource
                    if (subBlock->size == blockDesc->size)
                    {
                        blockDesc->block.free();
                        m_blocks.erase(m_blocks.begin() + blockIndex);
                    }
                    else
                    {
                        struct SubBlock freeSubBlock = { SubBlockRef(),
                                                         subBlock->blockDesc,
                                                         subBlock->offset,
                                                         subBlock->size };

                        blockDesc->freeSubBlocks.push_back(freeSubBlock);

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

        // All sizes are expressed in bytes
        struct Stats
        {
            uint64_t alignmentSavings = 0;
            uint64_t fragmentedSize = 0;
        };

        Stats const& getStats() const { return m_stats; }

        uint64_t getSize()
        {
            uint64_t size = 0;
            for (auto blockDesc : m_blocks)
            {
                size += blockDesc->size;
            }
            return size;
        }

        std::vector<BlockDesc*> getBlocks()
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
            newBlock->block.allocate(blockAllocationSize);
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
                // Keep previous allocation size
                subBlock->size    = minUnusedMemoryIter->size;
                subBlock->offset  = minUnusedMemoryIter->offset;
                foundFreeSubBlock = true;

                // Remove from the list
                suballocatorBlock->freeSubBlocks.erase(minUnusedMemoryIter);
                suballocatorBlock->numSubBlocks++;

                subBlock->unusedSize = subBlock->size - sizeInBytes;
                m_stats.fragmentedSize += subBlock->unusedSize;
            }

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
    };

}// end rtxmu namespace

