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
#include <d3d12.h>
#include <assert.h>
#include <string>

#pragma comment(lib, "D3D12.lib")

namespace rtxmu
{
    struct Allocator
    {
        ID3D12Device5* device;
    };

    class D3D12Block
    {
    public:

        static D3D12_GPU_VIRTUAL_ADDRESS getGPUVA(D3D12Block block,
                                                  uint64_t   offset);

        void allocate(uint64_t              size,
                      D3D12_HEAP_TYPE       heapType,
                      D3D12_RESOURCE_STATES state,
                      uint32_t              alignment);

        void free();

        uint64_t getVMA();

        ID3D12Resource* getResource();

        static void setAllocator(Allocator* allocator);

    protected:
        static Allocator* m_allocator;

    private:

        ID3D12Resource* m_resource = nullptr;
    };

    class D3D12ScratchBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size, std::string name)
        {
            D3D12Block::allocate(size, heapType, state, alignment);

            name = std::string("RTXMU Scratch Suballocator Block #").append(name);
            std::wstring wideString(name.begin(), name.end());
            getResource()->SetName(wideString.c_str());

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
            D3D12Block::free();
        }
    };

    class D3D12AccelStructBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size, std::string name)
        {
            D3D12Block::allocate(size, heapType, state, alignment);

            name = std::string("RTXMU Result BLAS Suballocator Block #").append(name);
            std::wstring wideString(name.begin(), name.end());
            getResource()->SetName(wideString.c_str());

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
            D3D12Block::free();
        }
    };

    class D3D12CompactedAccelStructBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size, std::string name)
        {
            D3D12Block::allocate(size, heapType, state, alignment);

            name = std::string("RTXMU Compacted BLAS Suballocator Block #").append(name);
            std::wstring wideString(name.begin(), name.end());
            getResource()->SetName(wideString.c_str());

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
            D3D12Block::free();
        }
    };

    class D3D12ReadBackBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_READBACK;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size, std::string name)
        {
            D3D12Block::allocate(size, heapType, state, alignment);

            name = std::string("RTXMU Readback CPU Suballocator Block #").append(name);
            std::wstring wideString(name.begin(), name.end());
            getResource()->SetName(wideString.c_str());

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
            D3D12Block::free();
        }
    };

    class D3D12CompactionWriteBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size, std::string name)
        {
            D3D12Block::allocate(size, heapType, state, alignment);

            name = std::string("RTXMU Compaction Size GPU Suballocator Block #").append(name);
            std::wstring wideString(name.begin(), name.end());
            getResource()->SetName(wideString.c_str());

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
            D3D12Block::free();
        }
    };

    class D3D12UploadCPUBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_GENERIC_READ;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_UPLOAD;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size, std::string name)
        {
            D3D12Block::allocate(size, heapType, state, alignment);

            name = std::string("RTXMU Upload to CPU Suballocator Block #").append(name);
            std::wstring wideString(name.begin(), name.end());
            getResource()->SetName(wideString.c_str());

            if (Logger::isEnabled(Level::DBG))
            {
                char buf[128];
                snprintf(buf, sizeof buf, "RTXMU Upload CPU Suballocator Block Allocation of size %" PRIu64 "\n", size);
                Logger::log(Level::DBG, buf);
            }
        }

        void free()
        {
            if (Logger::isEnabled(Level::DBG))
            {
                Logger::log(Level::DBG, "RTXMU Upload to CPU Suballocator Block Release\n");
            }
            D3D12Block::free();
        }
    };

    class D3D12UploadGPUBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size, std::string name)
        {
            D3D12Block::allocate(size, heapType, state, alignment);

            name = std::string("RTXMU Upload to GPU Suballocator Block #").append(name);
            std::wstring wideString(name.begin(), name.end());
            getResource()->SetName(wideString.c_str());

            if (Logger::isEnabled(Level::DBG))
            {
                char buf[128];
                snprintf(buf, sizeof buf, "RTXMU Upload GPU Suballocator Block Allocation of size %" PRIu64 "\n", size);
                Logger::log(Level::DBG, buf);
            }
        }

        void free()
        {
            if (Logger::isEnabled(Level::DBG))
            {
                Logger::log(Level::DBG, "RTXMU Upload to GPU Suballocator Block Release\n");
            }
            D3D12Block::free();
        }
    };
}