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
#include <d3d12.h>

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

        void allocate(Allocator*            allocator,
                      uint64_t              size,
                      D3D12_HEAP_TYPE       heapType,
                      D3D12_RESOURCE_STATES state);

        void free(Allocator* allocator);

        uint64_t getVMA();

        ID3D12Resource* getResource();

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

        void allocate(Allocator* allocator, uint64_t size)
        {
            D3D12Block::allocate(allocator, size, heapType, state);
        }
    };

    class D3D12AccelStructBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(Allocator* allocator, uint64_t size)
        {
            D3D12Block::allocate(allocator, size, heapType, state);
        }
    };

    class D3D12ReadBackBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_READBACK;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(Allocator* allocator, uint64_t size)
        {
            D3D12Block::allocate(allocator, size, heapType, state);
        }
    };

    class D3D12CompactionWriteBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(Allocator* allocator, uint64_t size)
        {
            D3D12Block::allocate(allocator, size, heapType, state);
        }
    };

    class D3D12UploadCPUBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_GENERIC_READ;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_UPLOAD;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(Allocator* allocator, uint64_t size)
        {
            D3D12Block::allocate(allocator, size, heapType, state);
        }
    };

    class D3D12UploadGPUBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(Allocator* allocator, uint64_t size)
        {
            D3D12Block::allocate(allocator, size, heapType, state);
        }
    };
}
