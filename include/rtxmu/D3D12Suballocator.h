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
    // If suballocator blocks are larger than 4 MB then use placed resources for large page sizes
    constexpr bool Use4MBAlignedPlacedResources = true;

    class D3D12Block
    {
    public:
        static ID3D12Device5* m_device;
        ID3D12Resource*       m_resource = nullptr;

        static D3D12_GPU_VIRTUAL_ADDRESS getGPUVA(D3D12Block block,
                                                  uint64_t   offset)
        {
            D3D12_GPU_VIRTUAL_ADDRESS gpuVA = block.m_resource->GetGPUVirtualAddress() + offset;
            return gpuVA;
        }

        void allocate(uint64_t              size,
                      D3D12_HEAP_TYPE       heapType,
                      D3D12_RESOURCE_STATES state,
                      uint32_t              alignment)
        {
            D3D12_RESOURCE_DESC desc = {};
            desc.Dimension           = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Alignment           = 0;
            desc.Width               = size;
            desc.Height              = 1;
            desc.DepthOrArraySize    = 1;
            desc.MipLevels           = 1;
            desc.Format              = DXGI_FORMAT_UNKNOWN;
            desc.SampleDesc.Count    = 1;
            desc.SampleDesc.Quality  = 0;
            desc.Layout              = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

            if ((heapType != D3D12_HEAP_TYPE_READBACK) &&
                (heapType != D3D12_HEAP_TYPE_UPLOAD))
            {
                desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            }
            else
            {
                desc.Flags = D3D12_RESOURCE_FLAG_NONE;
            }

            D3D12_HEAP_PROPERTIES heapProperties = {};
            heapProperties.Type                  = heapType;
            heapProperties.MemoryPoolPreference  = D3D12_MEMORY_POOL_UNKNOWN;
            heapProperties.CPUPageProperty       = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            heapProperties.CreationNodeMask      = 1;
            heapProperties.VisibleNodeMask       = 1;

            if ((Use4MBAlignedPlacedResources == true) &&
                (alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT))
            {
                ID3D12Heap*     heap     = nullptr;                
                D3D12_HEAP_DESC heapDesc = {};
                heapDesc.SizeInBytes     = size;
                heapDesc.Properties      = heapProperties;
                heapDesc.Alignment       = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;
                heapDesc.Flags           = D3D12_HEAP_FLAG_NONE;

                m_device->CreateHeap(&heapDesc,
                                    IID_PPV_ARGS(&heap));

                m_device->CreatePlacedResource(heap,
                                               0,
                                               &desc,
                                               state,
                                               nullptr,
                                               IID_PPV_ARGS(&m_resource));
            }
            else if (alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT)
            {
                m_device->CreateCommittedResource(&heapProperties,
                                                D3D12_HEAP_FLAG_NONE,
                                                &desc,
                                                state,
                                                nullptr,
                                                IID_PPV_ARGS(&m_resource));
            }
        }

        void free()
        {
            m_resource->Release();
            m_resource = nullptr;
        }

        uint64_t getVMA() { return static_cast<uint64_t>(m_resource->GetGPUVirtualAddress()); }
    };

    class D3D12ScratchBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size)
        {
            D3D12Block::allocate(size, heapType, state, alignment);
        }
    };

    class D3D12AccelStructBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size)
        {
            D3D12Block::allocate(size, heapType, state, alignment);
        }
    };

    class D3D12ReadBackBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_READBACK;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size)
        {
            D3D12Block::allocate(size, heapType, state, alignment);
        }
    };

    class D3D12CompactionWriteBlock : public D3D12Block
    {
    public:
        static constexpr D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        static constexpr D3D12_HEAP_TYPE heapType = D3D12_HEAP_TYPE_DEFAULT;
        static constexpr uint32_t alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

        uint32_t getAlignment() { return alignment; }

        void allocate(uint64_t size)
        {
            D3D12Block::allocate(size, heapType, state, alignment);
        }
    };
}
