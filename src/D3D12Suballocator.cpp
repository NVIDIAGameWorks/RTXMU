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
*
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "rtxmu/D3D12Suballocator.h"

namespace rtxmu
{
    Allocator* D3D12Block::m_allocator = nullptr;
    void D3D12Block::setAllocator(Allocator* allocator)
    {
        m_allocator = allocator;
    }

    D3D12_GPU_VIRTUAL_ADDRESS D3D12Block::getGPUVA(D3D12Block block,
                                                    uint64_t   offset)
    {
        D3D12_GPU_VIRTUAL_ADDRESS gpuVA = block.m_resource->GetGPUVirtualAddress() + offset;
        return gpuVA;
    }

    ID3D12Resource* D3D12Block::getResource()
    {
        return m_resource;
    }

    void D3D12Block::allocate(uint64_t              size,
                              D3D12_HEAP_TYPE       heapType,
                              D3D12_RESOURCE_STATES state,
                              uint32_t              alignment)
    {
        ID3D12Device5* device = m_allocator->device;

        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension           = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Alignment           = alignment;
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

        device->CreateCommittedResource(&heapProperties,
                                        D3D12_HEAP_FLAG_NONE,
                                        &desc,
                                        state,
                                        nullptr,
                                        IID_PPV_ARGS(&m_resource));
    }

    void D3D12Block::free()
    {
        m_resource->Release();
        m_resource = nullptr;
    }

    uint64_t D3D12Block::getVMA() { return static_cast<uint64_t>(m_resource->GetGPUVirtualAddress()); }
}
