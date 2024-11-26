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

#include <queue>
#include <mutex>
#include <cinttypes>
#include "Logger.h"

namespace rtxmu
{
    constexpr uint64_t SizeOfCompactionDescriptor           = 8;
    constexpr uint32_t BlockAlignment                       = 65536;
    constexpr uint32_t AccelStructAlignment                 = 256;
    constexpr uint64_t CompactionSizeSuballocationBlockSize = 65536;
    constexpr uint64_t DefaultSuballocatorBlockSize         = 8388608;
    constexpr uint64_t ReservedId                           = 0;

    struct AccelerationStructure
    {
        uint64_t compactionSize  = 0;
        uint64_t resultSize      = 0;
        uint64_t scratchSize     = 0;
        bool isCompacted         = false;
        bool requestedCompaction = false;
        bool readyToFree         = false;
    };

    template<typename T>
    class AccelStructManager
    {
    public:

        AccelStructManager(Level logVerbosity) :
        m_buildLogger(""),
        m_suballocationBlockSize(0),
        m_totalUncompactedMemory(0),
        m_totalCompactedMemory(0)
        {
            // Reserve acceleration structure index 0 to not be used
            m_asBufferBuildQueue.resize(1, nullptr);

            Logger::setLoggerSettings(logVerbosity);
        }

        ~AccelStructManager()
        {
            for (T* accelStruct : m_asBufferBuildQueue)
            {
                if (accelStruct != nullptr)
                {
                    delete accelStruct;
                }
            }
        }

        // Resets all queues and frees all memory in suballocators
        void Reset()
        {
            m_totalUncompactedMemory = 0;
            m_totalCompactedMemory = 0;

            for (T* accelStruct : m_asBufferBuildQueue)
            {
                if (accelStruct != nullptr)
                {
                    delete accelStruct;
                }
            }
        }

    protected:
        uint64_t GetAccelStructId()
        {
            uint64_t asId = 0;
            if (m_asIdFreeList.size() > 0)
            {
                asId = m_asIdFreeList.front();
                m_asBufferBuildQueue[asId] = new T();
                m_asIdFreeList.pop();
            }
            else
            {
                m_asBufferBuildQueue.push_back(new T());
                asId = m_asId++;
            }
            return asId;
        }

        void ReleaseAccelStructId(uint64_t accelStructId)
        {
            m_asIdFreeList.push(accelStructId);
            delete m_asBufferBuildQueue[accelStructId];
            m_asBufferBuildQueue[accelStructId] = nullptr;
        }
        
        // Logger
        std::string m_buildLogger;

        // Every suballocator block gets allocated with a configurable size
        uint32_t m_suballocationBlockSize = 0;

        // Limits the amount of transient compaction buffer memory
        uint64_t m_totalUncompactedMemory = 0;
        uint64_t m_totalCompactedMemory = 0;

        std::vector<T*> m_asBufferBuildQueue;
        std::queue<uint64_t> m_asIdFreeList;
        // Increment by 1 the asId by the reserved id that can't be used
        uint64_t m_asId = ReservedId + 1;

        std::mutex m_threadSafeLock;

        Level m_logVerbosity;
    };
}