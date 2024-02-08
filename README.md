# RTXMU - RTX Memory Utility

Please read the NVIDIA Blog post on how to use RTXMU:
https://developer.nvidia.com/blog/reducing-acceleration-structure-memory-with-nvidia-rtxmu/

Presentation with slides about RTXMU:
https://www.nvidia.com/en-us/on-demand/session/gdc21-gdc21-01/

To generate a test project:
1) Clone me
2) at root directory mkdir build
3) cd build
4) cmake -G "Visual Studio 16 2019" -A x64 ..

RTXMU - RTX Memory Utility SDK.

The design of this SDK is to allow developers to use compaction and suballocation of
acceleration structure buffers to reduce the memory footprint.  Compaction is proven to reduce the total memory
footprint by more than a half.  Suballocation is proven to reduce memory as well by tightly packing
acceleration structure buffers that are less than 64 KB.

The intended use of this SDK is to batch up all of the acceleration structure build inputs and pass them to
RTXMU which in turn will perform all the suballocation memory requests and build details
including compaction.  Post build info is abstracted away by the SDK in order to do compaction under the hood.
RTXMU returns acceleration structure handle ids that are used to reference the underlying memory buffers.  These handle
ids are passed into RTXMU to create compaction copy workloads, deallocate unused build resources or remove all memory
associated with an acceleration structure.

## Building Ray Traced Sample Application integrated with RTXMU:
Requirements:
Windows or Linux, CMake 3.12, C++ 17, Operating system that supports DXR and/or Vulkan RT, Git

Build Steps for running the donut sample application:
1)  Clone the repository under the rtxmu branch using the command: git clone --recursive https://github.com/NVIDIAGameWorks/donut_examples.git
2)  Open CMake and point “Where is the source code:” to the donut_examples folder.  Create a build folder in the donut_examples folder and point “Where to build the binaries:” to the build folder.
    Then select the cmake variable NVRHI_WITH_RTXMU to ON, click Configure at the bottom and once that is complete, click Generate.  If you are building with Visual Studio then select 2019 and x64 version.
    Now you have a Visual Studio project you can build.
3)  Now open the donut_examples.sln file in Visual Studio and build the entire project.
4)  Find the rt_bindless application folder under Examples/Bindless Ray Tracing and right click the project and set as Startup Project.
5)  By default, Bindless Ray Tracing will run on DXR but if you want to run the Vulkan version just add -vk as a command line argument in the project.

Sample Application built on Nvidia’s Donut engine framework:
https://github.com/NVIDIAGameWorks/donut_examples

RTXMU SDK library code:
https://github.com/NVIDIAGameWorks/RTXMU


## Pseudocode examples using the SDK:

    // Grab RTXMU singleton
    rtxmu::DxAccelStructManager rtxMemUtil(device);

    // Initialize suballocator blocks to 8 MB
    rtxMemUtil.Initialize(8388608);
    
    // Batch up all the acceleration structure build inputs, can be compacted or non compacted
    std::vector<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS> bottomLevelBuildDescs;
    for (auto entity : entityList)
    {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS bottomLevelInputs = {};
    
        bottomLevelInputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
    
        if (entity->isAnimated() == true)
        {
            bottomLevelInputs.Flags |=
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE |
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        }
        else
        {
            bottomLevelInputs.Flags |=
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        }
    
        bottomLevelInputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
        bottomLevelInputs.NumDescs       = 1;
        bottomLevelInputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        bottomLevelInputs.pGeometryDescs = GetBlasGeometryDesc(entity);
    
        bottomLevelBuildDescs.push_back(bottomLevelInputs);
    }
    
    // Records all of the build calls and populates a list of acceleration structure ids
    // that can be passed into the library to get the current GPUVA of the acceleration structure
    std::vector<uint64_t> accelStructIds;
    rtxMemUtil.PopulateBuildCommandList(commandList.Get(),
                                        bottomLevelBuildDescs.data(),
                                        bottomLevelBuildDescs.size(),
                                        accelStructIds);

    // Receives acceleration structure inputs and places UAV barriers for them
    rtxMemUtil.PopulateUAVBarriersCommandList(commandList.Get(), accelStructIds);

    // Performs copies to bring over any compaction size data
    rtxMemUtil.PopulateCompactionSizeCopiesCommandList(commandList, accelStructIds);

    // Create mapping of the accel struct ids to the model objects
    for (int asBufferIndex = 0; asBufferIndex < bottomLevelBuildModels.size(); asBufferIndex++)
    {
        _blasMap[bottomLevelBuildModels[asBufferIndex]] = accelStructIds[asBufferIndex];
    }

    // ----------------------------------------------------------------------------------- //
    // Building the TLAS instance description arrays using AccelStructIds handle vector
    // ----------------------------------------------------------------------------------- //
    for (int instanceIndex = 0; instanceIndex < instanceCount; instanceIndex++)
    {
        memcpy(&_instanceDesc[instanceIndex].Transform,
               &_blasTransforms[instanceIndex * (transformOffset / sizeof(float))],
               sizeof(float) * 12);
    
        _instanceDesc[instanceIndex].InstanceMask                        = 1;
        _instanceDesc[instanceIndex].Flags                               = D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        _instanceDesc[instanceIndex].InstanceID                          = 0;
        _instanceDesc[instanceIndex].InstanceContributionToHitGroupIndex = 0;
    
        // Use the acceleration structure handle to get either the result or the compaction buffer based
        //on the build state of the acceleration structure
        const uint64_t asHandle = _blasMap[entity->getModel()];
        _instanceDesc[instanceIndex].AccelerationStructure = rtxMemUtil.GetAccelStructGPUVA(asHandle);
    }

    // Executing all of the initial builds prior to generating the compaction workloads 
    // because the compaction sizes aren't available in system memory until execution is finished

    commandList->Close();
    gfxQueue->ExecuteCommandLists(1, CommandListCast(commandList.GetAddressOf()));

    int fenceValue = _gfxNextFenceValue[cmdListIndex]++;
    _gfxQueue->Signal(_gfxCmdListFence[cmdListIndex].Get(), fenceValue);

    HANDLE fenceWriteEventECL = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    // Wait for just-submitted command list to finish
    _gfxCmdListFence[cmdListIndex]->SetEventOnCompletion(fenceValue, fenceWriteEventECL);
    WaitForSingleObject(fenceWriteEventECL, INFINITE);

    // Acceleration structures that have finished building on the GPU can now be queued to perform compaction
    if (newBuildsToCompact.size() > 0)
    {
        rtxMemUtil.PopulateCompactionCommandList(commandList.Get(), newBuildsToCompact);
    }

    // Executing all of the compaction workloads prior to cleaning up the initial larger
    // acceleration structure buffer is required to peform the initial build to compaction build copy

    commandList->Close();
    gfxQueue->ExecuteCommandLists(1, CommandListCast(commandList.GetAddressOf()));

    int fenceValue = _gfxNextFenceValue[cmdListIndex]++;
    _gfxQueue->Signal(_gfxCmdListFence[cmdListIndex].Get(), fenceValue);

    HANDLE fenceWriteEventECL = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    // Wait for just-submitted command list to finish
    _gfxCmdListFence[cmdListIndex]->SetEventOnCompletion(fenceValue, fenceWriteEventECL);
    WaitForSingleObject(fenceWriteEventECL, INFINITE);

    // Acceleration structures that have finished building and compacting which means we can deallocate resources
    // used in the build and compaction process (scratch and original build buffer memory).
    if (newBuildsToGarbageCollect.size() > 0)
    {
        rtxMemUtil.GarbageCollection(newBuildsToGarbageCollect);
    }


    // Removing acceleration structure no longer required
    std::vector<uint64_t> accelStructIds;
    for (auto model : modelCountsInEntities)
    {
        if (model.second == 0)
        {
            // Push id onto remove list
            accelStructIds.push_back(_blasMap[model.first]);
            // Remove the blas entry from the list
            _blasMap.erase(model.first);
        }
    }
    
    // Remove all of the no longer referenced accel struct ids by calling RemoveAccelerationStructures
    if (accelStructIds.size() > 0)
    {
        rtxMemUtil.RemoveAccelerationStructures(accelStructIds);
    }

## License
RTXMU is licensed under the [MIT License](LICENSE.txt).
