cmake_minimum_required(VERSION 3.10)

project(rtxmu VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(CMakeDependentOption)

if (NOT RTXMU_VULKAN_INCLUDE_DIR)
	find_package(Vulkan REQUIRED FATAL_ERROR)
	set(RTXMU_VULKAN_INCLUDE_DIR ${Vulkan_INCLUDE_DIR})
endif()

cmake_dependent_option(RTXMU_WITH_D3D12 "Support D3D12" ON WIN32 OFF)
option(RTXMU_WITH_VULKAN "Support Vulkan" ON)

set (HEADER_FILES
	include/rtxmu/Suballocator.h
	include/rtxmu/AccelStructManager.h)

set (SRC_FILES "")

if (RTXMU_WITH_D3D12)
	list (APPEND SRC_FILES
		src/D3D12Suballocator.cpp
		src/D3D12AccelStructManager.cpp)

	list (APPEND HEADER_FILES
		include/rtxmu/D3D12Suballocator.h
		include/rtxmu/D3D12AccelStructManager.h)
endif()

if (RTXMU_WITH_VULKAN)
	list (APPEND SRC_FILES
		src/VulkanSuballocator.cpp
		src/VkAccelStructManager.cpp)

	list (APPEND HEADER_FILES
		include/rtxmu/VkAccelStructManager.h
		include/rtxmu/VulkanSuballocator.h)
endif()

add_library(rtxmu STATIC ${HEADER_FILES} ${SRC_FILES})

target_include_directories(rtxmu PUBLIC include)
target_include_directories(rtxmu PRIVATE ${RTXMU_VULKAN_INCLUDE_DIR})

set_target_properties(rtxmu PROPERTIES LINKER_LANGUAGE CXX)
