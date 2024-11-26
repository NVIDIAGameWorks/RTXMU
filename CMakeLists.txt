cmake_minimum_required(VERSION 3.10)

project(rtxmu VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(CMakeDependentOption)

if (NOT RTXMU_VULKAN_INCLUDE_DIR)
	#find_package(Vulkan REQUIRED FATAL_ERROR)
	find_package(Vulkan)
	if (Vulkan_FOUND)
		set(RTXMU_VULKAN_INCLUDE_DIR ${Vulkan_INCLUDE_DIR})
	else()
		# CMake may fail to locate the libraries but could be able to
		# provide some path in Vulkan SDK include directory variable
		# 'Vulkan_INCLUDE_DIRS', try to extract path from this.
		message(STATUS "Failed to locate Vulkan SDK, retrying again...")
		if(EXISTS "${VULKAN_PATH}")
			message(STATUS "Successfully located the Vulkan SDK: ${VULKAN_PATH}")
		else()
			message("Error: Unable to locate Vulkan SDK. Please turn off auto locate option by specifying 'AUTO_LOCATE_VULKAN' as 'OFF'")
			message("and specify manually path using 'VULKAN_SDK' and 'VULKAN_VERSION' variables in the CMakeLists.txt.")
		endif()
	endif()
endif()

cmake_dependent_option(RTXMU_WITH_D3D12 "Support D3D12" ON WIN32 OFF)
option(RTXMU_WITH_VULKAN "Support Vulkan" ON)

set (HEADER_FILES
	include/rtxmu/Logger.h
	include/rtxmu/Suballocator.h
	include/rtxmu/AccelStructManager.h)

set (SRC_FILES "")

if (RTXMU_WITH_D3D12)
	list (APPEND SRC_FILES
		src/D3D12Suballocator.cpp
		src/D3D12AccelStructManager.cpp
		src/Logger.cpp)

	list (APPEND HEADER_FILES
		include/rtxmu/D3D12Suballocator.h
		include/rtxmu/D3D12AccelStructManager.h)
endif()

if (RTXMU_WITH_VULKAN)
	list (APPEND SRC_FILES
		src/VulkanSuballocator.cpp
		src/VkAccelStructManager.cpp
		src/Logger.cpp)

	list (APPEND HEADER_FILES
		include/rtxmu/VkAccelStructManager.h
		include/rtxmu/VulkanSuballocator.h)
endif()

add_library(rtxmu STATIC ${HEADER_FILES} ${SRC_FILES})

target_include_directories(rtxmu PUBLIC include)
target_include_directories(rtxmu PRIVATE ${RTXMU_VULKAN_INCLUDE_DIR})

set_target_properties(rtxmu PROPERTIES LINKER_LANGUAGE CXX)
