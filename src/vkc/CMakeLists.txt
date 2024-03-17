cmake_minimum_required(VERSION 3.20)

find_package(Vulkan REQUIRED)

add_subdirectory(VulkanMemoryAllocator)

OPTION(SPIRV_REFLECT_EXECUTABLE     "Build spirv-reflect executable" OFF)
OPTION(SPIRV_REFLECT_STATIC_LIB     "Build a SPIRV-Reflect static library" ON)
add_subdirectory(SPIRV-Reflect)

add_library(vkc 
  base_context.hpp
  base_context.cpp 
  buffer.cpp
  shaders.cpp
  image.cpp
  sampler_pool.cpp
  cmd_utils.cpp)

target_compile_definitions(vkc PRIVATE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC)
target_compile_options(vkc PRIVATE -g)
target_include_directories(vkc PUBLIC ./)
target_link_libraries(vkc PUBLIC Vulkan::Vulkan GPUOpen::VulkanMemoryAllocator spirv-reflect-static)