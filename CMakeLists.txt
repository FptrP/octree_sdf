cmake_minimum_required(VERSION 3.20)
project(vulkan_nn)

add_subdirectory(src/vkc)

add_executable(main src/main.cpp)
#target_include_directories(main PRIVATE src/vkc)
target_compile_options(main PRIVATE -g)
target_link_libraries(main PUBLIC vkc)