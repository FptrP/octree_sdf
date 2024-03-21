cmake_minimum_required(VERSION 3.20)
project(vulkan_nn)

find_package(glm REQUIRED)

add_subdirectory(src/vkc)

add_executable(main 
  src/main.cpp
  src/sdf/sdf.cpp
  src/sdf/sdf_sparse.cpp
  src/stbi/stb_impl.cpp)

#target_include_directories(main PRIVATE src/vkc)
target_compile_options(main PRIVATE -g)
target_link_libraries(main PUBLIC vkc)
#target_link_libraries(main PRIVATE glm::glm-header-only)