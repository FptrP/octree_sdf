#ifndef VKC_HPP_INCLUDED
#define VKC_HPP_INCLUDED

#include "base_context.hpp"
#include "buffer.hpp"
#include "context.hpp"
 
namespace vkc
{
  // context 
  // base context -> instance, device, allocator
  
  // buffers manager - allocate, map, unmap  
  
  // pipelines manager 
  // program -> specialize ->  pipeline 
  // pipeline has descriptor set layout
  
  // descriptor sets 

  /*
    DescriptorSetInfo

    Program 
      - DescriptorSetInfo


    Pipeline
      - DescriptorSetInfo
      - DescriptorSetLayout
      - PushConstantSize 
  */

  /*
    command buffer pool and so on
  */
}


#endif