#ifndef VKC_BUFFER_HPP_INCLUDED
#define VKC_BUFFER_HPP_INCLUDED

#include "base_context.hpp"

namespace vkc
{

class Buffer
{
public:
  Buffer(std::shared_ptr<BaseContext> owner, uint32_t size, vk::BufferUsageFlags usage, VmaMemoryUsage mem_usage = VMA_MEMORY_USAGE_AUTO);
  ~Buffer();

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;

  std::shared_ptr<BaseContext> getContext() const { return ctx_; }

  vk::Buffer apiBuffer() const { return buffer_; } 
  uint32_t getSize() const { return size_; }
  
  void *map();
  void unmap();
  
private:
  std::shared_ptr<BaseContext> ctx_;
  vk::Buffer buffer_;
  VmaAllocation allocation = nullptr;
  uint32_t size_ = 0;
};

using BufferPtr = std::shared_ptr<Buffer>;

}

#endif