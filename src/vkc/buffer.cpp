#include "buffer.hpp"

namespace vkc
{

Buffer::Buffer(std::shared_ptr<BaseContext> owner, uint32_t size, vk::BufferUsageFlags usage, VmaMemoryUsage mem_usage)
  : ctx_(owner)
{

  vk::BufferCreateInfo bufInfo {};
  bufInfo.size = size;
  bufInfo.usage = usage;
  bufInfo.sharingMode = vk::SharingMode::eExclusive;
  
  VmaAllocationCreateInfo allocCreateInfo {};
  allocCreateInfo.usage = mem_usage;
  
  VkBufferCreateInfo cbufInfo = bufInfo;
  VkBuffer cbuffer;

  auto res = vmaCreateBuffer(ctx_->allocator(), &cbufInfo, &allocCreateInfo, &cbuffer, &allocation, nullptr);
  assert(res == VK_SUCCESS);
  
  buffer_ = cbuffer;
  size_ = size;
}

Buffer::~Buffer()
{
  if (buffer_ && allocation)
    vmaDestroyBuffer(ctx_->allocator(), buffer_, allocation);
}

void* Buffer::map()
{
  void *out;
  auto res = vmaMapMemory(ctx_->allocator(), allocation, &out);
  assert(res == VK_SUCCESS);
  return out;
}

void Buffer::unmap()
{
  vmaUnmapMemory(ctx_->allocator(), allocation);
}

}