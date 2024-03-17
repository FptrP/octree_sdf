#ifndef VKC_BASE_COTEXT_HPP_INCLUDED
#define VKC_BASE_COTEXT_HPP_INCLUDED

#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>

#include <memory>
#include <cassert>

namespace vkc
{

class BaseContext : public std::enable_shared_from_this<BaseContext>
{
public:
  BaseContext(bool enable_debug);
  ~BaseContext();

  BaseContext(const BaseContext &) = delete;
  BaseContext &operator=(const BaseContext &) = delete;

  vk::Instance instance() const { return *instance_; }
  vk::PhysicalDevice physicalDevice() const { return physDevice_; }
  vk::Device device() const { return *device_; }

  vk::Queue mainQueue() const { return mainQueue_; }
  uint32_t mainQueueFamily() const { return mainQueueFamily_; }

  VmaAllocator allocator() const { return allocator_; }
  
private:
  vk::UniqueInstance instance_;
  vk::PhysicalDevice physDevice_;
  vk::UniqueDevice device_;

  uint32_t mainQueueFamily_;
  vk::Queue mainQueue_;

  VmaAllocator allocator_ = nullptr;
};

}

#endif