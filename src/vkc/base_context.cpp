#define VMA_IMPLEMENTATION
#include "base_context.hpp"

#include <algorithm>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace vkc
{

static vk::PhysicalDevice findDevice(const std::vector<vk::PhysicalDevice> &devices)
{
  auto it = std::find_if(devices.begin(), devices.end(), [](vk::PhysicalDevice dev)
  {
    return dev.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
  });

  if (it != devices.end())
    return *it;

  return devices.at(0);
}

BaseContext::BaseContext(bool enable_debug)
{
  //vk::createInstance()
  VULKAN_HPP_DEFAULT_DISPATCHER.init();

  std::vector<const char*> enabledLayers {};
  std::vector<const char*> enabledExtensions {};

  if (enable_debug)
    enabledLayers.push_back("VK_LAYER_KHRONOS_validation");

  vk::ApplicationInfo appInfo {"vkc", VK_MAKE_VERSION(0, 0, 1), "vkc", VK_MAKE_VERSION(0, 0, 1), VK_API_VERSION_1_3};

  vk::InstanceCreateInfo instanceInfo {};
  instanceInfo.setPApplicationInfo(&appInfo);
  instanceInfo.setPEnabledLayerNames(enabledLayers);
  instanceInfo.setPEnabledExtensionNames(enabledExtensions);

  instance_ = vk::createInstanceUnique(instanceInfo);

  VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance_);

  auto physicalDevices = instance_->enumeratePhysicalDevices();
  
  physDevice_ = findDevice(physicalDevices);

  auto deviceFeatures =  physDevice_.getFeatures();

  auto queueFamilies = physDevice_.getQueueFamilyProperties2();

  auto requiredFlags = vk::QueueFlagBits::eCompute|vk::QueueFlagBits::eGraphics|vk::QueueFlagBits::eTransfer;

  auto queueIt = std::find_if(queueFamilies.begin(), queueFamilies.end(), [=](const auto &prop) {
    return (prop.queueFamilyProperties.queueFlags & requiredFlags) == requiredFlags; 
  });

  float priorities[1] {1.f};

  mainQueueFamily_ = std::distance(queueFamilies.begin(), queueIt);

  vk::DeviceQueueCreateInfo queueInfo({}, mainQueueFamily_, priorities);

  std::vector<vk::DeviceQueueCreateInfo> queueInfos {
    queueInfo
  };

  std::vector<const char*> deviceExtensions {};

  vk::DeviceCreateInfo deviceInfo {};
  deviceInfo.setQueueCreateInfos(queueInfos);
  deviceInfo.setPEnabledExtensionNames(deviceExtensions);
  deviceInfo.setPEnabledFeatures(&deviceFeatures);

  device_ = physDevice_.createDeviceUnique(deviceInfo);

  VULKAN_HPP_DEFAULT_DISPATCHER.init(*device_);

  mainQueue_ = device_->getQueue(mainQueueFamily_, 0);

  #define FILL_ENTRY(x) .x = VULKAN_HPP_DEFAULT_DISPATCHER.x

  VmaVulkanFunctions vmaFuncs {
    FILL_ENTRY(vkGetInstanceProcAddr),
    FILL_ENTRY(vkGetDeviceProcAddr),
    FILL_ENTRY(vkGetPhysicalDeviceProperties),
    FILL_ENTRY(vkGetPhysicalDeviceMemoryProperties),
    FILL_ENTRY(vkAllocateMemory),
    FILL_ENTRY(vkFreeMemory),
    FILL_ENTRY(vkMapMemory),
    FILL_ENTRY(vkUnmapMemory),
    FILL_ENTRY(vkFlushMappedMemoryRanges),
    FILL_ENTRY(vkInvalidateMappedMemoryRanges),
    FILL_ENTRY(vkBindBufferMemory),
    FILL_ENTRY(vkBindImageMemory),
    FILL_ENTRY(vkGetBufferMemoryRequirements),
    FILL_ENTRY(vkGetImageMemoryRequirements),
    FILL_ENTRY(vkCreateBuffer),
    FILL_ENTRY(vkDestroyBuffer),
    FILL_ENTRY(vkCreateImage),
    FILL_ENTRY(vkDestroyImage),
    FILL_ENTRY(vkCmdCopyBuffer),
    FILL_ENTRY(vkGetBufferMemoryRequirements2KHR),
    FILL_ENTRY(vkGetImageMemoryRequirements2KHR),
    FILL_ENTRY(vkBindBufferMemory2KHR),
    FILL_ENTRY(vkBindImageMemory2KHR),
    FILL_ENTRY(vkGetPhysicalDeviceMemoryProperties2KHR),
    FILL_ENTRY(vkGetDeviceBufferMemoryRequirements),
    FILL_ENTRY(vkGetDeviceImageMemoryRequirements)
  };

  #undef FILL_ENTRY

  VmaAllocatorCreateInfo allocInfo{
    .flags = 0,
    .physicalDevice = physDevice_,
    .device = device_.get(),
    .pVulkanFunctions = &vmaFuncs,
    .instance = instance_.get(),
    .vulkanApiVersion = VK_API_VERSION_1_3
  };

  assert(vmaCreateAllocator(&allocInfo, &allocator_) == VK_SUCCESS);
}

BaseContext::~BaseContext()
{
  if (allocator_)
    vmaDestroyAllocator(allocator_);
}


}