#include "image.hpp"

namespace vkc
{

ImageViewPtr BaseImage::createView(const vk::ImageViewCreateInfo &info)
{
  return std::make_shared<ImageView>(shared_from_this(), info);
}

Image::Image(std::shared_ptr<BaseContext> &&ctx_, const vk::ImageCreateInfo &info_)
  : BaseImage {std::move(ctx_)}
{
  info = info_;

  VmaAllocationCreateInfo allocCreateInfo {};
  allocCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  VkImageCreateInfo cInfo = info;
  VkImage cImageHandle = nullptr;
  
  auto res = vmaCreateImage(ctx->allocator(), &cInfo, &allocCreateInfo, &cImageHandle, &allocation, nullptr);
  assert(res == VK_SUCCESS);

  handle = cImageHandle;
}

Image::~Image()
{
  if (handle && allocation)
    vmaDestroyImage(ctx->allocator(), handle, allocation);
  handle = nullptr;
  allocation = nullptr;
}

ImageView::ImageView(std::shared_ptr<BaseImage> image_, const vk::ImageViewCreateInfo &info_)
{
  owner = image_;
  info = info_;
  info.image = owner->getImage();

  auto ctx = owner->getContext();
  handle = ctx->device().createImageView(info);
}

ImageView::~ImageView()
{
  if (handle)
  {
    auto ctx = owner->getContext();
    ctx->device().destroyImageView(handle);
  }
}

}