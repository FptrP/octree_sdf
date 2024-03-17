#ifndef VKC_IMAGE_HPP_INCLUDED
#define VKC_IMAGE_HPP_INCLUDED

#include "base_context.hpp"

namespace vkc
{

class Image;
class ImageView;

using ImagePtr = std::shared_ptr<Image>;
using ImageViewPtr = std::shared_ptr<ImageView>;

class BaseImage : public std::enable_shared_from_this<BaseImage>
{
public:
  BaseImage(std::shared_ptr<BaseContext> &&ctx_) : ctx {std::move(ctx_)} {}

  vk::Image &getImage() { return handle; }
  const vk::Image &getImage() const { return handle; }
  
  // info.
  ImageViewPtr createView(const vk::ImageViewCreateInfo &info);

  BaseImage(const BaseImage&) = delete;
  BaseImage &operator=(const BaseImage&) = delete;

  std::shared_ptr<BaseContext> getContext() const { return ctx; }

protected:
  std::shared_ptr<BaseContext> ctx;
  vk::Image handle;
};

class Image : public BaseImage
{
public:
  Image(std::shared_ptr<BaseContext> &&ctx_, const vk::ImageCreateInfo &info_);
  ~Image();

  const vk::ImageCreateInfo &getInfo() const { return info; }

private:
  VmaAllocation allocation;
  vk::ImageCreateInfo info {};
};

class ImageView
{
public:
  ImageView(std::shared_ptr<BaseImage> image_, const vk::ImageViewCreateInfo &info_);
  ~ImageView();

  vk::ImageView getView() const { return handle; }
  const vk::ImageViewCreateInfo &getInfo() const { return info; }

private:
  std::shared_ptr<BaseImage> owner;
  vk::ImageView handle {};
  vk::ImageViewCreateInfo info {};
};


}

#endif