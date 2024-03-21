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

class SparseImage3D : public BaseImage
{
public:
  SparseImage3D(std::shared_ptr<BaseContext> &&ctx_, vk::Extent3D ext, uint32_t num_mips, vk::Format fmt);
  ~SparseImage3D();

  void addPageMapping(uint32_t mip_level, vk::Offset3D offset_in_blk); 
  void removePageMapping(uint32_t mip_level, vk::Offset3D offset_in_blk);
  bool isPageMapped(uint32_t mip_level, vk::Offset3D offset_in_blk) const;

  void updateMemoryPages();

  const vk::ImageCreateInfo &getInfo() const { return info; }
  vk::Extent3D getBlockSize() const { return blockSize; }

private:
  vk::ImageCreateInfo info {};
  vk::Extent3D blockSize {0, 0, 0};
  uint32_t bytesPerPixel = 4;
  uint32_t memoryTypeBits = 0;
  
  vk::SparseImageMemoryRequirements sparseRequirements;
  

  struct MappedPage
  {
    vk::Offset3D offset;
    uint32_t mip;
    VmaAllocation allocation;
  };

  std::vector<MappedPage> mappedPages;
};

using SparseImage3DPtr = std::shared_ptr<SparseImage3D>;

}

#endif