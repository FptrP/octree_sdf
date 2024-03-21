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

SparseImage3D::SparseImage3D(std::shared_ptr<BaseContext> &&ctx_, vk::Extent3D ext, uint32_t num_mips, vk::Format fmt)
 : BaseImage {std::move(ctx_)}
{
  auto imgType = vk::ImageType::e3D;
  auto sampleCount = vk::SampleCountFlagBits::e1;
  
  auto imgUsage = vk::ImageUsageFlagBits::eTransferDst
    |vk::ImageUsageFlagBits::eTransferSrc
    |vk::ImageUsageFlagBits::eSampled
    |vk::ImageUsageFlagBits::eStorage;

  auto imgTiling = vk::ImageTiling::eOptimal;
  auto imgAspect = vk::ImageAspectFlagBits::eColor;

  auto fmtInfo = ctx->physicalDevice().getSparseImageFormatProperties(fmt, imgType, sampleCount, imgUsage, imgTiling).at(0);
  blockSize = fmtInfo.imageGranularity;
  
  info = vk::ImageCreateInfo {};
  info.setExtent(ext);
  info.setArrayLayers(1);
  info.setMipLevels(num_mips); // mip tail not covered
  info.setImageType(imgType);
  info.setFormat(fmt);
  info.setSamples(sampleCount);
  info.setSharingMode(vk::SharingMode::eExclusive);
  info.setTiling(imgTiling);
  info.setUsage(imgUsage);
  info.setFlags(vk::ImageCreateFlagBits::eSparseBinding|vk::ImageCreateFlagBits::eSparseResidency);

  handle = ctx->device().createImage(info);

  auto imageReq = ctx->device().getImageMemoryRequirements(handle);
  memoryTypeBits = imageReq.memoryTypeBits;

  // TODO: calculate bytes per pixel
  bytesPerPixel = sizeof(float);
  //auto req = ctx->device().getImageMemoryRequirements(handle);
  //auto req = ctx->device().getImageSparseMemoryRequirements(handle);

  //vk::FormatProperties::
  
}

SparseImage3D::~SparseImage3D()
{
  // process allocations
  for (auto &page : mappedPages)
    vmaFreeMemory(ctx->allocator(), page.allocation);
  
  mappedPages.clear();

  if (handle)
    ctx->device().destroyImage(handle);
}

void SparseImage3D::addPageMapping(uint32_t mip_level, vk::Offset3D offset_in_blk)
{
  assert(!isPageMapped(mip_level, offset_in_blk));
  assert(mip_level < info.mipLevels);

  uint32_t pageSize = bytesPerPixel * blockSize.width * blockSize.height * blockSize.depth;
  
  vk::MemoryRequirements memoryReq {};
  memoryReq.memoryTypeBits = memoryTypeBits;
  memoryReq.alignment = pageSize;
  memoryReq.size = pageSize;

  VmaAllocationCreateInfo allocInfo {};
  allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  VkMemoryRequirements cMemoryReq = memoryReq;
  VmaAllocation allocation = nullptr;
  auto res = vmaAllocateMemory(ctx->allocator(), &cMemoryReq, &allocInfo, &allocation, nullptr);
  assert(res == VK_SUCCESS);

  MappedPage page;
  page.allocation = allocation;
  page.mip = mip_level;
  page.offset = offset_in_blk;

  mappedPages.push_back(page);
}

void SparseImage3D::removePageMapping(uint32_t mip_level, vk::Offset3D offset_in_blk)
{
  auto it = std::find_if(mappedPages.begin(), mappedPages.end(), [=](const MappedPage &page) {
    return page.mip == mip_level && page.offset == offset_in_blk;
  });

  assert(it != mappedPages.end());
  vmaFreeMemory(ctx->allocator(), it->allocation);
  mappedPages.erase(it);
}

bool SparseImage3D::isPageMapped(uint32_t mip_level, vk::Offset3D offset_in_blk) const
{
  auto it = std::find_if(mappedPages.begin(), mappedPages.end(), [=](const MappedPage &page) {
    return page.mip == mip_level && page.offset == offset_in_blk;
  });

  return it != mappedPages.end();
}

void SparseImage3D::updateMemoryPages()
{
  std::vector<vk::SparseImageMemoryBind> memBinds;
  memBinds.reserve(mappedPages.size());

  for (auto &srcPage : mappedPages)
  {
    VmaAllocationInfo allocInfo {};
    vmaGetAllocationInfo(ctx->allocator(), srcPage.allocation, &allocInfo);

    vk::SparseImageMemoryBind memBind {};
    
    vk::Offset3D srcOffset;
    srcOffset.setX(srcPage.offset.x * blockSize.width);
    srcOffset.setY(srcPage.offset.y * blockSize.height);
    srcOffset.setZ(srcPage.offset.z * blockSize.depth);
    
    vk::ImageSubresource dstSubres{};
    dstSubres.setArrayLayer(0);
    dstSubres.setAspectMask(vk::ImageAspectFlagBits::eColor);
    dstSubres.setMipLevel(srcPage.mip);


    memBind.setOffset(srcOffset);
    memBind.setExtent(blockSize);
    memBind.setSubresource(dstSubres);
    memBind.setMemory(allocInfo.deviceMemory);
    memBind.setMemoryOffset(allocInfo.offset);

    memBinds.push_back(memBind);
  }


  vk::BindSparseInfo bindInfo {};
  vk::SparseImageMemoryBindInfo imageBind {};
  imageBind.setBinds(memBinds);
  imageBind.setImage(handle);

  bindInfo.setPImageBinds(&imageBind);
  bindInfo.setImageBindCount(1);

  //imageBind.setBinds()

  //bindInfo.setImageBinds()

  ctx->mainQueue().bindSparse(bindInfo);
  ctx->mainQueue().waitIdle();
}

}