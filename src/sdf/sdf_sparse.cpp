#include "sdf_sparse.hpp"

#include <fstream>

namespace sdf
{

static std::vector<float> extract_block(const SDFDenseCPU &src_sdf, vk::Offset3D offset, vk::Extent3D ext)
{
  assert(offset.x + ext.width <= src_sdf.w && offset.y + ext.height <= src_sdf.h && offset.z + ext.depth <= src_sdf.d);
  
  auto srcExt = vk::Extent3D {src_sdf.w, src_sdf.h, src_sdf.d};

  std::vector<float> res;
  res.resize(ext.width * ext.height * ext.depth);

  for (uint32_t z = 0; z < ext.depth; z++)
  {
    for (uint32_t y = 0; y < ext.height; y++)
    {
      for (uint32_t x = 0; x < ext.width; x++)
      {
        uint32_t srcX = x + offset.x;
        uint32_t srcY = y + offset.y;
        uint32_t srcZ = z + offset.z;

        float srcVal = src_sdf.dist.at(srcX + srcY * srcExt.width + srcZ * srcExt.width * srcExt.height);
        res.at(x + y * ext.width + z * ext.width * ext.height) = srcVal;
      }
    }
  }

  return res;
}

SDFBlockList dense_to_block_list(const SDFDenseCPU &src_sdf, vk::Extent3D block_size)
{
  SDFBlockList outSDF;
  outSDF.dstSize = vk::Extent3D {src_sdf.w, src_sdf.h, src_sdf.d};
  outSDF.numMips = 1;

  assert(src_sdf.w % block_size.width == 0 && src_sdf.h % block_size.height == 0 && src_sdf.d % block_size.depth == 0);

  uint32_t numBlkX = src_sdf.w/block_size.width;
  uint32_t numBlkY = src_sdf.h/block_size.height;
  uint32_t numBlkZ = src_sdf.d/block_size.depth;

  std::vector<SDFBlock> blocks;
  blocks.reserve(numBlkX * numBlkY * numBlkZ);

  for (uint32_t blkX = 0; blkX < numBlkX; blkX++)
  {
    for (uint32_t blkY = 0; blkY < numBlkY; blkY++)
    {
      for (uint32_t blkZ = 0; blkZ < numBlkZ; blkZ++)
      {
        vk::Offset3D ofs;
        ofs.x = blkX * block_size.width;
        ofs.y = blkY * block_size.height;
        ofs.z = blkZ * block_size.depth;

        SDFBlock res;
        res.dstMip = 0;
        res.offsetInBlocks = vk::Offset3D{blkX, blkY, blkZ};
        res.distances = extract_block(src_sdf, ofs, block_size);
        
        blocks.push_back(std::move(res));
      }
    }
  }

  outSDF.blocks = std::move(blocks);
  return outSDF;
}

SDFBlockList load_blocks_from_bin(const char *path)
{
  std::ifstream file(path, std::ios::binary);
  std::vector<uint32_t> metadata(9);
  file.read((char *)metadata.data(), metadata.size() * sizeof(decltype(metadata)::value_type));

  vk::Extent3D blockSize {metadata[0], metadata[1], metadata[2]};
  vk::Extent3D topMipSize {metadata[3], metadata[4], metadata[5]};

  uint32_t floatsPerBlock = blockSize.width * blockSize.height * blockSize.depth;
  
  uint32_t topMip = metadata[6];
  uint32_t numBlocks = metadata[7];
  uint32_t numFloats = metadata[8];

  vk::Extent3D mip0Size {topMipSize.width * (1 << topMip), topMipSize.height * (1 << topMip), topMipSize.depth * (1 << topMip)};
  vk::Extent3D texSize {mip0Size.width * blockSize.width, mip0Size.height * blockSize.height, mip0Size.depth * blockSize.depth};

  std::vector<BinBlockEntry> binBlocks(numBlocks);
  std::vector<float> floatData(numFloats);

  file.read((char *)binBlocks.data(), binBlocks.size() * sizeof(BinBlockEntry));
  file.read((char *)floatData.data(), floatData.size() * sizeof(float));

  file.close();

  SDFBlockList outBlockList;
  outBlockList.numMips = topMip + 1;
  outBlockList.dstSize = texSize;
  outBlockList.blocks.reserve(binBlocks.size());


  for (auto &binBlk : binBlocks)
  {
    SDFBlock blk;
    blk.offsetInBlocks = vk::Offset3D{binBlk.coords.width, binBlk.coords.height, binBlk.coords.depth};
    blk.dstMip = binBlk.mip;
    blk.distances.resize(floatsPerBlock);
  
    std::memcpy(blk.distances.data(), floatData.data() + binBlk.dataOffset, floatsPerBlock * sizeof(float));

    outBlockList.blocks.push_back(std::move(blk));
  }


  return outBlockList;
}


static inline uint32_t calc_max_mip_levels(vk::Extent3D size)
{
  uint32_t m = std::max(size.width, std::max(size.height, size.depth));
  assert(m > 0);
  return uint32_t(std::log2(m)) + 1;
}

static inline vk::Extent3D calc_mip_size(vk::Extent3D size, uint32_t mip)
{
  for (uint32_t i = 0; i < mip; i++)
  {
    size.width = std::max(size.width/2u, 1u);
    size.height = std::max(size.height/2u, 1u);
    size.depth = std::max(size.depth/2u, 1u);
  }
  return size;
}

std::unique_ptr<SDFSparse> create_sdf_from_blocks(const SparseSDFCreateInfo &info)
{
  auto ctx = info.staging->getContext();

  auto image = std::make_shared<vkc::SparseImage3D>(info.staging->getContext(), 
    info.dstSize, info.numMips, vk::Format::eR32Sfloat);

  auto imgBlockSize = image->getBlockSize();
  auto staging = info.staging;

  uint32_t pageSize = sizeof(float) * imgBlockSize.width * imgBlockSize.height * imgBlockSize.depth;
  assert(staging->getSize() >= pageSize); // at least one block

  assert(imgBlockSize == info.srcBlockSize);
  assert(info.numMips < calc_max_mip_levels(info.dstSize)); // tail mips not supported yet

  for (auto &srcBlock : info.blocks)  
  {
    assert(srcBlock.dstMip < info.numMips);
    vk::Extent3D mipSize = calc_mip_size(info.dstSize, srcBlock.dstMip);
    // TODO: impl miptail. for now check that all blocks are not in mip tail
    assert(mipSize.width >= imgBlockSize.width && mipSize.height >= imgBlockSize.height && imgBlockSize.depth >= imgBlockSize.depth);
    assert(srcBlock.distances.size() == imgBlockSize.height * imgBlockSize.width * imgBlockSize.depth);
    image->addPageMapping(srcBlock.dstMip, srcBlock.offsetInBlocks);
  }

  image->updateMemoryPages();

  auto cmd = info.cmd;

  uint32_t pagesPerTransfer = staging->getSize()/pageSize;
  uint32_t pagesToTransfer = info.blocks.size();
  uint32_t numTransfers = (pagesToTransfer + pagesPerTransfer - 1u)/pagesPerTransfer;
  
  std::vector<vk::BufferImageCopy> copyRegions;

  for (uint32_t i = 0; i < numTransfers; i++)
  {
    uint32_t startI = i * pagesPerTransfer; 
    uint32_t pagesCount = std::min(pagesPerTransfer, uint32_t(info.blocks.size()) - startI);
    
    uint32_t bufferOffset = 0;
    uint8_t *bufferPtr = (uint8_t*)staging->map();

    for (uint32_t k = 0; k < pagesCount; k++)
    {
      const auto &srcData = info.blocks[startI + k];
      
      vk::BufferImageCopy region {};
      region.bufferOffset = bufferOffset;
      region.imageExtent = imgBlockSize;
      region.imageOffset.x = srcData.offsetInBlocks.x * imgBlockSize.width;
      region.imageOffset.y = srcData.offsetInBlocks.y * imgBlockSize.height;
      region.imageOffset.z = srcData.offsetInBlocks.z * imgBlockSize.depth;
      region.imageSubresource = vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, srcData.dstMip, 0, 1};
    
      std::memcpy(bufferPtr + bufferOffset, srcData.distances.data(), pageSize);

      bufferOffset += pageSize;
      copyRegions.push_back(region);
    }

    staging->unmap();

    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});

    if (i == 0)
      vkc::image_layout_transition(cmd, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTopOfPipe);

    cmd.copyBufferToImage(staging->apiBuffer(), image->getImage(), vk::ImageLayout::eTransferDstOptimal, copyRegions);

    if (i == numTransfers - 1) 
      vkc::image_layout_transition(cmd, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eBottomOfPipe);
    cmd.end();

    vk::SubmitInfo submitInf {};
    submitInf.setPCommandBuffers(&cmd);
    submitInf.setCommandBufferCount(1);

    auto queue = ctx->mainQueue();
    queue.submit(submitInf);
    queue.waitIdle();

    copyRegions.clear();
  }

  cmd.reset();

  vk::ImageViewCreateInfo viewInfo {};
  viewInfo.setViewType(vk::ImageViewType::e3D);
  viewInfo.setFormat(image->getInfo().format);
  viewInfo.setSubresourceRange(vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, image->getInfo().mipLevels, 0, 1});

  auto view = image->createView(viewInfo);

  auto res = std::make_unique<SDFSparse>();
  res->image = image;
  res->view = view;
  return res;
}

}