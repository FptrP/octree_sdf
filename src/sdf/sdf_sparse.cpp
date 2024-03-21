#include "sdf_sparse.hpp"

namespace sdf
{

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
    assert(mipSize.width <= imgBlockSize.width && mipSize.height <= imgBlockSize.height && imgBlockSize.depth <= imgBlockSize.depth);
    image->addPageMapping(srcBlock.dstMip, srcBlock.offsetInBlocks);
  }

  image->updateMemoryPages();

  auto cmd = info.cmd;

  uint32_t blocksPerTransfer = staging->getSize()/pageSize;
  uint32_t pagesToTransfer = info.blocks.size();
  uint32_t numTransfers = (pagesToTransfer + blocksPerTransfer - 1u)/blocksPerTransfer;
  
  for (uint32_t i = 0; i < numTransfers; i++)
  {


    cmd.reset();
    cmd.begin({});
    //cmd.copyBufferToImage(staging->apiBuffer(), image->getImage(), )
    cmd.end();

    vk::SubmitInfo submitInf {};
    submitInf.setPCommandBuffers(&cmd);
    submitInf.setCommandBufferCount(1);

    auto queue = ctx->mainQueue();
    queue.submit(submitInf);
    queue.waitIdle();
  }
}

}