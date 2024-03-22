#include "sdf_sparse.hpp"

#include <fstream>
#include <vulkan/vulkan_format_traits.hpp>

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

static vkc::ImagePtr create_page_mapping(const SparseSDFCreateInfo &info, vk::Extent3D block_size)
{
  auto ctx = std::static_pointer_cast<vkc::Context>(info.staging->getContext());
  
  vk::Extent3D imgSize = info.dstSize;
  imgSize.width /= block_size.width;
  imgSize.height /= block_size.height;
  imgSize.depth /= block_size.depth;

  vk::ImageCreateInfo imgInfo {};
  imgInfo.setImageType(vk::ImageType::e3D);
  imgInfo.setExtent(imgSize);
  imgInfo.setFormat(vk::Format::eR8Uint);
  imgInfo.setArrayLayers(1);
  imgInfo.setMipLevels(1);
  imgInfo.setTiling(vk::ImageTiling::eOptimal);
  imgInfo.setSharingMode(vk::SharingMode::eExclusive);
  imgInfo.setUsage(vk::ImageUsageFlagBits::eTransferDst|vk::ImageUsageFlagBits::eTransferSrc|vk::ImageUsageFlagBits::eSampled);

  auto image = ctx->create_image(imgInfo);

  //imgInfo.set
  //auto image = ctx->create_image(); 
  
  std::vector<uint8_t> pageMips(imgSize.width * imgSize.height * imgSize.depth);
  std::fill(pageMips.begin(), pageMips.end(), uint8_t(255u));

  for (auto &blk : info.blocks)
  {
    //blk.dstMip -> calc offset, extent in mip 0 blocks
    uint32_t mipSize = 1 << blk.dstMip;
    
    vk::Offset3D offst {
      blk.offsetInBlocks.x * mipSize,
      blk.offsetInBlocks.y * mipSize,
      blk.offsetInBlocks.z * mipSize
    };
    
    for (uint32_t ix = offst.x; ix < offst.x + mipSize; ix++)
    {
      for (uint32_t iy = offst.y; iy < offst.y + mipSize; iy++)
      {
        for (uint32_t iz = offst.z; iz < offst.z + mipSize; iz++)
        {
          uint32_t index = ix + iy * imgSize.width + iz * imgSize.width * imgSize.height;
          pageMips.at(index) = std::min(pageMips.at(index), uint8_t(blk.dstMip));
        }
      }
    }
  }

  for (uint32_t i = 0; i < pageMips.size(); i++)
    assert(pageMips[i] < 255); // check that all pages are mapped (at least in 1 mip)

  auto staging = info.staging;
  assert(staging->getSize() >= pageMips.size() * sizeof(uint8_t));

  auto ptr = staging->map();
  std::memcpy(ptr, pageMips.data(), pageMips.size() * sizeof(uint8_t));
  staging->unmap();

  auto cmd = info.cmd;

  vk::BufferImageCopy copyRegion {};
  copyRegion.setImageExtent(imgSize);
  copyRegion.setImageSubresource(vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1});

  cmd.reset();
  cmd.begin(vk::CommandBufferBeginInfo{});
  vkc::image_layout_transition(cmd, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTopOfPipe);

  cmd.copyBufferToImage(staging->apiBuffer(), image->getImage(), vk::ImageLayout::eTransferDstOptimal, {copyRegion});
  vkc::image_layout_transition(cmd, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eBottomOfPipe);
  cmd.end();

  vk::SubmitInfo submitInf {};
  submitInf.setPCommandBuffers(&cmd);
  submitInf.setCommandBufferCount(1);

  auto queue = ctx->mainQueue();
  queue.submit(submitInf);
  queue.waitIdle();

  return image;
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

  auto pageMapImage = create_page_mapping(info, imgBlockSize);
  
  viewInfo.setViewType(vk::ImageViewType::e3D);
  viewInfo.setFormat(pageMapImage->getInfo().format);
  viewInfo.setSubresourceRange(vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});

  auto pageMapView = pageMapImage->createView(viewInfo);

  auto res = std::make_unique<SDFSparse>();
  res->image = image;
  res->view = view;
  res->pageMapping = pageMapImage;
  res->pageMappingView = pageMapView;

  return res;
}

SDFSparseRenderer::SDFSparseRenderer(vkc::ContextPtr ctx, vk::DescriptorPool pool)
{
  auto prog = ctx->loadComputeProgram("src/shaders/trace_sparse_sdf.spv");
  pipeline = prog->makePipeline({});
  set = prog->allocDescSetUnique(pool);

  nearestSampler = ctx->getSampler(vkc::DEF_NEAREST_SAMPLER);

  auto smpInfo = vkc::DEF_SMOOTH_SAMPLER;
  smpInfo.mipmapMode = vk::SamplerMipmapMode::eNearest;

  linearSampler = ctx->getSampler(smpInfo);
}

void SDFSparseRenderer::render(vk::CommandBuffer cmd, const SDFRenderParams &params, const SDFSparse &sdf, vkc::BufferPtr out_buffer)
{
  assert(out_buffer->getSize() >= params.outWidth * params.outHeight * sizeof(glm::vec4));

  auto ctx = std::static_pointer_cast<vkc::Context>(out_buffer->getContext());
  
  struct PushConsts
  {
    glm::vec4 camera_pos;
    glm::vec4 camera_x;
    glm::vec4 camera_y;
    glm::vec4 camera_z;

    float projDist; // distance to proj plane

    uint outWidth;
    uint outHeight;
    uint samplesPerPixel;
  
    float sdfAABBScale; // sdf is a cube in range [-scale, +scale] centered in (0, 0, 0)
    uint _pad0;
    uint _pad1;
    uint _pad2;
  } pc {
    .camera_pos = glm::vec4{params.camera.pos, 1.f},
    .camera_x = glm::vec4{params.camera.x, 0.f},
    .camera_y = glm::vec4{params.camera.y, 0.f},
    .camera_z = glm::vec4{params.camera.z, 0.f},
    .projDist = 1.f/(2.f * tanf(params.fovy * 0.5)),

    .outWidth = params.outWidth,
    .outHeight = params.outHeight,
    .samplesPerPixel = 1,

    .sdfAABBScale = params.sdfScale
  };

  pipeline->getProgram()->writeDescSet(*set, 0, {
    {0, vkc::BufferBinding {out_buffer}},
    {1, vkc::ImageBinding {nullptr, sdf.view, vk::ImageLayout::eShaderReadOnlyOptimal}},
    {2, vkc::ImageBinding {nearestSampler, nullptr, vk::ImageLayout::eUndefined}},
    {3, vkc::ImageBinding {linearSampler, nullptr, vk::ImageLayout::eUndefined}},
    {4, vkc::ImageBinding {nearestSampler, sdf.pageMappingView, vk::ImageLayout::eShaderReadOnlyOptimal}}
  });

  const uint32_t workGroupX = 8;
  const uint32_t workGroupY = 4;

  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->getPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->getProgram()->getPipelineLayout(), 0, {*set}, {});
  cmd.pushConstants(pipeline->getProgram()->getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
  cmd.dispatch((pc.outWidth + workGroupX - 1)/workGroupX, (pc.outHeight + workGroupY - 1)/workGroupY, 1);
}


}