#include <iostream>
#include <random>
#include <cmath>

#include <vkc.hpp>

#include "sdf/sdf.hpp"
#include "sdf/sdf_sparse.hpp"
#include "args_parser.hpp"

#include "stbi/stb_image.h"
#include "stbi/stb_image_write.h"

static void upload_buffer(vk::CommandBuffer cmd, vkc::BufferPtr dst, vkc::BufferPtr staging, const std::vector<float> &src)
{
  auto ctx = dst->getContext();
  auto queue = ctx->mainQueue();

  assert(dst->getSize() >= src.size() * sizeof(float));

  uint32_t dataPerTransfer = std::min(staging->getSize()/sizeof(float), src.size());
  uint32_t dataToTransfer = src.size();
  
  for (uint32_t offs = 0; offs < dataToTransfer; offs += dataPerTransfer)
  {
    uint32_t toTransfer = std::min(dataPerTransfer, dataToTransfer - offs);
    
    auto ptr = staging->map();
    std::memcpy(ptr, src.data() + offs, toTransfer * sizeof(float));
    staging->unmap();

    vk::BufferCopy copyRegion {};
    copyRegion.setSrcOffset(0);
    copyRegion.setDstOffset(offs * sizeof(float));
    copyRegion.setSize(toTransfer * sizeof(float));

    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});
    cmd.copyBuffer(staging->apiBuffer(), dst->apiBuffer(), {copyRegion});
    cmd.end();

    vk::SubmitInfo submitInf {};
    submitInf.setPCommandBuffers(&cmd);
    submitInf.setCommandBufferCount(1);

    queue.submit(submitInf);
    queue.waitIdle();
  }

  cmd.reset();
}

static std::vector<float> download_buffer(vk::CommandBuffer cmd, vkc::BufferPtr src, vkc::BufferPtr staging)
{
  auto ctx = src->getContext();
  auto queue = ctx->mainQueue();

  std::vector<float> dst;
  dst.resize(src->getSize()/sizeof(float));

  //assert(dst->getSize() >= src.size() * sizeof(float));

  uint32_t dataPerTransfer = std::min(staging->getSize()/sizeof(float), src->getSize()/sizeof(float));
  uint32_t dataToTransfer = src->getSize()/sizeof(float);
  
  for (uint32_t offs = 0; offs < dataToTransfer; offs += dataPerTransfer)
  {
    uint32_t toTransfer = std::min(dataPerTransfer, dataToTransfer - offs);
    
    vk::BufferCopy copyRegion {};
    copyRegion.setSrcOffset(offs * sizeof(float));
    copyRegion.setDstOffset(0);
    copyRegion.setSize(toTransfer * sizeof(float));

    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});
    cmd.copyBuffer(src->apiBuffer(), staging->apiBuffer(), {copyRegion});
    cmd.end();

    vk::SubmitInfo submitInf {};
    submitInf.setPCommandBuffers(&cmd);
    submitInf.setCommandBufferCount(1);

    queue.submit(submitInf);
    queue.waitIdle();
  
    auto ptr = staging->map();
    std::memcpy(dst.data() + offs, ptr, toTransfer * sizeof(float));
    staging->unmap();
  }

  cmd.reset();
  return dst;
}

static void save_vec4_to_png(const char *path, uint32_t w, uint32_t h, const std::vector<float> &data, bool ignore_alpha = true)
{
  std::vector<uint32_t> outData;
  outData.resize(w * h);
  assert(data.size() == 4 * w * h);

  for (uint32_t i = 0; i < data.size()/4; i++)
  {
    glm::vec4 val{data[4*i], data[4*i + 1], data[4*i + 2], data[4*i + 3]};
    glm::uvec4 conv = glm::clamp(255.f * val, glm::vec4(0.f), glm::vec4(255.f));
    
    const uint32_t MSK = 255;
    conv = conv & MSK;

    if (ignore_alpha)
      conv.a = 255;

    uint32_t packed = conv.r|(conv.g << 8)|(conv.b << 16)|(conv.a  << 24); 
    outData[i] = packed;
  }

  stbi_write_png(path, w, h, 4, outData.data(), 0);
}

int main(int argc, char **argv)
{
  AppArgs g_args {argc, argv};

  auto ctx = std::make_shared<vkc::Context>(true);

  vk::UniqueCommandPool cmdPool;

  {
    vk::CommandPoolCreateInfo poolInfo {};
    poolInfo.setQueueFamilyIndex(ctx->mainQueueFamily());
    poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    cmdPool = ctx->device().createCommandPoolUnique(poolInfo);
  }

  vk::UniqueDescriptorPool descPool;

  {
    vk::DescriptorPoolSize sizes[] {
      {vk::DescriptorType::eStorageBuffer, 512},
      {vk::DescriptorType::eUniformBuffer, 512},
      {vk::DescriptorType::eSampledImage, 512},
      {vk::DescriptorType::eStorageImage, 512},
      {vk::DescriptorType::eSampler, 512}
    };

    vk::DescriptorPoolCreateInfo info {};
    info.maxSets = 512;
    info.setPoolSizes(sizes);
    info.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
    descPool = ctx->device().createDescriptorPoolUnique(info);
  }

  vk::UniqueCommandBuffer cmd;

  {
    vk::CommandBufferAllocateInfo allocInfo {};
    allocInfo.setCommandPool(*cmdPool);
    allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
    allocInfo.setCommandBufferCount(1);

    auto res = ctx->device().allocateCommandBuffersUnique(allocInfo);
    cmd = std::move(res[0]);
  }

  const auto bufUsage = vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc;
  auto stagingBuffer = ctx->create_buffer(32 << 20ul, bufUsage, VMA_MEMORY_USAGE_CPU_TO_GPU);
  
  sdf::SDFRenderParams renderParams {};
  renderParams.camera = g_args.camera;
  renderParams.fovy = g_args.fovy;
  renderParams.outWidth = g_args.width;
  renderParams.outHeight = g_args.height;
  renderParams.sdfScale = g_args.modelScale;
  //renderParams.camera = sdf::Camera::lookAt({-1.f, 0.0f, 0.f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});

  std::unique_ptr<sdf::SDFDense> sdf;// = sdf::create_sdf_gpu(ctx, {128, 128, 128});
  std::unique_ptr<sdf::SDFSparse> sdfSparse;
  // load sdf
  {
    //auto sdfCpu = sdf::load_from_bin(g_args.modelName.c_str());
    //auto sdfCpu = sdf::create_octahedron(128, 0.45);
    //sdf = sdf::create_sdf_gpu(ctx, {sdfCpu.w, sdfCpu.h, sdfCpu.d});
    //sdf::upload_sdf(*cmd, *sdf, sdfCpu, stagingBuffer);

    //auto sdfBlockList = sdf::dense_to_block_list(sdfCpu, vk::Extent3D{32, 32, 16});

    auto sdfBlockList = sdf::load_blocks_from_bin(g_args.modelName.c_str());

    sdf::SparseSDFCreateInfo info {};
    info.cmd = *cmd;
    info.numMips = sdfBlockList.numMips;
    info.blocks = std::move(sdfBlockList.blocks);
    info.dstSize = sdfBlockList.dstSize;
    info.staging = stagingBuffer;
    info.srcBlockSize = vk::Extent3D{32, 32, 16};

    sdfSparse = sdf::create_sdf_from_blocks(info);
  }

  auto sdfDenseRenderer = std::make_unique<sdf::SDFDenseRenderer>(ctx, *descPool);
  auto sdfSparseRenderer = std::make_unique<sdf::SDFSparseRenderer>(ctx, *descPool);
  // trace sdf
  {
    cmd->begin(vk::CommandBufferBeginInfo{});

    //sdfDenseRenderer->render(*cmd, renderParams, sdfSparse->view, stagingBuffer);
    sdfSparseRenderer->render(*cmd, renderParams, *sdfSparse, stagingBuffer);
    cmd->end();

    vk::SubmitInfo submitInf {};
    submitInf.setPCommandBuffers(&cmd.get());
    submitInf.setCommandBufferCount(1);

    auto queue = ctx->mainQueue();
    queue.submit(submitInf);
    queue.waitIdle();
  }

  //readback buffer
  float *data = (float*)stagingBuffer->map();

  std::vector<float> temp;

  temp.resize(renderParams.outWidth * renderParams.outHeight * 4);
  std::memcpy(temp.data(), data, renderParams.outWidth * renderParams.outHeight * 4 * sizeof(float));
  stagingBuffer->unmap(); 


  //stbi_write_hdr("out.hdr", renderParams.outWidth, renderParams.outHeight, 4, temp.data());
  save_vec4_to_png("out.png", renderParams.outWidth, renderParams.outHeight, temp);

  return 0;
}