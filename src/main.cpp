#include <iostream>
#include <random>
#include <cmath>

#include <vkc.hpp>

#include "vknn/context.hpp"
#include "vknn/mm_kernel.hpp"

static std::vector<float> make_random_matrix(uint32_t rows, uint32_t cols)
{
  std::random_device r;
  std::default_random_engine e1(r());
  std::uniform_real_distribution<float> uniform_dist {-1.f, 1.f};

  std::vector<float> mat;
  mat.resize(rows * cols); 

  for (uint32_t i = 0; i < mat.size(); i++)
    mat[i] = uniform_dist(e1);

  return mat;
} 

// m1 M x N
// m2 N x K
// res M x K 
static std::vector<float> mult_mat_cpu(const std::vector<float> &m1, const std::vector<float> &m2, uint32_t m, uint32_t n, uint32_t k)
{
  std::vector<float> out;
  out.resize(m * k, 0.f);

  for (uint32_t row = 0; row < m; row++)
  {
    for (uint32_t col = 0; col < k; col++)
    {
      //res[row, col] = dot(m1's row, m2's col ) 

      for (uint32_t i = 0; i < n; i++)
        out[row * k + col] += m1[row * n + i] * m2[i * k + col]; 
    }
  }

  return out;
}

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

int main()
{
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
      {vk::DescriptorType::eUniformBuffer, 512}
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

  auto stagingBuffer = ctx->create_buffer(32 << 20ul, vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_CPU_TO_GPU);

  const uint32_t batch_size = 64;
  const uint32_t features = 512;
  const uint32_t out_features = 1024; 

  const auto bufUsage = vk::BufferUsageFlagBits::eTransferDst|vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc;

  auto input_mat = make_random_matrix(batch_size, features);
  auto weight_mat = make_random_matrix(features, out_features);

  auto input_buffer = ctx->create_buffer(input_mat.size() * sizeof(float), bufUsage, VMA_MEMORY_USAGE_GPU_ONLY);
  auto weight_buffer = ctx->create_buffer(weight_mat.size() * sizeof(float), bufUsage, VMA_MEMORY_USAGE_GPU_ONLY);
  auto output_buffer = ctx->create_buffer(batch_size * out_features * sizeof(float), bufUsage, VMA_MEMORY_USAGE_GPU_ONLY);

  upload_buffer(*cmd, input_buffer, stagingBuffer, input_mat);
  upload_buffer(*cmd, weight_buffer, stagingBuffer, weight_mat);

  // launch kernel
  auto prog = ctx->loadComputeProgram("src/shaders/matrix_mult.spv");
  auto pipeline = prog->makePipeline({});

  vk::UniqueDescriptorSet set;
  {
    auto descLayouts = {prog->getDescLayout(0)};

    vk::DescriptorSetAllocateInfo allocInfo {};
    allocInfo.setDescriptorPool(*descPool);
    allocInfo.setSetLayouts(descLayouts);

    auto res = ctx->device().allocateDescriptorSetsUnique(allocInfo);
    set = std::move(res[0]);
  }
  
  prog->writeDescSet(*set, 0, {
    {0, vkc::BufferBinding {input_buffer}},
    {1, vkc::BufferBinding {weight_buffer}},
    {2, vkc::BufferBinding {output_buffer}}
  });

  {
    struct PushConst
    {
      uint32_t A_row;
      uint32_t A_col;
      uint32_t B_row;
      uint32_t B_col;
    } pc;

    pc.A_row = batch_size;
    pc.A_col = features;
    pc.B_row = features;
    pc.B_col = out_features;

    uint32_t groupSizeX = 8;
    uint32_t groupSizeY = 4;

    cmd->begin(vk::CommandBufferBeginInfo {});

    cmd->bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->getPipeline());
    cmd->bindDescriptorSets(vk::PipelineBindPoint::eCompute, prog->getPipelineLayout(), 0, {*set}, {});
    cmd->pushConstants<PushConst>(prog->getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, {pc});
    cmd->dispatch((out_features + groupSizeX - 1)/groupSizeX, (batch_size + groupSizeY - 1)/groupSizeY, 1);
    cmd->end();

    vk::SubmitInfo submitInf {};
    submitInf.setPCommandBuffers(&cmd.get());
    submitInf.setCommandBufferCount(1);

    auto queue = ctx->mainQueue();
    queue.submit(submitInf);
    queue.waitIdle();
  }

  auto gpu_res = download_buffer(*cmd, output_buffer, stagingBuffer);
  auto cpu_res = mult_mat_cpu(input_mat, weight_mat, batch_size, features, out_features);

  float maxDiff = 0.f;

  for (uint32_t i = 0; i < gpu_res.size(); i++)
  {
    maxDiff = std::max(maxDiff, std::fabs(gpu_res[i] - cpu_res[i]));
  }

  std::cout << "Max diff " << maxDiff << "\n";

  return 0;
}