#include "base_context.hpp"
#include "buffer.hpp"
#include "image.hpp"
#include "shaders.hpp"
#include "sampler_pool.hpp"

namespace vkc
{

class Buffer;

class Context : public BaseContext
{
public:

  Context(bool enable_debug) : BaseContext{enable_debug} {}

  ~Context()
  { 
    progManager.clear();
    samplerPool.clear(device());
  }

  std::shared_ptr<Buffer> create_buffer(uint32_t size, vk::BufferUsageFlags usage, VmaMemoryUsage mem_usage = VMA_MEMORY_USAGE_AUTO)
  {
    return std::make_shared<Buffer>(shared_from_this(), size, usage, mem_usage);
  }

  std::shared_ptr<ComputeProgram> loadComputeProgram(const std::string &p)
  {
    return progManager.loadComputeProgram(shared_from_this(), p);
  }

  ImagePtr create_image(const vk::ImageCreateInfo &info)
  {
    return std::make_shared<Image>(shared_from_this(), info);
  }

  vk::Sampler getSampler(const vk::SamplerCreateInfo &info)
  {
    return samplerPool.getSampler(shared_from_this(), info);
  }

private:
  ProgramManager progManager;
  SamplerPool samplerPool;
};

using ContextPtr = std::shared_ptr<Context>;


}