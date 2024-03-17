#include "base_context.hpp"
#include "buffer.hpp"
#include "shaders.hpp"

namespace vkc
{

class Buffer;

class Context : public BaseContext
{
public:

  Context(bool enable_debug) : BaseContext{enable_debug} {}

  ~Context() { progManager.clear(); }

  std::shared_ptr<Buffer> create_buffer(uint32_t size, vk::BufferUsageFlags usage, VmaMemoryUsage mem_usage = VMA_MEMORY_USAGE_AUTO)
  {
    return std::make_shared<Buffer>(shared_from_this(), size, usage, mem_usage);
  }

  std::shared_ptr<ComputeProgram> loadComputeProgram(const std::string &p)
  {
    return progManager.loadComputeProgram(shared_from_this(), p);
  }

private:
  ProgramManager progManager;
};


}