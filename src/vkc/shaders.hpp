#ifndef VKC_SHADERS_HPP_INCLUDED
#define VKC_SHADERS_HPP_INCLUDED

#include "base_context.hpp"
#include "buffer.hpp"
#include "image.hpp"

#include <unordered_map>
#include <string>
#include <optional>
#include <algorithm>
#include <filesystem>
#include <variant>

#include <spirv_reflect.h>

namespace vkc
{

class ComputePipeline;

// push const info. descriptor set info. spec const info 

struct ConstSpecEntry
{
  uint32_t constId = 0;
  std::variant<uint32_t, int, float> value; //should be enough for now... 

  static constexpr uint32_t ConstSize = sizeof(uint32_t);
};

struct BufferBinding
{
  BufferBinding(BufferPtr buf) : buffer {std::move(buf)}
  {
    offset = 0;
    range = buffer->getSize();
  }
  
  BufferBinding(BufferPtr buf, uint32_t offs, uint32_t size)
    : buffer {std::move(buf)}, offset {offs}, range{size} {}

  BufferPtr buffer;
  uint32_t offset;
  uint32_t range;
};

struct ImageBinding
{
  vk::Sampler sampler;
  ImageViewPtr view;
  vk::ImageLayout layout {vk::ImageLayout::eShaderReadOnlyOptimal};
};

struct DescriptorWrite
{
  uint32_t binding;
  std::variant<BufferBinding, ImageBinding> value;
};

class ComputeProgram : public std::enable_shared_from_this<ComputeProgram>
{
public:

  ComputeProgram(std::weak_ptr<BaseContext> ctx_, const std::vector<uint32_t> &code);
  ~ComputeProgram() { spvReflectDestroyShaderModule(&mod); }
  
  ComputeProgram(const ComputeProgram &) = delete;
  ComputeProgram &operator=(const ComputeProgram &) = delete;

  std::shared_ptr<ComputePipeline> makePipeline(const std::vector<ConstSpecEntry> &const_specs);

  void writeDescSet(vk::DescriptorSet set, uint32_t set_id, const std::vector<DescriptorWrite> &writes);

  vk::DescriptorSet allocDescSet(vk::DescriptorPool pool, uint32_t set_id = 0);
  vk::UniqueDescriptorSet allocDescSetUnique(vk::DescriptorPool pool, uint32_t set_id = 0);

  vk::DescriptorSetLayout getDescLayout(uint32_t index) const { return *descLayouts.at(index); }
  vk::PipelineLayout getPipelineLayout() const { return *pLayout; }
  
private:
  std::weak_ptr<BaseContext> ctx_;

  SpvReflectShaderModule mod;
  vk::UniqueShaderModule shaderMod;
  std::vector<vk::UniqueDescriptorSetLayout> descLayouts;
  vk::UniquePipelineLayout pLayout;
};


class ProgramManager
{
public:
  ProgramManager() {}

  std::shared_ptr<ComputeProgram> loadComputeProgram(std::weak_ptr<BaseContext> ctx, const std::string &path);

  void clear() { programs.clear(); }

private:

  std::unordered_map<std::string, std::shared_ptr<ComputeProgram>> programs;
};


class ComputePipeline
{
public:
  ComputePipeline(std::shared_ptr<BaseContext> ctx, std::shared_ptr<ComputeProgram> base_prog, vk::UniquePipeline &&pipeline)
    : ctx_{ctx}, program {base_prog}, pipeline {std::move(pipeline)} {}

  std::shared_ptr<ComputeProgram> getProgram() const { return program; } 

  vk::Pipeline getPipeline() const { return *pipeline; }

private:
  std::shared_ptr<BaseContext> ctx_;
  std::shared_ptr<ComputeProgram> program;
  vk::UniquePipeline pipeline;
};

using ComputePipelinePtr = std::shared_ptr<ComputePipeline>; 

}


#endif