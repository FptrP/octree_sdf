#ifndef VKC_SAMPLER_POOL_HPP_INCLUDED
#define VKC_SAMPLER_POOL_HPP_INCLUDED

#include "base_context.hpp"

#include <unordered_map>

namespace vkc
{

struct SamplerHashFunc {
  std::size_t operator()(const vk::SamplerCreateInfo &info) const;
};

struct SamplerEqualFunc : std::binary_function<vk::SamplerCreateInfo, vk::SamplerCreateInfo, bool> {
  bool operator()(const vk::SamplerCreateInfo &l, const vk::SamplerCreateInfo &r) const;
};

class SamplerPool
{
public:
  
  vk::Sampler getSampler(const std::shared_ptr<BaseContext> &ctx, const vk::SamplerCreateInfo &info);

  void clear(vk::Device device);

private:
  std::unordered_map<vk::SamplerCreateInfo, vk::Sampler, SamplerHashFunc, SamplerEqualFunc> samplers; 
};


static constexpr vk::SamplerCreateInfo DEF_SMOOTH_SAMPLER
{
  {}, 
  vk::Filter::eLinear,
  vk::Filter::eLinear,
  vk::SamplerMipmapMode::eLinear,
  vk::SamplerAddressMode::eClampToEdge,
  vk::SamplerAddressMode::eClampToEdge,
  vk::SamplerAddressMode::eClampToEdge
};

static constexpr vk::SamplerCreateInfo DEF_NEAREST_SAMPLER {
  {}, 
  vk::Filter::eNearest,
  vk::Filter::eNearest,
  vk::SamplerMipmapMode::eNearest,
  vk::SamplerAddressMode::eClampToEdge,
  vk::SamplerAddressMode::eClampToEdge,
  vk::SamplerAddressMode::eClampToEdge
};

}


#endif