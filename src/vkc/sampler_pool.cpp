#include "sampler_pool.hpp"

namespace vkc
{

template <typename T>
inline void hash_combine(std::size_t &s, const T &v) {
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s<< 6) + (s>> 2); 
}

 
std::size_t SamplerHashFunc::operator()(const vk::SamplerCreateInfo &info) const
{
  std::size_t h = 0;
  hash_combine(h, info.magFilter);
  hash_combine(h, info.minFilter);
  hash_combine(h, info.mipmapMode);
  hash_combine(h, info.addressModeU);
  hash_combine(h, info.addressModeV);
  hash_combine(h, info.addressModeW);
  hash_combine(h, info.mipLodBias);
  hash_combine(h, info.anisotropyEnable);
  hash_combine(h, info.maxAnisotropy);
  hash_combine(h, info.compareEnable);
  hash_combine(h, info.compareOp);
  hash_combine(h, info.minLod);
  hash_combine(h, info.maxLod);
  hash_combine(h, info.borderColor);
  hash_combine(h, info.unnormalizedCoordinates);
  return h;
}

bool SamplerEqualFunc::operator()(const vk::SamplerCreateInfo &l, const vk::SamplerCreateInfo &r) const {
  return 
    (l.magFilter == r.magFilter) &&
    (l.minFilter == r.minFilter) &&
    (l.mipmapMode == r.mipmapMode) &&
    (l.addressModeU == r.addressModeU) &&
    (l.addressModeV == r.addressModeV) &&
    (std::abs(l.mipLodBias - r.mipLodBias) < 1e-6) &&
    (l.anisotropyEnable == r.anisotropyEnable) &&
    (std::abs(l.maxAnisotropy - r.maxAnisotropy) < 1e-6) &&
    (l.compareEnable == r.compareEnable) &&
    (std::abs(l.minLod - r.minLod) < 1e-6) &&
    (std::abs(l.maxLod - r.maxLod) < 1e-6) &&
    (l.borderColor == r.borderColor) &&
    (l.unnormalizedCoordinates == r.unnormalizedCoordinates);
}

vk::Sampler SamplerPool::getSampler(const std::shared_ptr<BaseContext> &ctx, const vk::SamplerCreateInfo &info)
{
  auto it = samplers.find(info);
  if (it != samplers.end())
    return it->second;

  vk::Sampler newSampler = ctx->device().createSampler(info);
  samplers[info] = newSampler;
  return newSampler;
}

void SamplerPool::clear(vk::Device device)
{
  for (auto &[k, v] : samplers)
    device.destroySampler(v);
  samplers.clear();
}

}