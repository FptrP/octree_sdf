#include "shaders.hpp"

#include <fstream>

namespace vkc
{

ComputeProgram::ComputeProgram(std::weak_ptr<BaseContext> ctx, const std::vector<uint32_t> &code)
  : ctx_ {ctx}
{

  auto ctx_ptr = ctx_.lock();

  auto res = spvReflectCreateShaderModule(code.size() * sizeof(uint32_t), code.data(), &mod);
  assert(res == SPV_REFLECT_RESULT_SUCCESS);
  assert(mod.shader_stage == SpvReflectShaderStageFlagBits::SPV_REFLECT_SHADER_STAGE_COMPUTE_BIT); // only compute 

  assert(mod.push_constant_block_count <= 1); // only 1 push constant block 

  descLayouts.reserve(mod.descriptor_set_count);

  std::vector<vk::DescriptorSetLayoutBinding> bindings;

  for (uint32_t set_id = 0; set_id < mod.descriptor_set_count; set_id++)
  {
    const auto &srcSet = mod.descriptor_sets[set_id];  
    
    bindings.clear();
    bindings.reserve(srcSet.binding_count);

    for (uint32_t b_id = 0; b_id < srcSet.binding_count; b_id++)
    {
      if (!srcSet.bindings[b_id])
        continue;

      const auto &src = *srcSet.bindings[b_id];
    
      vk::DescriptorSetLayoutBinding binding {};
    
      binding.binding = src.binding;
      binding.descriptorType = vk::DescriptorType{src.descriptor_type};
      binding.descriptorCount = 1; // todo - array bindings
      binding.stageFlags = vk::ShaderStageFlagBits::eCompute;

      bindings.push_back(binding);
    }

    vk::DescriptorSetLayoutCreateInfo info {};
    info.setBindings(bindings);
    
    descLayouts.push_back(ctx_ptr->device().createDescriptorSetLayoutUnique(info));
  }

  std::vector<vk::DescriptorSetLayout> baseLayouts;
  baseLayouts.reserve(descLayouts.size());

  for (uint32_t i = 0; i < descLayouts.size(); i++)
    baseLayouts.push_back(*descLayouts[i]);

  vk::PushConstantRange pcRanges [] {
    {vk::ShaderStageFlagBits::eCompute, 0, 0}
  };

  if (mod.push_constant_block_count)
    pcRanges[0].size = mod.push_constant_blocks[0].size;

  vk::PipelineLayoutCreateInfo info {};
  info.setSetLayouts(baseLayouts);
  info.setPushConstantRanges(pcRanges);

  pLayout = ctx_ptr->device().createPipelineLayoutUnique(info);

  vk::ShaderModuleCreateInfo modInfo {};
  modInfo.setCode(code);
  
  shaderMod = ctx_ptr->device().createShaderModuleUnique(modInfo);
}

std::shared_ptr<ComputePipeline> ComputeProgram::makePipeline(const std::vector<ConstSpecEntry> &const_specs)
{
  std::vector<uint32_t> specData;
  std::vector<vk::SpecializationMapEntry> specEntries; 
  
  specEntries.reserve(const_specs.size());
  specData.resize(const_specs.size());

  uint32_t offset = 0;
  
  for (auto &spec : const_specs)
  {
    vk::SpecializationMapEntry dst {};
    dst.constantID = spec.constId;
    dst.offset = offset * ConstSpecEntry::ConstSize;
    dst.size = ConstSpecEntry::ConstSize;

    //std::memcpy specData[offset]
    std::visit([&](auto val)
    {
      std::memcpy(&specData[offset], &val, sizeof(val));
    }, 
    spec.value);

    dst.offset++;
  }

  vk::SpecializationInfo specInfo {};
  specInfo.setMapEntries(specEntries);
  specInfo.setPData(specData.data());
  specInfo.setDataSize(specData.size() * ConstSpecEntry::ConstSize);
  
  //specInfo.setMapEntries()
  vk::PipelineShaderStageCreateInfo stage {};
  stage.setStage(vk::ShaderStageFlagBits::eCompute);
  stage.setModule(*shaderMod);
  stage.setPName(mod.entry_point_name);

  if (const_specs.size())
    stage.setPSpecializationInfo(&specInfo);

  vk::ComputePipelineCreateInfo info {};

  info.layout = *pLayout;
  info.setStage(stage);
  
  auto ctx_ptr = ctx_.lock(); 

  auto res = ctx_ptr->device().createComputePipelineUnique(nullptr, info);
  assert(res.result == vk::Result::eSuccess);

  return std::make_shared<ComputePipeline>(std::move(ctx_ptr), shared_from_this(), std::move(res.value));
}

void ComputeProgram::writeDescSet(vk::DescriptorSet set, uint32_t set_id, const std::vector<DescriptorWrite> &writes)
{
  auto setInfo = std::find_if(mod.descriptor_sets, mod.descriptor_sets + mod.descriptor_set_count, [=](const SpvReflectDescriptorSet &set)
  {
    return set.set == set_id;
  });

  assert(setInfo != mod.descriptor_sets + mod.descriptor_set_count);
  
  auto findBinding = [&](uint32_t index) -> SpvReflectDescriptorBinding*
  {
    for (uint32_t i = 0; i < setInfo->binding_count; i++)
    {
      if (setInfo->bindings[i] && setInfo->bindings[i]->binding == index)
        return setInfo->bindings[i];
    }
    return nullptr;
  };

  std::vector<std::unique_ptr<vk::DescriptorBufferInfo>> bufferInfos;  
  std::vector<std::unique_ptr<vk::DescriptorImageInfo>> imageInfos;
  std::vector<vk::WriteDescriptorSet> writeDesc;
  
  writeDesc.reserve(writes.size());

  for (auto &src : writes)
  {
    auto bindingInfo = findBinding(src.binding);
    assert(bindingInfo);

    auto bufBinding = std::get_if<BufferBinding>(&src.value);
    auto imgBinding = std::get_if<ImageBinding>(&src.value);

    assert(bufBinding || imgBinding);

    vk::WriteDescriptorSet write {};
    write.setDstSet(set);
    write.setDescriptorType(vk::DescriptorType{bindingInfo->descriptor_type});
    write.setDescriptorCount(1);
    write.setDstBinding(src.binding);
    write.setDstArrayElement(0);

    if (bufBinding)
    {
      auto bufInfo = std::make_unique<vk::DescriptorBufferInfo>();
      bufInfo->setBuffer(bufBinding->buffer->apiBuffer());
      bufInfo->setOffset(bufBinding->offset);
      bufInfo->setRange(bufBinding->range);
      
      write.setPBufferInfo(bufInfo.get());
      bufferInfos.push_back(std::move(bufInfo));
    }
    else if (imgBinding)
    {
      auto imgInfo = std::make_unique<vk::DescriptorImageInfo>();
      imgInfo->setSampler(imgBinding->sampler);
      imgInfo->setImageView(imgBinding->view? imgBinding->view->getView() : nullptr);
      imgInfo->setImageLayout(imgBinding->layout);
      
      write.setPImageInfo(imgInfo.get());
      imageInfos.push_back(std::move(imgInfo));
    }

    writeDesc.push_back(write);
  }

  auto ctx_ptr = ctx_.lock();
  ctx_ptr->device().updateDescriptorSets(writeDesc, {});
}

vk::DescriptorSet ComputeProgram::allocDescSet(vk::DescriptorPool pool, uint32_t set_id)
{
  auto layouts = {descLayouts.at(set_id).get()};

  vk::DescriptorSetAllocateInfo allocInfo {};
  allocInfo.setDescriptorPool(pool);
  allocInfo.setSetLayouts(layouts);

  auto ctx = ctx_.lock();
  auto res = ctx->device().allocateDescriptorSets(allocInfo);
  return res[0];  
}

vk::UniqueDescriptorSet ComputeProgram::allocDescSetUnique(vk::DescriptorPool pool, uint32_t set_id)
{
  auto layouts = {descLayouts.at(set_id).get()};

  vk::DescriptorSetAllocateInfo allocInfo {};
  allocInfo.setDescriptorPool(pool);
  allocInfo.setSetLayouts(layouts);

  auto ctx = ctx_.lock();
  auto res = ctx->device().allocateDescriptorSetsUnique(allocInfo);
  return std::move(res[0]);  
}

std::shared_ptr<ComputeProgram> ProgramManager::loadComputeProgram(std::weak_ptr<BaseContext> ctx, const std::string &path)
{
  auto file_path = std::filesystem::absolute(path).string();

  if (auto it = programs.find(file_path); it != programs.end())
    return it->second;
  
  std::shared_ptr<ComputeProgram> prog;
  
  {
    std::ifstream file{file_path.c_str(), std::ios::ate|std::ios::binary};

    auto fileSize = file.tellg();

    assert(fileSize % sizeof(uint32_t) == 0);

    std::vector<uint32_t> buffer(fileSize/sizeof(uint32_t)); 
    file.seekg(0);
    file.read((char*)buffer.data(), buffer.size() * sizeof(uint32_t));
    file.close();

    prog = std::make_shared<ComputeProgram>(ctx, buffer);
  }
  
  programs[file_path] = prog;
  return prog;
}

}