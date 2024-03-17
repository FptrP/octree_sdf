#include "cmd_utils.hpp"

namespace vkc
{

void image_layout_transition(vk::CommandBuffer cmd, ImagePtr ptr, vk::ImageLayout src, vk::ImageLayout dst, vk::PipelineStageFlags opt_stage)
{
  vk::AccessFlags srcAccess {};
  vk::AccessFlags dstAccess {};
  vk::PipelineStageFlags srcStage {};
  vk::PipelineStageFlags dstStage {};

  if (opt_stage == vk::PipelineStageFlags{})
  {
    srcAccess = vk::AccessFlagBits::eMemoryRead|vk::AccessFlagBits::eMemoryWrite;
    dstAccess = vk::AccessFlagBits::eMemoryRead;
    srcStage = vk::PipelineStageFlagBits::eAllCommands;
    dstStage = vk::PipelineStageFlagBits::eAllCommands;
  }
  else if (opt_stage == vk::PipelineStageFlagBits::eTopOfPipe)
  {
    dstAccess = vk::AccessFlagBits::eMemoryRead;
    srcStage = opt_stage;
    dstStage = vk::PipelineStageFlagBits::eAllCommands;
  }
  else if (opt_stage == vk::PipelineStageFlagBits::eBottomOfPipe)
  {
    srcAccess = vk::AccessFlagBits::eMemoryRead|vk::AccessFlagBits::eMemoryWrite;
    srcStage = vk::PipelineStageFlagBits::eAllCommands;
    dstStage = opt_stage;
  }

  vk::ImageMemoryBarrier imgBarrier {
    srcAccess,
    dstAccess,
    src,
    dst
  };
  imgBarrier.setImage(ptr->getImage());
  imgBarrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, ptr->getInfo().mipLevels, 0, ptr->getInfo().arrayLayers});

  cmd.pipelineBarrier(srcStage, dstStage, {}, {}, {}, {imgBarrier});
} 

void global_memory_barrier(vk::CommandBuffer cmd)
{
  vk::MemoryBarrier memBarrier{};
  memBarrier.setSrcAccessMask(vk::AccessFlagBits::eMemoryWrite|vk::AccessFlagBits::eMemoryRead);
  memBarrier.setDstAccessMask(vk::AccessFlagBits::eMemoryRead);

  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, {memBarrier}, {}, {});
}

}