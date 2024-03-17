#ifndef VKC_CMD_UTILS_HPP_INCLUDED
#define VKC_CMD_UTILS_HPP_INCLUDED

#include "base_context.hpp"
#include "image.hpp"

namespace vkc
{
// stage == {} - barrier for all commands
// stage = vk::PipelineStageFlagBits::eTopOfPipe|bottom - no barriers
void image_layout_transition(vk::CommandBuffer cmd, ImagePtr ptr, vk::ImageLayout src, vk::ImageLayout dst, vk::PipelineStageFlags opt_stage = {});

// sync all stages, make all memory visible and available
void global_memory_barrier(vk::CommandBuffer cmd);

}

#endif