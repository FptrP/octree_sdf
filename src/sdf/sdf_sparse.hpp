#ifndef SDF_SPARSE_HPP_INCLUDED
#define SDF_SPARSE_HPP_INCLUDED

#include "sdf.hpp"

namespace sdf
{

struct SDFSparse
{
  vkc::SparseImage3DPtr image;
  vkc::ImageViewPtr view;
};

struct SDFBlock
{
  vk::Offset3D offsetInBlocks;
  uint32_t dstMip;

  float *distances; // pointer to blockSIze of distances
};


struct SparseSDFCreateInfo
{
  vk::Extent3D dstSize;
  uint32_t numMips;

  vkc::BufferPtr staging;
  vk::CommandBuffer cmd; //allocated with reset flag
  
  vk::Extent3D srcBlockSize;
  std::vector<SDFBlock> blocks;
};

std::unique_ptr<SDFSparse> create_sdf_from_blocks(const SparseSDFCreateInfo &info);

}

#endif