#ifndef SDF_SPARSE_HPP_INCLUDED
#define SDF_SPARSE_HPP_INCLUDED

#include "sdf.hpp"

namespace sdf
{

struct SDFSparse
{
  vkc::SparseImage3DPtr image;
  vkc::ImageViewPtr view;
  
  vkc::ImagePtr pageMapping;
  vkc::ImageViewPtr pageMappingView; 
};

struct SDFBlock
{
  vk::Offset3D offsetInBlocks;
  uint32_t dstMip;

  std::vector<float> distances; // pointer to blockSIze of distances
};


struct SDFBlockList
{
  vk::Extent3D dstSize;
  uint32_t numMips;
  std::vector<SDFBlock> blocks;
};

SDFBlockList dense_to_block_list(const SDFDenseCPU &src_sdf, vk::Extent3D block_size);

struct BinBlockEntry // block info stored in .bin file
{
  vk::Extent3D coords;
  uint32_t mip;
  uint32_t dataOffset;
};

SDFBlockList load_blocks_from_bin(const char *path);

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