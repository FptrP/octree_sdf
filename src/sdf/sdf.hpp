#ifndef SDF_SDF_HPP_INCLUDED
#define SDF_SDF_HPP_INCLUDED

#include <vkc.hpp>
#include <vector>

#include <glm/glm.hpp>

namespace sdf
{

struct SDFDense
{
  vk::Extent3D ext; //sdf size

  vkc::ImagePtr image;
  vkc::ImageViewPtr view;
};

struct SDFDenseCPU
{
  uint32_t w;
  uint32_t h;
  uint32_t d;
  
  //distances packing in buffer should match vulkan accesses to texels in vkCmdCopyBufferToImage:  
  //texelOffset = bufferOffset + (x × blockSize) + (y × rowExtent) + (z × sliceExtent) + (layer × layerExtent)
  std::vector<float> dist;
};


SDFDenseCPU create_sphere_sdf(uint32_t dim, float r);

//assumes that file contains dim^3 floats  
SDFDenseCPU load_from_bin(const char *path);

std::unique_ptr<SDFDense> create_sdf_gpu(vkc::ContextPtr ctx, vk::Extent3D ext);
void upload_sdf(vk::CommandBuffer cmd, SDFDense &dst, const SDFDenseCPU &src, vkc::BufferPtr staging);


struct Camera
{
  glm::vec3 pos;
  glm::vec3 x;
  glm::vec3 y;
  glm::vec3 z;

  static Camera lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up);

};

struct SDFRenderParams
{
  uint32_t outWidth = 800;
  uint32_t outHeight = 600;

  Camera camera;

  float fovy = glm::radians(90.f);
  float sdfScale = 1.f;
};

struct SDFDenseRenderer
{
  SDFDenseRenderer(vkc::ContextPtr ctx, vk::DescriptorPool pool);

  // out buffer size must be >= width x height x vec4 
  void render(vk::CommandBuffer cmd, const SDFRenderParams &params, const SDFDense &sdf, vkc::BufferPtr out_buffer);


private:
  vkc::ComputePipelinePtr pipeline;
  vk::UniqueDescriptorSet set;
};



}



#endif