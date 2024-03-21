#include "sdf.hpp"

#include <cmath>
#include <fstream>

namespace sdf
{

SDFDenseCPU create_sphere_sdf(uint32_t dim, float r)
{
  SDFDenseCPU sdf {dim, dim, dim, {}};
  sdf.dist.resize(dim * dim * dim);

  float scale = 1.f/dim;

  uint32_t rowExt = dim;
  uint32_t sliceExt = rowExt * dim;

  for (uint32_t zi = 0; zi < dim; zi++)
  {
    for (uint32_t yi = 0; yi < dim; yi++)
    {
      for (uint32_t xi = 0; xi < dim; xi++)
      {
        float x = (xi + 0.5f) * scale - 0.5f;
        float y = (yi + 0.5f) * scale - 0.5f;
        float z = (zi + 0.5f) * scale - 0.5f;

        float d = sqrt(x * x + y * y + z * z) - r;

        sdf.dist[xi + yi * rowExt + zi * sliceExt] = d;
      }
    }
  }
  return sdf;
}

inline float sdBox(glm::vec3 p, glm::vec3 b)
{
  glm::vec3 q = glm::abs(p) - b;
  return glm::length(glm::max(q, glm::vec3(0.0))) + glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

inline float mengerSponge(glm::vec3 p, float r)
{
  const int MENGER_ITERS = 3;

  float d = sdBox(p, glm::vec3(r));
  float res = d;

  float s = 1.0;
  for (int m = 0; m < MENGER_ITERS; m++)
  {
    
    glm::vec3 a = glm::mod(p * s, glm::vec3(2.0, 2.0, 2.0)) - glm::vec3(1.0);
    
    //float3 a = float3(fmod(p.x * s, 2.0) - 1.0, fmod(p.y * s, 2.0) - 1.0, fmod(p.z * s, 2.0) - 1.0);
    
    s *= 3.0;
    glm::vec3 r = glm::abs(glm::vec3(1.0) - 3.0f * glm::abs(a));

    float da = glm::max(r.x, r.y);
    float db = glm::max(r.y, r.z);
    float dc = glm::max(r.z, r.x);
    float c = (glm::min(da, glm::min(db, dc)) - 1.0) / s;

    if (c > d)
    {
      d = c;
      res = d;
    }
  }

  return res;
}

SDFDenseCPU create_menger_sponge(uint32_t dim, float r)
{
  SDFDenseCPU sdf {dim, dim, dim, {}};
  sdf.dist.resize(dim * dim * dim);

  float scale = 1.f/dim;

  uint32_t rowExt = dim;
  uint32_t sliceExt = rowExt * dim;

  for (uint32_t zi = 0; zi < dim; zi++)
  {
    for (uint32_t yi = 0; yi < dim; yi++)
    {
      for (uint32_t xi = 0; xi < dim; xi++)
      {
        float x = (xi + 0.5f) * scale - 0.5f;
        float y = (yi + 0.5f) * scale - 0.5f;
        float z = (zi + 0.5f) * scale - 0.5f;

        float d = mengerSponge(glm::vec3{x, y, z}, r);

        sdf.dist[xi + yi * rowExt + zi * sliceExt] = d;
      }
    }
  }
  return sdf;
}

static inline float sdOctahedron(glm::vec3 p, float s)
{
  p = glm::abs(p);
  return (p.x+p.y+p.z-s)*0.57735027;
}

SDFDenseCPU create_octahedron(uint32_t dim, float r)
{
  SDFDenseCPU sdf {dim, dim, dim, {}};
  sdf.dist.resize(dim * dim * dim);

  float scale = 1.f/dim;

  uint32_t rowExt = dim;
  uint32_t sliceExt = rowExt * dim;

  for (uint32_t zi = 0; zi < dim; zi++)
  {
    for (uint32_t yi = 0; yi < dim; yi++)
    {
      for (uint32_t xi = 0; xi < dim; xi++)
      {
        float x = (xi + 0.5f) * scale - 0.5f;
        float y = (yi + 0.5f) * scale - 0.5f;
        float z = (zi + 0.5f) * scale - 0.5f;

        float d = sdOctahedron(glm::vec3{x, y, z}, r);

        sdf.dist[xi + yi * rowExt + zi * sliceExt] = d;
      }
    }
  }
  return sdf;
}

SDFDenseCPU load_from_bin(const char *path)
{
  std::ifstream file{path, std::ios::ate|std::ios::binary};

  auto fileSize = file.tellg();
  file.seekg(0);

  assert(fileSize % sizeof(float) == 0);

  uint32_t numFloats = fileSize/sizeof(float);
  uint32_t estDim = powf(numFloats, 1.f/3.f);

  if (estDim * estDim * estDim != numFloats)
  {
    file.read((char *)(&numFloats), sizeof(numFloats));
    estDim = powf(numFloats, 1.f/3.f);
    assert(estDim * estDim * estDim == numFloats);
  }

  std::vector<float> buffer(numFloats); 
  
  file.read((char*)buffer.data(), buffer.size() * sizeof(float));
  file.close();

  SDFDenseCPU res {estDim, estDim, estDim, std::move(buffer)};
  return res;
}

std::unique_ptr<SDFDense> create_sdf_gpu(vkc::ContextPtr ctx, vk::Extent3D ext)
{
  auto imgUsage = vk::ImageUsageFlagBits::eSampled
    |vk::ImageUsageFlagBits::eTransferDst
    |vk::ImageUsageFlagBits::eTransferSrc
    |vk::ImageUsageFlagBits::eStorage;

  auto sdf = std::make_unique<SDFDense>();
  sdf->ext = ext;

  vk::ImageCreateInfo imgInfo {
    {},
    vk::ImageType::e3D, 
    vk::Format::eR32Sfloat, 
    ext, 1, 1, 
    vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
    imgUsage, vk::SharingMode::eExclusive
  };
  
  sdf->image = ctx->create_image(imgInfo);

  vk::ImageViewCreateInfo viewInfo {
    {}, {}, vk::ImageViewType::e3D, vk::Format::eR32Sfloat, {}, 
    vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
  };

  sdf->view = sdf->image->createView(viewInfo);

  return sdf;
}

void upload_sdf(vk::CommandBuffer cmd, SDFDense &dst, const SDFDenseCPU &src, vkc::BufferPtr staging)
{
  // todo: handle case when staging buffer is smaller then amount of data
  assert(staging->getSize()/sizeof(float) >= src.d * src.w * src.h);
  auto dstExt = dst.image->getInfo().extent;
  assert(src.w == dstExt.width && src.h == dstExt.height && src.d == dstExt.depth);

  {
    void *ptr = staging->map();
    std::memcpy(ptr, src.dist.data(), src.dist.size() * sizeof(float));
    staging->unmap();
  }

  auto ctx = dst.image->getContext();
  auto queue = ctx->mainQueue();

  vk::BufferImageCopy copyRegion {};
  copyRegion.setBufferOffset(0);
  copyRegion.setBufferRowLength(src.w);
  copyRegion.setBufferImageHeight(src.h);
  copyRegion.setImageSubresource(vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1});
  copyRegion.setImageOffset({0, 0, 0});
  copyRegion.setImageExtent(dst.ext);

  cmd.reset();
  cmd.begin(vk::CommandBufferBeginInfo{});

  vkc::image_layout_transition(cmd, dst.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTopOfPipe);
  cmd.copyBufferToImage(staging->apiBuffer(), dst.image->getImage(), vk::ImageLayout::eTransferDstOptimal, {copyRegion});
  vkc::image_layout_transition(cmd, dst.image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eBottomOfPipe);
  cmd.end();

  vk::SubmitInfo submitInf {};
  submitInf.setPCommandBuffers(&cmd);
  submitInf.setCommandBufferCount(1);

  queue.submit(submitInf);
  queue.waitIdle();

  cmd.reset();
}

Camera Camera::lookAt(glm::vec3 eye, glm::vec3 center, glm::vec3 up)
{
  Camera c {};
  c.pos = eye;
  c.z = glm::normalize(eye - center);
  c.x = glm::normalize(glm::cross(up, c.z));
  c.y = glm::normalize(glm::cross(c.z, c.x));
  return c;
}

SDFDenseRenderer::SDFDenseRenderer(vkc::ContextPtr ctx, vk::DescriptorPool pool)
{
  auto prog = ctx->loadComputeProgram("src/shaders/trace_dense_sdf.spv");
  pipeline = prog->makePipeline({});
  set = prog->allocDescSetUnique(pool);
}

void SDFDenseRenderer::render(vk::CommandBuffer cmd, const SDFRenderParams &params, vkc::ImageViewPtr sdfView, vkc::BufferPtr out_buffer)
{
  assert(out_buffer->getSize() >= params.outWidth * params.outHeight * sizeof(glm::vec4));

  auto ctx = std::static_pointer_cast<vkc::Context>(out_buffer->getContext());
  
  struct PushConsts
  {
    glm::vec4 camera_pos;
    glm::vec4 camera_x;
    glm::vec4 camera_y;
    glm::vec4 camera_z;

    float projDist; // distance to proj plane

    uint outWidth;
    uint outHeight;
    uint samplesPerPixel;
  
    float sdfAABBScale; // sdf is a cube in range [-scale, +scale] centered in (0, 0, 0)
    uint _pad0;
    uint _pad1;
    uint _pad2;
  } pc {
    .camera_pos = glm::vec4{params.camera.pos, 1.f},
    .camera_x = glm::vec4{params.camera.x, 0.f},
    .camera_y = glm::vec4{params.camera.y, 0.f},
    .camera_z = glm::vec4{params.camera.z, 0.f},
    .projDist = 1.f/(2.f * tanf(params.fovy * 0.5)),

    .outWidth = params.outWidth,
    .outHeight = params.outHeight,
    .samplesPerPixel = 1,

    .sdfAABBScale = params.sdfScale
  };

  pipeline->getProgram()->writeDescSet(*set, 0, {
    {0, vkc::BufferBinding {out_buffer}},
    {1, vkc::ImageBinding {ctx->getSampler(vkc::DEF_SMOOTH_SAMPLER), sdfView, vk::ImageLayout::eShaderReadOnlyOptimal}}
  });

  const uint32_t workGroupX = 8;
  const uint32_t workGroupY = 4;

  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->getPipeline());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline->getProgram()->getPipelineLayout(), 0, {*set}, {});
  cmd.pushConstants(pipeline->getProgram()->getPipelineLayout(), vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
  cmd.dispatch((pc.outWidth + workGroupX - 1)/workGroupX, (pc.outHeight + workGroupY - 1)/workGroupY, 1);
}

}