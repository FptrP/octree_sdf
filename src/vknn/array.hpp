#ifndef VKNN_ARRAY_HPP_INCLUDED
#define VKNN_ARRAY_HPP_INCLUDED

#include <vkc.hpp>

namespace vknn
{

enum ArrayType
{
  Cpu,
  Gpu
};

class BaseArray
{

  //const std::vector<uint32_t> dims() const { return dims_; }
  
  uint32_t length(uint32_t dimension) const { return dims_.at(dimension); }
  uint32_t dims() const { return dims_.size(); }

private:
  std::vector<uint32_t> dims_;
};

template <typename T> 
class CpuArray : BaseArray
{

};

template <typename T>
class GpuArray : BaseArray
{

};


}

#endif
