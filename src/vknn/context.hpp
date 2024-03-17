#ifndef VKNN_CONTEXT_HPP_INCLUDED
#define VKNN_CONTEXT_HPP_INCLUDED

#include <vkc.hpp>

#include "array.hpp"

namespace vknn
{

class Node
{
  virtual void forward() = 0;
  virtual void backward() = 0;

  std::vector<BaseArray> &get_parameters(); // for optimizer 
  std::vector<BaseArray> &get_gradient(); //  

protected:

  //registerParam()

private:

};


}


#endif