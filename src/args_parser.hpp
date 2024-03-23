#ifndef ARGS_PARSER_HPP_INCLUDED
#define ARGS_PARSER_HPP_INCLUDED

#include "sdf/sdf.hpp"

#include <optional>
#include <cinttypes>
#include <algorithm>

struct AppArgs
{
  AppArgs(int argc, char **argv)
    : args(argv, argv + argc)
  {
    glm::vec3 eye {0.f, 0.0f, -2.0f};
    glm::vec3 center {0.f, 0.f, 0.f};
    glm::vec3 up {0.f, 1.f, 0.f};

    if (auto arr = parseRealArray("-eye", 3))
      eye = glm::vec3{arr->at(0), arr->at(1), arr->at(2)};
    
    if (auto arr = parseRealArray("-center", 3))
      center = glm::vec3{arr->at(0), arr->at(1), arr->at(2)};
    
    if (auto arr = parseRealArray("-up", 3))
      up = glm::vec3{arr->at(0), arr->at(1), arr->at(2)};

    camera = sdf::Camera::lookAt(eye, center, up);
    
    if (auto v = parseInt("-width"))
      width = *v;

    if (auto v = parseInt("-height"))
      height = *v;
    
    if (auto v = parseReal("-fovy"))
      fovy = glm::radians(*v);
    
    if (auto v = parseString("-model"))
      modelName = *v;
    
    if (auto v = parseInt("-spp"))
      numSPP = *v;

    if (auto v = parseReal("-scale"))
      modelScale = *v;
    
    if (auto v = parseString("-mode"))
    {
      assert(*v == "sparse" || *v == "dense");
      modelType = *v;
    }
  }


  sdf::Camera camera = sdf::Camera::lookAt({0.f, 0.0f, -2.0f}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});

  uint32_t width = 1024;
  uint32_t height = 1024;
  float fovy = glm::radians(90.f);
  
  uint32_t numSPP = 1;

  std::string modelName = "sample_models/chair_grid_32.bin";
  std::string modelType = "dense";

  float modelScale = 1.f;


private:
  std::vector<std::string> args;


  std::optional<int64_t> parseInt(const std::string &name)
  {
    auto it = std::find(args.begin(), args.end(), name);
    if (it == args.end())
      return std::nullopt;

    return std::stoll(args.at(std::distance(args.begin(), it) + 1));
  }

  std::optional<std::string> parseString(const std::string &name)
  {
    auto it = std::find(args.begin(), args.end(), name);
    if (it == args.end())
      return std::nullopt;
    return args.at(std::distance(args.begin(), it) + 1);
  }

  std::optional<double> parseReal(const std::string &name)
  {
    auto it = std::find(args.begin(), args.end(), name);
    if (it == args.end())
      return std::nullopt;
    return std::stod(args.at(std::distance(args.begin(), it) + 1));
  }

  std::optional<std::vector<double>> parseRealArray(const std::string &name, uint32_t num_elems)
  {
    auto it = std::find(args.begin(), args.end(), name);
    if (it == args.end())
      return std::nullopt;

    std::vector<double> result;
    result.reserve(num_elems);

    auto offset = std::distance(args.begin(), it);

    for (uint32_t i = 0; i < num_elems; i++)
      result.push_back(std::stod(args.at(offset + 1 + i)));

    return std::move(result);
  }


};

#endif