Build:
- cmake -S ./ -B ./build
- cd build
- make -j 4

- Command line parameters (see args_parser.hpp)
  - -eye <float> <float> <float> 
  - -center <float> <float> <float>
  - -up <float> <float> <float>
  - -model <path> - model to load 
  - -mode <sparse|dense> - trace mod

  
External dependencies:
- Vulkan
- glm