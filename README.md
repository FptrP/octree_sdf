ToDo:

1. Trace simple sdf grids: 

- add stbi_image write/read 
- add shader pass that fills out buffer with some color
- add Camera, projection params. Draw AABB of sdf
- Trace sdf

2. Add oct tree sdf
- Simple algorithm to build tree representation: if values 8 blocks have same sign - merge
- Add virtual texture class. allocate and map memory pages according to oct tree structure
- Implement copy
- Add trace pipeline for it 


External dependencies:
- Vulkan
- glm