import sys
import struct
import os

from mesh_to_sdf import mesh_to_voxels
import trimesh


if __name__ == '__main__':
  assert len(sys.argv) >= 3
  
  sdf_in = sys.argv[1]
  voxel_res = int(sys.argv[2])
  out_name = sys.argv[3]

  mesh = trimesh.load(sdf_in)

  print("Mesh loaded, generating voxel sdf....")

  voxels = mesh_to_voxels(mesh, voxel_res, pad=False) # outputs cube [-1; 1]^3
  voxels = voxels * 0.5 # resize to [-0.5, 0.5]
  
  out_res = voxels.shape[0]
  base_dir = os.path.split(sdf_in)[0]
  out_path = os.path.join(base_dir, f"{out_name}_{out_res}.bin")
  
  print("Voxel sdf generated. Saving to ", out_path)

  with open(out_path, "wb") as out_file:  
    for z in range(out_res):
      for y in range(out_res):
        for x in range(out_res):
          out_file.write(struct.pack("<f", voxels[x, y, z]))
  
  print("Done")

  




