#ifndef COMMON_GLSL_INCLUDED
#define COMMON_GLSL_INCLUDED

struct Camera
{
  vec3 pos;
  vec3 x;
  vec3 y;
  vec3 z;
};

vec3 screen_to_world_dir(vec2 uv, in Camera camera, float aspect, float proj_dist)
{
  const float y_camera = (0.5f - uv.y);
  const float x_camera = aspect * (uv.x - 0.5f);
  return normalize(x_camera * camera.x + y_camera * camera.y - proj_dist * camera.z);
}

// if tNear < tFar no intersection
vec2 ray_aabb_intersection(in vec3 ray_origin, in vec3 ray_dir, vec3 aabb_min, vec3 aabb_max)
{
  vec3 tMin = (aabb_min - ray_origin)/ray_dir;
  vec3 tMax = (aabb_max - ray_origin)/ray_dir;

  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax); 

  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar = min(min(t2.x, t2.y), t2.z);
  return vec2(tNear, tFar);
}


#endif