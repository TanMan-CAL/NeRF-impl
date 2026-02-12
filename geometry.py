# geometry.py

import numpy as np

class Geometry:
    def transform(camera2world, points):
        # transform formula: world pos = R * camera pos + t
        ones = np.ones((*points.shape[:-1], 1))
        points_1s = np.concatenate([points, ones], axis=-1)

        points_rotat_trans = points_1s @ camera2world.T

        return points_rotat_trans[..., :3]

    def pixel_to_camera(K, uv, depth=1):
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

        x = (uv[...,0] - cx) / fx * depth
        y = (uv[...,1] - cy) / fy * depth
        z = np.ones_like(x) * depth

        return np.stack([x, y, z], axis=-1)
    
    def pixel_to_ray(K, camera2world, uv):
        ray_origin = camera2world[:3, 3]

        point = Geometry.pixel_to_camera(K, uv, depth=1.0)

        x = Geometry.transform(camera2world, point)

        ray_direction = x - ray_origin

        ray_direction = ray_direction / np.linalg.norm(ray_direction, axis=-1, keepdims=True)

        return ray_origin, ray_direction
