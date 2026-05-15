#src/geometry.py
import numpy as np


class Geometry:
    @staticmethod
    def transform(camera2world, points):
        ones = np.ones((*points.shape[:-1], 1), dtype=np.float32)
        points_1s = np.concatenate([points, ones], axis=-1)
        points_rotat_trans = points_1s @ camera2world.T
        return points_rotat_trans[..., :3]

    @staticmethod
    def pixel_to_camera(K, uv, depth=1.0):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x = (uv[..., 0] - cx) / fx * depth
        y = (uv[..., 1] - cy) / fy * depth
        z = np.ones_like(x) * depth

        return np.stack([x, y, z], axis=-1).astype(np.float32)

    @staticmethod
    def pixel_to_ray(K, camera2world, uv):
        ray_origin = camera2world[:3, 3]
        points_camera = Geometry.pixel_to_camera(K, uv, depth=1.0)
        points_world = Geometry.transform(camera2world, points_camera)

        ray_direction = points_world - ray_origin
        ray_direction = ray_direction / np.linalg.norm(ray_direction, axis=-1, keepdims=True)

        return ray_origin.astype(np.float32), ray_direction.astype(np.float32)
