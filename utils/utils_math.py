import math
import torch
import numpy as np


def degrees_to_radians(degrees):
    return degrees * math.pi / 180.0


def reflect(v, n):
    return v - 2 * v.dot(n) * n


def linear_to_gamma(linear_component):
    if linear_component > 0:
        return math.sqrt(linear_component)
    return 0


def PSNR(MSE, eps=1e-10):
    return -10.0 * torch.log10(MSE + eps)


def marker_corners_world(center_xyz, size_m, yaw_degrees=0.0):
    half = size_m * 0.5
    corners = np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float32,
    )

    yaw = np.deg2rad(yaw_degrees)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return corners @ rotation.T + np.asarray(center_xyz, dtype=np.float32)


def estimate_near_far(c2ws, marker_layout_by_id, marker_size_m, near_min=0.01):
    points = []
    for config in marker_layout_by_id.values():
        points.append(
            marker_corners_world(
                center_xyz=config["center_xyz"],
                size_m=marker_size_m,
                yaw_degrees=float(config.get("yaw_degrees", 0.0)),
            )
        )

    points = np.concatenate(points, axis=0)
    camera_centers = c2ws[:, :3, 3]

    distances = []
    for camera_center in camera_centers:
        d = np.linalg.norm(points - camera_center[None, :], axis=-1)
        distances.append(d)

    distances = np.concatenate(distances, axis=0)
    near = float(max(near_min, np.percentile(distances, 2) * 0.8))
    far = float(np.percentile(distances, 98) * 1.2)

    if far <= near:
        far = near + 1.0

    return near, far
