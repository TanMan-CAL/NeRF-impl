#src/rays_data.py
import numpy as np
import torch
from geometry import Geometry


class RaysData:
    def __init__(self, images, K, camera2world):
        self.images = images
        self.K = K
        self.camera2world = camera2world

        N, H, W, _ = images.shape

        yy, xx = np.meshgrid(
            np.arange(H),
            np.arange(W),
            indexing="ij"
        )

        uv = np.stack([xx, yy], axis=-1).reshape(-1, 2)
        uv = uv + 0.5

        self.uvs = uv
        self.pixels = images.reshape(-1, 3)
        self.image_ids = np.repeat(np.arange(N), H * W)

    def sample_rays(self, n_rays):
        ray = np.random.choice(len(self.uvs), n_rays, replace=False)
        uv = self.uvs[ray]
        rgb = self.pixels[ray]
        camera_ids = self.image_ids[ray]

        rays_origin, rays_direction = [], []

        for u, cam_id in zip(uv, camera_ids):
            origin, direction = Geometry.pixel_to_ray(self.K, self.camera2world[cam_id], u)
            rays_origin.append(origin) 
            rays_direction.append(direction)

        return (np.stack(rays_origin), np.stack(rays_direction), rgb)

    def sample_along_rays(self, rays_origin, rays_direction, near, far, n_samples, perturbation=False, return_directions=False, return_t=False):
        """Sample points along each ray: x(t) = o + t*d

        Takes and returns torch tensors, and every intermediate is built on the
        same device as `rays_origin`, so a training step never moves ray data
        off the GPU. dtype follows the inputs rather than defaulting to float64.
        """
        device, dtype = rays_origin.device, rays_origin.dtype
        t = torch.linspace(near, far, n_samples, device=device, dtype=dtype)

        if perturbation:
            interval_size = (far - near) / n_samples
            t = t + torch.rand(n_samples, device=device, dtype=dtype) * interval_size

        points = rays_origin[:, None, :] + (rays_direction[:, None, :] * t[None, :, None])

        outputs = [points]
        if return_directions:
            # a view, not a copy; the later reshape materializes it if it has to
            outputs.append(rays_direction[:, None, :].expand_as(points))

        if return_t:
            outputs.append(t)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)
