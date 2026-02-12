# src/rays_data.py
import numpy as np
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
        uv = uv + 0.5  # pixel center offset; NEED TO DOUBLE CHECK THIS (ASK ZAC)

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
            origin, distance = Geometry.pixel_to_ray(self.K, self.camera2world[cam_id], u)
            rays_origin.append(origin)
            rays_direction.append(distance)

        return (np.stack(rays_origin), np.stack(rays_direction), rgb)

    def sample_along_rays(self, rays_origin, rays_direction, near, far, n_samples, perturbation=False):
        random_samples = np.linspace(near, far, n_samples)

        if perturbation:
            interval_size = (far - near) / n_samples
            random_samples = random_samples + (np.random.rand(*random_samples.shape)*interval_size)  # to prevent overfitting
        
        # sample rays for NeRF
        # x(t) = o + td
        points = (rays_origin[:, None, :] + (rays_direction[:, None, :] * random_samples[None, :, None]))
        return points