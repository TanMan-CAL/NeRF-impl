#src/dataloader.py
import torch

class RaysDataLoader:
    def __init__(self, rays_data):
        self.rays_data = rays_data

    def sample_rays(self, n_rays):
        rays_origin, rays_direction, pixels = self.rays_data.sample_rays(n_rays)
        return rays_origin, rays_direction, pixels

    def sample_along_rays(self, rays_origin, rays_direction, near, far, n_samples, perturbation=False, return_directions=False, return_t=False):
        return self.rays_data.sample_along_rays(rays_origin, rays_direction, near, far, n_samples, perturbation=perturbation, return_directions=return_directions, return_t=return_t)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image):
        self.image = torch.from_numpy(image).float() / 255.0
        self.H, self.W, _ = image.shape

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        coordinates = torch.stack([xx, yy], dim=-1).float()

        coordinates[..., 0] = coordinates[..., 0] / self.W
        coordinates[..., 1] = coordinates[..., 1] / self.H

        self.coordinates = coordinates.view(-1, 2)
        self.colors = self.image.view(-1, 3)

    def __len__(self):
        return self.coordinates.shape[0]

    def __getitem__(self, idx):
        return self.coordinates[idx], self.colors[idx]

    def sample_batch(self, batch_size):
        idx = torch.randint(0, len(self), (batch_size,))
        return self.coordinates[idx], self.colors[idx]
