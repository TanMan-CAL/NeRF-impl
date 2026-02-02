# dataloader.py
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, image):
        self.image = torch.from_numpy(image).float() / 255.0
        self.H, self.W, _ = image.shape

        yy, xx = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="ij")
        coordinates = torch.stack([xx, yy], dim=-1).float()
        # print(coordinates.size())

        # normalize data
        for i in range(coordinates.shape[0]):
            for j in range(coordinates.shape[1]):
                coordinates[i, j, 0] = coordinates[i, j, 0] / self.W
                coordinates[i, j, 1] = coordinates[i, j, 1] / self.H

        self.coordinates = coordinates.view(-1, 2)
        self.colors = self.image.view(-1, 3)

    
    def __len__(self):
        return self.coordinates.shape[0]

    def __getitem__(self, idx):
        return self.coordinates[idx], self.colors[idx]

    def sample_batch(self, batch_size):
        idx = torch.randint(0, len(self), (batch_size))
        return self.coordinates[idx], self.colors[idx]


