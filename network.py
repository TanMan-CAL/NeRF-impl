import torch

class SPE(torch.nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
    
    def forward(self, x):
        encoded = [x]

        for i in range(self.L):
            encoded.append(torch.sin((2 ** i) * torch.pi * x))
            encoded.append(torch.cos((2 ** i) * torch.pi * x))
        
        torch.cat(encoded, dim=-1)


class NeuralField(torch.nn.Module):
    def __init__(self, spe_dim, width=256, depth=4):
        super().__init__()
        layers = []
        input_dim = spe_dim
        
        for __ in range(depth):
            layers.append(torch.nn.Linear(input_dim, width))
            layers.append(torch.nn.ReLU())
            input_dim = width
        
        layers.append(torch.nn.Linear(width, 3)) # r g b
        layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DataLoader(torch.utils.data.Dataset):
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
        self.colours = self.image.view(-1, 3)

    
    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.colors[idx]

    def sample_batch(self, dataset, batch_size):
        idx = torch.randint(0, len(dataset), (batch_size))
        coords = dataset.coords[idx]
        colors = dataset.colors[idx]
        return coords, colors



