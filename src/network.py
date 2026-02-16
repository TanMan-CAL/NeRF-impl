# src/network.py
import torch
import torch.nn as nn

class SPE(torch.nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
    
    def forward(self, x):
        encoded = [x]

        for i in range(self.L):
            encoded.append(torch.sin((2 ** i) * torch.pi * x))
            encoded.append(torch.cos((2 ** i) * torch.pi * x))
        
        return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
    def __init__(self, pos_L=10, dir_L=4, width=256):
        super().__init__()

        self.pos_encoder= SPE(pos_L)
        self.dir_encoder = SPE(dir_L)
        pos_dimension = 3 * (2 * pos_L + 1)
        dir_dimension = 3 * (2 * dir_L + 1)
            
        self.fc1 = nn.Linear(pos_dimension, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)

        self.sigma_layer = nn.Linear(width, 1)

        self.color_layer = nn.Sequential(
            nn.Linear(width + dir_dimension, width // 2),
            nn.ReLU(),
            nn.Linear(width // 2, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x, d):
        x = self.pos_enc(x)
        d = self.dir_enc(d)

        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))

        sigma = torch.relu(self.sigma_layer(h))

        h = torch.cat([h, d], dim=-1)
        rgb = self.color_layer(h)

        return rgb, sigma


class NeuralField(torch.nn.Module):
    def __init__(self, L=10, width=256, depth=4):
        super().__init__()
        layers = []
        input_dimension = 2 + (4 * L)
        self.spe = SPE(L=L)

        
        for __ in range(depth):
            layers.append(torch.nn.Linear(input_dimension, width))
            layers.append(torch.nn.ReLU())
            input_dimension = width
        
        layers.append(torch.nn.Linear(width, 3)) # r g b
        layers.append(torch.nn.Sigmoid())

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.spe(x)
        return self.model(x)