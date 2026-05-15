#src/network.py
import torch
import torch.nn as nn


class SPE(torch.nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, x):
        encoded = [x]
        for i in range(self.L):
            w = (2 ** i) * torch.pi
            encoded.append(torch.sin(w * x))
            encoded.append(torch.cos(w * x))
        return torch.cat(encoded, dim=-1)


class NeRF(nn.Module):
    """skip connection at layer 5"""

    def __init__(self, pos_L=10, dir_L=4, width=256):
        super().__init__()
        self.pos_encoder = SPE(pos_L)
        self.dir_encoder = SPE(dir_L)

        pos_dimension = 3 * (2 * pos_L + 1)
        dir_dimension = 3 * (2 * dir_L + 1)

        self.pts_linears = nn.ModuleList(
            [nn.Linear(pos_dimension, width)]
            + [nn.Linear(width, width)] * 4
            + [nn.Linear(width + pos_dimension, width)]
            + [nn.Linear(width, width)] * 2
        )

        self.sigma_layer = nn.Linear(width, 1)
        self.feature_layer = nn.Linear(width, width)
        self.views_layer = nn.Linear(width + dir_dimension, width // 2)
        self.rgb_layer = nn.Linear(width // 2, 3)

    def forward(self, x, d):
        x_encoded = self.pos_encoder(x)
        d_encoded = self.dir_encoder(d)

        h = x_encoded
        for i, layer in enumerate(self.pts_linears):
            if i == 5:
                h = torch.cat([h, x_encoded], dim=-1)
            h = torch.relu(layer(h))

        sigma = torch.relu(self.sigma_layer(h))

        feature = self.feature_layer(h)
        h = torch.cat([feature, d_encoded], dim=-1)
        h = torch.relu(self.views_layer(h))
        rgb = torch.sigmoid(self.rgb_layer(h))

        return rgb, sigma


class NeuralField(torch.nn.Module):
    def __init__(self, L=10, width=256, depth=4):
        super().__init__()
        layers = []
        input_dimension = 2 + (4 * L)
        self.spe = SPE(L=L)

        for _ in range(depth):
            layers.append(torch.nn.Linear(input_dimension, width))
            layers.append(torch.nn.ReLU())
            input_dimension = width

        layers.append(torch.nn.Linear(width, 3))
        layers.append(torch.nn.Sigmoid())
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.spe(x)
        return self.model(x)
