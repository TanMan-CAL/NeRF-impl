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
        
        torch.cat(encoded, -1)


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