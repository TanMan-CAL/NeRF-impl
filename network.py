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
    def __init__(self):
    
    def forward(self):
        return 