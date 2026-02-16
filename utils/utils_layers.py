# utils/utils_layers.py
import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super().__init__()
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        self.weight = nn.Parameter(torch.empty(out_dimension, in_dimension))
        self.bias = nn.Parameter(torch.empty(out_dimension))

        self.reset_parameters()

    def reset_parameters(self):
        # kaiming / fan-in
        bound = 1 / math.sqrt(self.in_dimension)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return x @ self.weight.T + self.bias

class ReLU(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=0.0)

class Sigmoid(nn.Module):
    def forward(self, x):
        return 1.0 / (1.0 + torch.exp(-x))
