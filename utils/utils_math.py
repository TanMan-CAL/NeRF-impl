# utils/utils_math.py
import math
import torch

def degrees_to_radians(degrees):
    return degrees * math.pi / 180.0

def reflect(v, n):
    return v - 2 * v.dot(n) * n

def linear_to_gamma(linear_component):
    if linear_component > 0:
        return math.sqrt(linear_component)
    return 0

def PSNR(MSE):
    return -10.0 * torch.log10(MSE) 
