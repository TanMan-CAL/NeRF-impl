import math
import torch
# Constants
INFINITY = float('inf')
PI = 3.1415926535897932385

# Utility functions
def degrees_to_radians(degrees):
    return degrees * PI / 180.0

def reflect(v, n):
    return v - 2 * v.dot(n) * n

def refract(uv, n, etai_over_etat):
    cos_theta = min(-uv.dot(n), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -math.sqrt(abs(1.0 - r_out_perp.length_squared())) * n
    return r_out_perp + r_out_parallel

def linear_to_gamma(linear_component):
    if linear_component > 0:
        return math.sqrt(linear_component)
    return 0

def PSNR(MSE):
    return -10.0 * torch.log10(MSE) 
