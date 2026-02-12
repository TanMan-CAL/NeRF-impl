# src/volume_rendering.py
import torch

# from https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/volume-rendering-summary-equations.html
class VolumeRender: # basic implementation nothing to do with NeRF logic
    def __init__(self, step_size):
        self.step_size = step_size

    def render(self, sigmas, rgbs):

        # alpha compositing: converts density to opacity
        alpha = 1.0 - torch.exp(-sigmas*self.step_size)
        eps = 1e-10

        # transmittance: how much light has been blocked up to point t
        transmittance = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + eps],dim=1), dim=1)[:, :-1]

        # weights: each sample to final color
        weights = transmittance * alpha
        
        # final color: weighted sum of all colors along the ray
        rendered_color = torch.sum(weights * rgbs, dim=1)

        return rendered_color, weights