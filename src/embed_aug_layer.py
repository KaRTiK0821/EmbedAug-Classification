import torch
import torch.nn as nn

class EmbedAug(nn.Module):
    def __init__(self, mask_prob=0.1, noise_std=0.1, mode='gaussian'):
        super().__init__()
        self.mask_prob = mask_prob
        self.noise_std = noise_std
        self.mode = mode

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.rand_like(x[:, :, 0]) < self.mask_prob
        mask = mask.unsqueeze(-1)
        if self.mode == 'zero':
            x = x.masked_fill(mask, 0.0)
        else:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.where(mask, noise, x)
        return x
