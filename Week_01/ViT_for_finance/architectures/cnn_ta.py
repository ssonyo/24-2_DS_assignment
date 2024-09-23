import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):
    def __init__(self, dim, depth, kernel_size=7, patch_size=2):
        super(CNN, self).__init__()
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim),
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dim, 3)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)