import torch.nn as nn
import torch.nn.functional as F
import torch

class AlzheimersClassification3DCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self._conv_layers = nn.Sequential(
            nn.Conv3d(1, 8, 5),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 5),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3),
            nn.ReLU(),
            nn.AdaptiveMaxPool3d((4, 4, 4))
        )
        self._fc1 = nn.Linear(4 * 4 * 4 * 32, 256)
        self._fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self._conv_layers(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self._fc1(x))
        x = self._fc2(x)
        x = torch.flatten(x)
        return x

# TODO: Swin transformer based classifier