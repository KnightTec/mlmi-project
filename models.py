import torch.nn as nn
import torch.nn.functional as F
import torch

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT

class AlzheimersClassification3DCNN(nn.Module):

    ## based on this paper: https://arxiv.org/pdf/2007.13224.pdf

    def __init__(self):
        super().__init__()
        self._conv_layers = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(8),

            nn.Conv3d(8, 8, 3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(8),

            nn.Conv3d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.BatchNorm3d(16),

            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        self._fc1 = nn.Linear(4 * 4 * 4 * 16, 256)
        self._dropout = nn.Dropout(0.2)
        self._fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self._conv_layers(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self._fc1(x))
        x = self._dropout(x)
        x = self._fc2(x)
        x = torch.flatten(x)
        return x

class SvinViTAlzheimersClassifier(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._svin_vit = SwinViT(
            in_chans=1, 
            embed_dim=48,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self._global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self._fc = nn.Linear(768, 1)

    def forward(self, x_in):
        hidden_states_out = self._svin_vit(x_in, True)[4]
        x = self._global_avg_pool(hidden_states_out)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self._fc(x)
        x = torch.flatten(x)
        return x
    
class SvinViTAlzheimersClassifierSmall(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._svin_vit = SwinViT(
            in_chans=1, 
            embed_dim=8,
            window_size=(5, 5, 5),
            patch_size=(2, 2, 2),
            depths=[2, 2, 2, 2],
            num_heads=[2, 4, 8, 16],
            mlp_ratio=2.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self._global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self._fc = nn.Linear(128, 1)

    def forward(self, x_in):
        hidden_states_out = self._svin_vit(x_in, True)[4]
        x = self._global_avg_pool(hidden_states_out)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self._fc(x)
        x = torch.flatten(x)
        return x
    