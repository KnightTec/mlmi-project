import torch.nn as nn
import torch.nn.functional as F
import torch

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT, MERGING_MODE
from monai.utils import look_up_option

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


class SwinVitPretrainedClassifier(nn.Module):

    # TODO: adaptive avg pool + linear layer

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        depths = (2, 2, 2, 2)
        num_heads = (3, 6, 12, 24)
        feature_size: int = 48
        drop_rate: float = 0.0
        attn_drop_rate: float = 0.0
        dropout_path_rate: float = 0.0
        use_checkpoint: bool = False
        spatial_dims: int = 3
        downsample="merging"

        # conv3d to expand to 4 features (HACK)
        self._conv = nn.Conv3d(1, 4, 3)

        self.swinViT = SwinViT(
            in_chans=4,
            embed_dim=feature_size,
            window_size=(7, 7, 7),
            patch_size=(2, 2, 2),
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        )
        # freeze the pretrained layers
        self.swinViT.requires_grad_(False)

        self._global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        feature_vector_size = 768
        self._fc0 = nn.Linear(feature_vector_size, 128)
        self._fc1 = nn.Linear(128, 1)

    def forward(self, x_in):
        x = self._conv(x_in)

        # x = x_in.repeat(1, 4, 1, 1, 1) 

        # TODO: look at the output of the convolutional layers; might be interesting

        # extract lowest level features
        hidden_states_out = self.swinViT(x, True)[4]
        x = self._global_avg_pool(hidden_states_out)
        x = torch.flatten(x, 1)
        x = F.relu(self._fc0(x))
        x = self._fc1(x)
        x = torch.flatten(x)
        return x

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )