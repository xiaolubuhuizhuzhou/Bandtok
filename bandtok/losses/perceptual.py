# VGG-based perceptual loss on log-Mel spectrograms.
#
# `VGGFeatureExtractor` / `logmel_to_vgg_img` are copied and modified from
# https://github.com/crowsonkb/vgg_loss/blob/master/vgg_loss.py under MIT License.
#
# `VGGL1Loss` is copied and modified from
# https://github.com/Stability-AI/stable-audio-tools under MIT License.
# Copyright (c) 2023 Stability AI. See LICENSES/LICENSE.stable-audio-tools.txt
"""A VGG-based perceptual loss function for PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from .base import LossModule


class VGGFeatureExtractor(nn.Module):
    """
    Extract intermediate feature maps from VGG16/VGG19 features.
    """
    # Common "relu" tap points for torchvision.vgg16.features
    # vgg16: relu1_2=3, relu2_2=8, relu3_3=15, relu4_3=22, relu5_3=29
    VGG16_RELU = {
        "relu1_1": 1, "relu1_2": 3,
        "relu2_1": 6, "relu2_2": 8,
        "relu3_1": 11, "relu3_2": 13, "relu3_3": 15,
        "relu4_1": 18, "relu4_2": 20, "relu4_3": 22,
        "relu5_1": 25, "relu5_2": 27, "relu5_3": 29,
    }

    # vgg19: relu1_2=3, relu2_2=8, relu3_4=17, relu4_4=26, relu5_4=35 (torchvision layout)
    VGG19_RELU = {
        "relu1_1": 1, "relu1_2": 3,
        "relu2_1": 6, "relu2_2": 8,
        "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
        "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
        "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
    }

    def __init__(
        self,
        model: str = "vgg16",
        layers=("relu2_2", "relu3_3", "relu4_3"),
        use_input_imagenet_norm: bool = False,
    ):
        """
        model: 'vgg16' or 'vgg19'
        layers: iterable of layer names (recommended) or integer indices into vgg.features
        use_input_imagenet_norm: whether to apply ImageNet mean/std normalization.
          - If your spectrogram is already normalized in [0,1], setting this True mimics standard VGG usage.
          - Many audio papers omit mean/std and just use [0,1] scaling; you can keep it False.
        """
        super().__init__()
        assert model in ("vgg16", "vgg19")

        if model == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
            name2idx = self.VGG16_RELU
        else:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
            name2idx = self.VGG19_RELU

        self.vgg = vgg.eval()
        self.vgg.requires_grad_(False)

        # resolve layers to sorted indices
        idxs = []
        for l in layers:
            if isinstance(l, int):
                idxs.append(l)
            elif isinstance(l, str):
                if l not in name2idx:
                    raise ValueError(f"Unknown layer name '{l}'. Available keys: {sorted(name2idx.keys())[:8]}...")
                idxs.append(name2idx[l])
            else:
                raise TypeError(f"layers must be int or str, got {type(l)}")
        self.layer_idxs = sorted(set(idxs))

        self.use_input_imagenet_norm = use_input_imagenet_norm
        if use_input_imagenet_norm:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean, persistent=False)
            self.register_buffer("std", std, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        x: [B,3,H,W] (ideally in [0,1] if not using mean/std; otherwise will be normalized)
        Return: list of feature maps at specified layers, in the same order as self.layer_idxs
        """
        if self.use_input_imagenet_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        feats = []
        for i, m in enumerate(self.vgg):
            x = m(x)
            if i in self.layer_idxs:
                feats.append(x)
        return feats


def logmel_to_vgg_img(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    """
    x: [B,1,H,W] log-mel (or mel-like) arbitrary range
    return: [B,3,H,W] in [0,1]
    """
    # per-sample min-max (avoid scale mismatch)
    x_min = x.amin(dim=(2,3), keepdim=True)
    x_max = x.amax(dim=(2,3), keepdim=True)
    x = (x - x_min) / (x_max - x_min + eps)
    x = x.clamp(0.0, 1.0)
    return x.repeat(1, 3, 1, 1)


class VGGL1Loss(LossModule):
    def __init__(self, key_a: str, key_b: str,
                 weight: float=1.0, name: str='vgg_l1_loss', decay=1.0,
                 model: str = "vgg16",
                 layers=("relu2_2", "relu3_3", "relu4_3"),
                 layer_weights=[1,1,1],
                 use_input_imagenet_norm: bool = False
                 ):
        super().__init__(name=name, weight=weight, decay=decay)
        self.key_a = key_a
        self.key_b = key_b

        self.feat = VGGFeatureExtractor(
            model=model,
            layers=layers,
            use_input_imagenet_norm=use_input_imagenet_norm,
        )
        self.layer_weights = layer_weights

    def forward(self, info):
        target = info[self.key_a]
        input = info[self.key_b]
        target = logmel_to_vgg_img(target)
        input = logmel_to_vgg_img(input)

        feats_input = self.feat(input)
        #  gradients should flow to spec_hat through x only.
        with torch.no_grad():
            feats_target = self.feat(target)

        loss = input.new_zeros(())
        for w, fx, fy in zip(self.layer_weights, feats_input, feats_target):
            loss = loss + w * F.l1_loss(fx, fy, reduction="mean")

        return self.weight * loss
