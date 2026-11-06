# Multi-scale PatchGAN discriminator operating on log-Mel spectrograms.
#
# The PatchGAN discriminator design follows Isola et al., "Image-to-Image Translation
# with Conditional Adversarial Networks" (CVPR 2017). The multi-scale + feature-matching
# training scheme follows common neural-audio-codec practice (MelGAN / HiFi-GAN / DAC).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from math import log2
from torch.utils.checkpoint import checkpoint


def _create_custom_forward(module):
    def custom_forward(*inputs):
        return module(*inputs)
    return custom_forward


def _checkpoint_call(module, x, use_ckpt: bool):
    if use_ckpt:
        return checkpoint(_create_custom_forward(module), x, use_reentrant=False)
    return module(x)


class MelPatchGAN2D(nn.Module):
    """
    Single 2D PatchGAN for mel spectrograms.
    Input:  [B, C, M, T]
    Output: logits [B, 1, m', t'], feats: list[tensor]
    """
    def __init__(
        self,
        in_channels: int = 1,
        ndf: int = 64,
        max_down: int = 16,
        use_ckpt: bool = False,
    ):
        super().__init__()
        self.use_ckpt = use_ckpt

        n_layers = max(1, int(log2(max_down)))

        self.input_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        blocks = []
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 16)
            blocks.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            ))
        self.blocks = nn.ModuleList(blocks)

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.out_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4, f"expected [B,C,M,T], got {tuple(x.shape)}"
        fmaps = []

        h = _checkpoint_call(self.input_layer, x, self.use_ckpt)
        fmaps.append(h)

        for blk in self.blocks:
            h = _checkpoint_call(blk, h, self.use_ckpt)
            fmaps.append(h)

        fmaps.append(self.out_layer(h))
        return fmaps


class MultiScaleMelPatchGAN(nn.Module):
    """
    Multiple 2D PatchGAN discriminators on mel spectrograms.

    Input:  x [B, C, M, T] (typically C=1, x in log-mel domain)
    Output: fmaps = list of (logits, feats) from each sub-discriminator
    """
    def __init__(
        self,
        channels: int = 1,
        scales: list[int] = (1, 2, 4),     # multi-scale downsample factors
        ndf: int = 64,
        max_downs: list[int] = (16, 16, 16),  # one per scale; controls #layers (log2)
        use_ckpt: bool = False,
        preprocess: str = "logmel",        # "logmel" or "linear"
        logmel_clamp: tuple[float, float] = (-12.0, 6.0),
    ):
        super().__init__()
        assert len(max_downs) == len(scales), "max_downs must match scales length"
        self.scales = list(scales)
        self.use_ckpt = use_ckpt
        self.preprocess_mode = preprocess
        self.logmel_clamp = logmel_clamp

        discs = []
        for s, md in zip(self.scales, max_downs):
            discs.append(MelPatchGAN2D(
                in_channels=channels,
                ndf=ndf,
                max_down=md,
                use_ckpt=use_ckpt
            ))
        self.discriminators = nn.ModuleList(discs)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,M,T]
        if self.preprocess_mode == "logmel":
            # 1) clamp to avoid extreme values
            lo, hi = self.logmel_clamp
            x = x.clamp(lo, hi)
            # 2) per-sample, per-channel mean removal (stabilizes)
            x = x - x.mean(dim=(-1, -2), keepdim=True)
            return x
        elif self.preprocess_mode == "linear":
            # linear mel magnitude: remove DC-like offset and peak normalize
            x = x - x.mean(dim=(-1, -2), keepdim=True)
            denom = x.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-9)
            x = 0.8 * x / denom
            return x
        else:
            raise ValueError(f"Unknown preprocess_mode={self.preprocess_mode}")

    def _downsample_2d(self, x: torch.Tensor, scale: int) -> torch.Tensor:
        if scale == 1:
            return x
        # avg pool on (M,T)
        return F.avg_pool2d(x, kernel_size=scale, stride=scale)

    def forward(self, x: torch.Tensor):
        x = self.preprocess(x)
        fmaps = []
        for disc, s in zip(self.discriminators, self.scales):
            x_s = self._downsample_2d(x, s)
            # checkpoint the whole discriminator call if requested
            if self.use_ckpt:
                out = checkpoint(_create_custom_forward(disc), x_s, use_reentrant=False)
            else:
                out = disc(x_s)
            fmaps.append(out)
        return fmaps


# Backwards-compatible alias: older checkpoints / configs refer to this class
# by its previous name.
MelDACDiscriminator = MultiScaleMelPatchGAN


class MelPatchGANLoss(nn.Module):

    def __init__(self, use_hinge: bool = False, **discriminator_kwargs):
        super().__init__()
        self.use_hinge = use_hinge
        self.discriminator = MultiScaleMelPatchGAN(**discriminator_kwargs)

    def forward(self, fake, real):
        d_fake = self.discriminator(fake)
        d_real = self.discriminator(real)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)

        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += (
                F.relu(x_fake[-1]).mean() +
                F.relu(1 - x_real[-1]).mean()
            ) if self.use_hinge else (
                (x_fake[-1] ** 2).mean() +
                ((1 - x_real[-1]) ** 2).mean()
            )
        loss_d /= len(d_fake)
        return loss_d

    def generator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake, real)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += (
                F.relu(1 - x_fake[-1]).mean()
                if self.use_hinge else
                ((1 - x_fake[-1]) ** 2).mean()
            )

        n_discriminators = len(d_fake)
        loss_feature = 0
        for i in range(n_discriminators):
            # Average over N model layers (except for the last item, which is logits).
            n_layers = len(d_fake[i]) - 1
            loss_feature += sum(map(
                lambda j: F.l1_loss(d_fake[i][j], d_real[i][j].detach()),
                range(n_layers)
            )) / n_layers

        # Average over K discriminators.
        loss_feature = loss_feature / n_discriminators

        loss_g /= len(d_fake)
        return loss_g, loss_feature

    def loss(self, reals, fakes):
        gen_loss, feature_distance = self.generator_loss(fakes, reals)
        dis_loss = self.discriminator_loss(fakes, reals)
        return dis_loss, gen_loss, feature_distance
