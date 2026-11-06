"""Tokenizer training.
"""

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from ..losses.base import L1Loss, MultiLoss, ValueLoss
from ..losses.discriminators import MelPatchGANLoss
from ..losses.perceptual import VGGL1Loss
from ..models.tokenizer import BandTok
from .distributed import unwrap_model
from .logger import MetricsLogger
from .trainer import StepOutput, TrainingTask


class DiscriminatorStep(nn.Module):
    """Adapter that exposes the PatchGAN losses through ``forward``.

    DDP only synchronizes gradients for parameters used inside the wrapped
    module's ``forward``, so the discriminator has to be *called*, not poked at
    through a method. ``mode`` selects which side of the GAN objective to compute.
    """

    def __init__(self, patchgan: MelPatchGANLoss) -> None:
        super().__init__()
        self.patchgan = patchgan

    def forward(self, reals: torch.Tensor, fakes: torch.Tensor, mode: str):
        if mode == "discriminator":
            # Operates on a detached reconstruction: no generator gradients.
            return self.patchgan.discriminator_loss(fake=fakes, real=reals)
        if mode == "generator":
            _, loss_adv, loss_fm = self.patchgan.loss(reals=reals, fakes=fakes)
            return loss_adv, loss_fm
        raise ValueError(f"unknown mode {mode!r}; expected 'discriminator' or 'generator'")


class TokenizerTask(TrainingTask):
    """Adversarial Mel-domain training for :class:`BandTok`.

    Args:
        tokenizer: the model to train.
        loss_config: the ``training.loss_configs`` block.
        warmup_steps: steps before the adversarial terms switch on.
        sample_rate: audio sample rate, used when logging demo audio.
    """

    def __init__(
        self,
        tokenizer: BandTok,
        loss_config: Dict[str, Any],
        warmup_steps: int = 0,
        sample_rate: int = 44100,
    ) -> None:
        self.tokenizer = tokenizer
        self.loss_config = loss_config
        self.warmup_steps = warmup_steps
        self.sample_rate = sample_rate

        weights = {name: config.get("weights", {}) for name, config in loss_config.items()}
        generator_losses: List[Any] = []

        if "bottleneck" in loss_config:
            generator_losses.append(
                ValueLoss(
                    key="vq/commitment_loss",
                    weight=weights["bottleneck"]["commit"],
                    name="commitment_loss",
                )
            )
        if weights.get("L_mel", {}).get("L_mel", 0) > 0:
            generator_losses.append(
                L1Loss(key_a="y_mel", key_b="y_g_mel", weight=weights["L_mel"]["L_mel"], name="L_mel")
            )
        if weights.get("L_vgg_l1", {}).get("L_vgg_l1", 0) > 0:
            generator_losses.append(
                VGGL1Loss(
                    key_a="y_mel",
                    key_b="y_g_mel",
                    weight=weights["L_vgg_l1"]["L_vgg_l1"],
                    name="L_vgg_l1",
                )
            )

        self.discriminator: Optional[DiscriminatorStep] = None
        if "discriminator_mel" in loss_config:
            disc_config = loss_config["discriminator_mel"]
            if disc_config.get("type") != "patchgan":
                raise ValueError(
                    f"unknown mel discriminator type {disc_config.get('type')!r}; expected 'patchgan'"
                )
            self.discriminator = DiscriminatorStep(
                MelPatchGANLoss(channels=tokenizer.out_channels, **disc_config.get("config", {}))
            )
            generator_losses += [
                ValueLoss(key="loss_adv", weight=weights["discriminator_mel"]["adversarial"], name="loss_adv"),
                ValueLoss(
                    key="feature_matching_distance",
                    weight=weights["discriminator_mel"]["feature_matching"],
                    name="loss_fm",
                ),
            ]
            self.discriminator_losses = MultiLoss(
                [
                    ValueLoss(
                        key="loss_dis",
                        weight=weights["discriminator_mel"]["discriminator"],
                        name="loss_dis",
                    )
                ]
            )

        if not generator_losses:
            raise ValueError("no generator losses configured; check training.loss_configs")
        self.generator_losses = MultiLoss(generator_losses)

        self.modules: Dict[str, nn.Module] = {"model": tokenizer}
        self.trainable_modules: Dict[str, nn.Module] = {"autoencoder": tokenizer}
        if self.discriminator is not None:
            self.modules["discriminator"] = self.discriminator
            self.trainable_modules["discriminator"] = self.discriminator

    # -- plumbing ---------------------------------------------------------

    def parameter_groups(self) -> Dict[str, Iterable]:
        groups: Dict[str, Iterable] = {"autoencoder": list(self.tokenizer.parameters())}
        if self.discriminator is not None:
            groups["discriminator"] = list(self.discriminator.parameters())
        return groups

    def to(self, device: torch.device) -> None:
        super().to(device)
        # The VGG perceptual loss owns frozen weights that also need moving.
        for loss_module in self.generator_losses.losses:
            if isinstance(loss_module, nn.Module):
                loss_module.to(device)

    def active_group(self, step: int) -> str:
        """Odd steps train the discriminator, even steps the generator."""
        if self.discriminator is None or step < self.warmup_steps or step % 2 == 0:
            return "autoencoder"
        return "discriminator"

    # -- steps ------------------------------------------------------------

    def train_step(self, batch: Any, step: int) -> StepOutput:
        audio = batch[0] if isinstance(batch, (list, tuple)) else batch
        autoencoder = self.trainable_modules["autoencoder"]
        y_mel = unwrap_model(autoencoder).mel(audio)

        group = self.active_group(step)

        if group == "discriminator":
            with torch.no_grad():
                y_g_mel, _ = autoencoder(y_mel)
            loss_dis = self.trainable_modules["discriminator"](
                reals=y_mel, fakes=y_g_mel, mode="discriminator"
            )
            loss, losses = self.discriminator_losses({"loss_dis": loss_dis})
            metrics: Dict[str, Any] = dict(losses)
            metrics["loss_d"] = loss
            return StepOutput(loss=loss, group=group, metrics=metrics)

        y_g_mel, info = autoencoder(y_mel)

        loss_info: Dict[str, Any] = {"y_mel": y_mel, "y_g_mel": y_g_mel}
        loss_info.update(info)

        if self.discriminator is not None:
            if step >= self.warmup_steps:
                loss_adv, loss_fm = self.trainable_modules["discriminator"](
                    reals=y_mel, fakes=y_g_mel, mode="generator"
                )
            else:
                zero = torch.zeros((), device=y_mel.device, dtype=y_mel.dtype)
                loss_adv, loss_fm = zero, zero
            loss_info["loss_adv"] = loss_adv
            loss_info["feature_matching_distance"] = loss_fm

        loss, losses = self.generator_losses(loss_info)
        metrics = dict(losses)
        metrics["loss_g"] = loss
        if "z" in info:
            metrics["latent_std"] = info["z"].std()
        return StepOutput(loss=loss, group=group, metrics=metrics)

    @torch.no_grad()
    def validation_step(self, batch: Any) -> Dict[str, float]:
        audio = batch[0] if isinstance(batch, (list, tuple)) else batch
        tokenizer = unwrap_model(self.trainable_modules["autoencoder"])
        y_mel = tokenizer.mel(audio)
        y_g_mel, info = tokenizer(y_mel)

        metrics = {"L_mel": float((y_mel - y_g_mel).abs().mean())}
        if "vq/commitment_loss" in info:
            metrics["commitment_loss"] = float(info["vq/commitment_loss"])
        codes = info.get("codes")
        if codes is not None:
            metrics["codebook_usage"] = codes.unique().numel() / tokenizer.codebook_size
        return metrics

    @torch.no_grad()
    def on_demo(self, batch: Any, step: int, logger: MetricsLogger) -> None:
        """Log reconstructed spectrograms and, if the vocoder loads, audio."""
        audio = batch[0] if isinstance(batch, (list, tuple)) else batch
        audio = audio[:2]
        tokenizer = unwrap_model(self.trainable_modules["autoencoder"])
        y_mel = tokenizer.mel(audio)
        y_g_mel, _ = tokenizer(y_mel)

        for index in range(y_mel.shape[0]):
            # flip so low frequencies are at the bottom of the logged image
            logger.log_image(f"demo/mel_real_{index}", y_mel[index, 0].flip(0), step)
            logger.log_image(f"demo/mel_recon_{index}", y_g_mel[index, 0].flip(0), step)
            logger.log_audio(f"demo/audio_real_{index}", audio[index], step, self.sample_rate)

        recon_audio = tokenizer.decode_by_vocoder(y_g_mel)
        for index in range(recon_audio.shape[0]):
            logger.log_audio(f"demo/audio_recon_{index}", recon_audio[index], step, self.sample_rate)


def create_tokenizer_task(model_config: Dict[str, Any], tokenizer: BandTok) -> TokenizerTask:
    """Build a :class:`TokenizerTask` from a tokenizer model config."""
    training_config = model_config["training"]
    sample_size = model_config["sample_size"]

    # Fail fast rather than train against a misaligned reconstruction target.
    tokenizer.check_sample_size(sample_size)

    disc_config = training_config["loss_configs"].get("discriminator_mel", {}).get("config", {})
    if disc_config:
        frames = tokenizer.mel.num_frames(sample_size)
        scales = disc_config.get("scales", [1])
        max_downs = disc_config.get("max_downs", [16])
        # Each scale first downsamples the spectrogram by `scale`, then halves it
        # log2(max_down) times; the final 4x4 convolution needs at least 4 frames.
        required = max(4 * down * scale for scale, down in zip(scales, max_downs))
        if frames < required:
            raise ValueError(
                f"sample_size={sample_size} gives {frames} Mel frames, but the multi-scale "
                f"discriminator (scales={scales}, max_downs={max_downs}) needs at least {required}. "
                f"Use a longer segment or smaller max_downs."
            )

    return TokenizerTask(
        tokenizer=tokenizer,
        loss_config=training_config["loss_configs"],
        warmup_steps=int(training_config.get("warmup_steps", 0)),
        sample_rate=model_config["sample_rate"],
    )
