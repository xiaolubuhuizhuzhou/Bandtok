"""Model construction from configs."""

from typing import Any, Dict, Optional

import torch.nn as nn

from ..config import load_model_config
from ..modules.cosmos import CosmosDecoder, CosmosEncoder
from ..modules.mel import LogMelSpectrogram
from ..modules.quantize import BandVQBottleneck
from .tokenizer import BandTok


def create_encoder_from_config(encoder_config: Dict[str, Any]) -> nn.Module:
    encoder_type = encoder_config.get("type")
    if encoder_type != "cosmos":
        raise ValueError(f"unknown encoder type {encoder_type!r}; BandTok uses 'cosmos'")
    return CosmosEncoder(**encoder_config["config"])


def create_decoder_from_config(decoder_config: Dict[str, Any]) -> nn.Module:
    decoder_type = decoder_config.get("type")
    if decoder_type != "cosmos":
        raise ValueError(f"unknown decoder type {decoder_type!r}; BandTok uses 'cosmos'")
    return CosmosDecoder(**decoder_config["config"])


def create_bottleneck_from_config(bottleneck_config: Dict[str, Any]) -> nn.Module:
    bottleneck_type = bottleneck_config.get("type")
    if bottleneck_type != "band_vq":
        raise ValueError(f"unknown bottleneck type {bottleneck_type!r}; BandTok uses 'band_vq'")
    return BandVQBottleneck(**bottleneck_config.get("config", {}))


def create_mel_from_config(config: Dict[str, Any]) -> LogMelSpectrogram:
    """Build the log-Mel front end from the top level of a model config."""
    return LogMelSpectrogram(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        num_mels=config["num_mels"],
        hop_size=config["hop_size"],
        win_size=config["win_size"],
        fmin=config.get("fmin", 0.0),
        fmax=config.get("fmax", None),
    )


def create_tokenizer_from_config(config: Dict[str, Any]) -> BandTok:
    """Build a :class:`BandTok` tokenizer from a tokenizer model config."""
    if config.get("model_type") != "bandtok":
        raise ValueError(f"expected model_type 'bandtok', got {config.get('model_type')!r}")

    model_config = config["model"]
    encoder_config = model_config["encoder"]["config"]
    bottleneck_config = model_config["bottleneck"].get("config", {})

    # The quantizer sees the encoder's output channels directly. A mismatch would
    # otherwise surface deep inside the quantizer as a bare assertion.
    latent_dim = encoder_config["z_channels"]
    bottleneck_dim = bottleneck_config.get("input_dim", latent_dim)
    if bottleneck_dim != latent_dim:
        raise ValueError(
            f"bottleneck input_dim={bottleneck_dim} does not match the encoder's "
            f"z_channels={latent_dim}; the quantizer operates on the encoder's output directly"
        )
    if model_config.get("latent_dim", latent_dim) != latent_dim:
        raise ValueError(
            f"model.latent_dim={model_config['latent_dim']} disagrees with the encoder's "
            f"z_channels={latent_dim}"
        )

    encoder = create_encoder_from_config(model_config["encoder"])
    mel_decoder = create_decoder_from_config(model_config["mel_decoder"])
    bottleneck = create_bottleneck_from_config(model_config["bottleneck"])

    return BandTok(
        encoder=encoder,
        mel_decoder=mel_decoder,
        bottleneck=bottleneck,
        mel=create_mel_from_config(config),
        spatial_compression=encoder_config["spatial_compression"],
        latent_dim=latent_dim,
        sample_rate=config["sample_rate"],
        sample_size=config.get("sample_size"),
        vocoder_path=model_config.get("vocoder_path"),
    )


def create_tokenizer_from_pretransform_config(
    pretransform_config: Dict[str, Any], strict: bool = False
) -> BandTok:
    """Build the frozen tokenizer an LM config points at, and load its weights.

    Args:
        pretransform_config: the LM config's ``model.pretransform`` block
            (``type: "bandtok"``), with a ``model_config`` path and an optional
            ``ckpt_path``.
        strict: require an exact state-dict match when loading the checkpoint.
    """
    kind = pretransform_config.get("type")
    if kind != "bandtok":
        raise ValueError(f"unknown pretransform type {kind!r}; expected 'bandtok'")

    inner = pretransform_config["config"]
    tokenizer_config = load_model_config(inner["model_config"])
    if tokenizer_config.get("model_type") != "bandtok":
        raise ValueError(
            f"pretransform type 'bandtok' needs a 'bandtok' model config, but "
            f"{inner['model_config']} is {tokenizer_config.get('model_type')!r}"
        )

    tokenizer = create_tokenizer_from_config(tokenizer_config)

    ckpt_path = inner.get("ckpt_path")
    if ckpt_path:
        from ..utils.checkpoint import load_model_weights

        load_model_weights(tokenizer, ckpt_path, strict=strict, name="tokenizer")
    else:
        print("[bandtok] no tokenizer ckpt_path given; the LM will train on tokens from a random tokenizer")

    tokenizer.eval()
    tokenizer.requires_grad_(False)
    return tokenizer


def create_model_from_config(config: Dict[str, Any], tokenizer: Optional[BandTok] = None) -> nn.Module:
    """Build either the tokenizer or the LM, depending on ``model_type``."""
    model_type = config.get("model_type")

    if model_type == "bandtok":
        return create_tokenizer_from_config(config)

    if model_type == "bandtok_lm":
        if tokenizer is None:
            tokenizer = create_tokenizer_from_pretransform_config(config["model"]["pretransform"])

        from .lm import create_lm_from_config

        return create_lm_from_config(config, tokenizer=tokenizer)

    raise ValueError(f"unknown model_type {model_type!r}; expected 'bandtok' or 'bandtok_lm'")
