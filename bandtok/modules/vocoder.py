"""Frozen BigVGAN-v2 vocoder wrapper.

BandTok reconstructs a log-Mel spectrogram; a pretrained, frozen BigVGAN-v2
vocoder turns that spectrogram back into a 44.1 kHz waveform. The vocoder is
never trained here, and it is deliberately kept out of the tokenizer's
``state_dict`` so that checkpoints only contain BandTok's own parameters.

The BigVGAN inference code is vendored under ``third_party/bigvgan`` (MIT,
NVIDIA). Weights are fetched from the Hugging Face Hub on first use, or loaded
from a local directory via ``--vocoder-path`` / ``$BANDTOK_VOCODER_PATH``.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

DEFAULT_VOCODER_ID = "nvidia/bigvgan_v2_44khz_128band_512x"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BIGVGAN_DIR = _REPO_ROOT / "third_party" / "bigvgan"


def _import_bigvgan():
    """Import the vendored BigVGAN module.

    BigVGAN's own modules import each other by top-level name (``import
    activations``), so its directory has to be on ``sys.path``.
    """
    if not (_BIGVGAN_DIR / "bigvgan.py").exists():
        raise FileNotFoundError(
            f"vendored BigVGAN code not found at {_BIGVGAN_DIR}. "
            "The repository should ship it under third_party/bigvgan."
        )
    if str(_BIGVGAN_DIR) not in sys.path:
        sys.path.insert(0, str(_BIGVGAN_DIR))
    import bigvgan  # noqa: E402  (path set up above)

    return bigvgan


def load_vocoder(
    path_or_id: Optional[str] = None,
    num_mels: int = 128,
    use_cuda_kernel: bool = False,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a frozen BigVGAN-v2 generator in eval mode.

    Args:
        path_or_id: local checkpoint directory or Hugging Face repo id. Defaults
            to ``$BANDTOK_VOCODER_PATH`` if set, else ``nvidia/bigvgan_v2_44khz_128band_512x``.
        num_mels: number of Mel bands the vocoder must accept; BandTok uses 128.
        use_cuda_kernel: use BigVGAN's fused CUDA activation kernel (compiles on
            first use). The plain PyTorch path is the default.
        device: device to move the vocoder to.

    Returns:
        The generator module, in eval mode with all parameters frozen and weight
        norm removed.
    """
    if num_mels != 128:
        raise ValueError(
            f"num_mels={num_mels} has no matching BigVGAN-v2 44.1 kHz checkpoint; BandTok uses 128."
        )

    path_or_id = path_or_id or os.environ.get("BANDTOK_VOCODER_PATH") or DEFAULT_VOCODER_ID
    bigvgan = _import_bigvgan()
    try:
        vocoder = bigvgan.BigVGAN.from_pretrained(path_or_id, use_cuda_kernel=use_cuda_kernel)
    except Exception as exc:
        raise RuntimeError(
            f"could not load the BigVGAN vocoder from {path_or_id!r}: {type(exc).__name__}: {exc}\n"
            "On a machine without network access, download the weights elsewhere and pass "
            "--vocoder-path (or set $BANDTOK_VOCODER_PATH). Setting HF_HUB_OFFLINE=1 makes the "
            "failure immediate instead of waiting through Hugging Face's retries."
        ) from exc
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval()
    for p in vocoder.parameters():
        p.requires_grad = False
    if device is not None:
        vocoder = vocoder.to(device)
    return vocoder


@torch.no_grad()
def vocode(vocoder: nn.Module, logmel: torch.Tensor) -> torch.Tensor:
    """Run the vocoder on a log-Mel spectrogram.

    Args:
        vocoder: module returned by :func:`load_vocoder`.
        logmel: ``[B, num_mels, frames]`` or ``[B, 1, num_mels, frames]``.

    Returns:
        ``[B, 1, samples]`` waveform.
    """
    if logmel.ndim == 4:
        if logmel.shape[1] != 1:
            raise ValueError(f"expected a single mel channel, got shape {tuple(logmel.shape)}")
        logmel = logmel.squeeze(1)
    if next(vocoder.parameters()).device != logmel.device:
        vocoder.to(logmel.device)
    audio = vocoder(logmel.float())
    if audio.ndim == 2:
        audio = audio.unsqueeze(1)
    return audio.clamp(-1.0, 1.0)
