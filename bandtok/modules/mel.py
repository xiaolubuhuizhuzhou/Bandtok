"""Log-Mel spectrogram front end.
"""

from typing import Optional

import torch
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x: torch.Tensor, C: float = 1.0, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x: torch.Tensor, C: float = 1.0) -> torch.Tensor:
    return torch.exp(x) / C


class LogMelSpectrogram(nn.Module):
    """Waveform -> log-Mel spectrogram, shaped for 2D (image-like) modelling.

    Args:
        sample_rate: input sample rate in Hz.
        n_fft: FFT size.
        num_mels: number of Mel bands (the frequency axis of the token grid).
        hop_size: hop length in samples.
        win_size: window length in samples.
        fmin: lowest Mel band edge in Hz.
        fmax: highest Mel band edge in Hz; ``None`` means ``sample_rate / 2``.
        clip_val: floor applied before the log.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        num_mels: int = 128,
        hop_size: int = 512,
        win_size: int = 2048,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        clip_val: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val

        mel_basis = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_size), persistent=False)

    @property
    def frame_rate(self) -> float:
        """Spectrogram frames per second."""
        return self.sample_rate / self.hop_size

    def num_frames(self, num_samples: int) -> int:
        """Number of frames produced for ``num_samples`` input samples (centred STFT)."""
        return num_samples // self.hop_size + 1

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute the log-Mel spectrogram.

        Args:
            audio: ``[B, T]``, ``[B, 1, T]`` or ``[T]`` waveform in [-1, 1].

        Returns:
            ``[B, 1, num_mels, frames]`` log-Mel spectrogram. The singleton
            channel makes the output directly consumable as a 1-channel image.
        """
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 3:
            if audio.shape[1] != 1:
                raise ValueError(f"expected mono audio with a singleton channel, got shape {tuple(audio.shape)}")
            audio = audio.squeeze(1)
        elif audio.ndim != 2:
            raise ValueError(f"expected [T], [B, T] or [B, 1, T] audio, got shape {tuple(audio.shape)}")

        spec = torch.stft(
            audio,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(audio.device),
            center=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis.to(spec.device), spec)
        spec = dynamic_range_compression(spec, clip_val=self.clip_val)
        return spec.unsqueeze(1)


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    center: bool = True,
) -> torch.Tensor:
    """Functional log-Mel spectrogram, returning ``[B, num_mels, frames]``.

    Provided for scripts that want a one-off transform without holding a module.
    Prefer :class:`LogMelSpectrogram` in training loops, which caches the
    filterbank instead of rebuilding it on every call.
    """
    if not center:
        raise NotImplementedError("BandTok always uses a centred STFT")
    transform = LogMelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        num_mels=num_mels,
        hop_size=hop_size,
        win_size=win_size,
        fmin=fmin,
        fmax=fmax,
    ).to(y.device)
    return transform(y).squeeze(1)
