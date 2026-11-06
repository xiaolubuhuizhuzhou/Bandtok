"""The BandTok tokenizer.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..modules.cosmos import CosmosDecoder, CosmosEncoder
from ..modules.mel import LogMelSpectrogram
from ..modules.quantize import Bottleneck
from ..modules.vocoder import load_vocoder, vocode


class BandTok(nn.Module):
    """2D Mel-spectrogram tokenizer.

    Args:
        encoder: Cosmos-style 2D encoder mapping ``[B, 1, F, T]`` log-Mel to a
            latent grid ``[B, C, F', T']``.
        mel_decoder: Cosmos-style 2D decoder mapping the latent grid back to a
            log-Mel spectrogram.
        bottleneck: discrete bottleneck (the shared-codebook quantizer).
        mel: log-Mel front end; also defines the spectrogram the decoder targets.
        spatial_compression: compression factor the encoder applies to both the
            time and the frequency axis. With 128 Mel bands and a factor of 8 the
            grid has ``num_bands = 16`` Mel-band positions per frame.
        latent_dim: channel width of the latent grid.
        sample_rate: audio sample rate in Hz.
        sample_size: training segment length in samples, used by downstream code
            to size token sequences.
        vocoder_path: local path or Hugging Face id for the BigVGAN vocoder.
    """

    def __init__(
        self,
        encoder: CosmosEncoder,
        mel_decoder: CosmosDecoder,
        bottleneck: Bottleneck,
        mel: Optional[LogMelSpectrogram] = None,
        spatial_compression: int = 8,
        latent_dim: int = 128,
        sample_rate: int = 44100,
        sample_size: Optional[int] = None,
        vocoder_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not getattr(bottleneck, "is_discrete", False):
            raise ValueError("BandTok requires a discrete bottleneck")

        self.encoder = encoder
        self.mel_decoder = mel_decoder
        self.bottleneck = bottleneck
        self.mel = mel if mel is not None else LogMelSpectrogram(sample_rate=sample_rate)

        if self.mel.num_mels % spatial_compression != 0:
            raise ValueError(
                f"num_mels={self.mel.num_mels} is not divisible by spatial_compression={spatial_compression}"
            )
        # The encoder compresses time and frequency by the same factor, so the
        # number of Mel-band token positions follows from the Mel resolution.
        # The BandTok-1D ablation instead folds the frequency axis into the
        # channels (``encoder.flatten``) and stacks residual codebooks, leaving a
        # single position per frame.
        self.spatial_compression = spatial_compression
        self.num_bands = 1 if getattr(encoder, "flatten", False) else self.mel.num_mels // spatial_compression
        self.latent_dim = latent_dim
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.in_channels = 1
        self.out_channels = 1

        self.is_discrete = True
        self.num_quantizers = bottleneck.num_quantizers
        self.codebook_size = bottleneck.codebook_size

        self._vocoder_path = vocoder_path
        self._vocoder: Optional[nn.Module] = None

    # -- geometry ---------------------------------------------------------

    @property
    def token_rate(self) -> float:
        """Audio tokens per second (all bands of all frames)."""
        return self.frame_rate * self.num_bands * self.num_quantizers

    @property
    def frame_rate(self) -> float:
        """Latent frames per second (the time axis of the token grid)."""
        return self.mel.frame_rate / self.spatial_compression

    @property
    def downsampling_ratio(self) -> int:
        """Audio samples per token, i.e. ``hop_size * spatial_compression / num_bands``."""
        return self.mel.hop_size * self.spatial_compression // self.num_bands

    @property
    def bitrate(self) -> float:
        """Bitrate in bits per second for the discrete token stream."""
        import math

        return self.token_rate * math.log2(self.codebook_size)

    def num_tokens(self, num_samples: int) -> int:
        """Number of audio tokens produced for ``num_samples`` input samples."""
        frames = self.mel.num_frames(num_samples)
        latent_frames = -(-frames // self.spatial_compression)  # ceil
        return latent_frames * self.num_bands * self.num_quantizers

    def aligned_sample_size(self, num_samples: int) -> int:
        """Largest segment length <= ``num_samples`` whose frame count is aligned.
        """
        frames = self.mel.num_frames(num_samples)
        if frames < self.spatial_compression:
            raise ValueError(
                f"{num_samples} samples give only {frames} Mel frames; at least "
                f"{self.spatial_compression} are needed "
                f"({(self.spatial_compression - 1) * self.mel.hop_size} samples)"
            )
        aligned_frames = (frames // self.spatial_compression) * self.spatial_compression
        return (aligned_frames - 1) * self.mel.hop_size

    def check_sample_size(self, num_samples: int) -> None:
        """Raise if a training segment length would make the mel loss misalign."""
        frames = self.mel.num_frames(num_samples)
        if frames % self.spatial_compression == 0:
            return
        if frames < self.spatial_compression:
            raise ValueError(
                f"sample_size={num_samples} gives only {frames} Mel frames, fewer than "
                f"spatial_compression={self.spatial_compression}; the segment is too short to encode. "
                f"The paper uses 65024."
            )
        raise ValueError(
            f"sample_size={num_samples} gives {frames} Mel frames, which is not a multiple of "
            f"spatial_compression={self.spatial_compression}; the reconstruction would be "
            f"shorter than the target. Use sample_size={self.aligned_sample_size(num_samples)} "
            f"(the paper uses 65024)."
        )

    # -- encode / decode --------------------------------------------------

    def encode_mel(
        self, logmel: torch.Tensor, return_info: bool = False, skip_bottleneck: bool = False
    ) -> torch.Tensor:
        """Encode a log-Mel spectrogram into a quantized latent grid.

        Args:
            logmel: ``[B, 1, F, T]`` (a ``[B, F, T]`` input gains the channel axis).
            return_info: also return the bottleneck info dict, which carries the
                commitment loss and the discrete codes.
            skip_bottleneck: return continuous latents without quantizing.

        Returns:
            ``[B, C, F', T']`` latents, or ``(latents, info)``.
        """
        if logmel.ndim == 3:
            logmel = logmel.unsqueeze(1)

        info: Dict[str, Any] = {}
        latents = self.encoder(logmel)
        info["pre_bottleneck_latents"] = latents

        if skip_bottleneck:
            return (latents, info) if return_info else latents

        flattened = self._flatten_grid(latents)
        latents, bottleneck_info = self.bottleneck.encode(flattened, return_info=True)
        if not getattr(self.encoder, "flatten", False):
            latents = self._unflatten_grid(latents)
        info.update(bottleneck_info)

        return (latents, info) if return_info else latents

    def decode_mel(self, latents: torch.Tensor, skip_bottleneck: bool = False) -> torch.Tensor:
        """Decode latents back to a log-Mel spectrogram ``[B, 1, F, T]``."""
        if not skip_bottleneck:
            latents = self.bottleneck.decode(latents)
        return self.mel_decoder(latents)

    def encode_audio(self, audio: torch.Tensor, return_info: bool = False, **kwargs) -> torch.Tensor:
        """Waveform ``[B, 1, N]`` -> quantized latent grid."""
        return self.encode_mel(self.mel(audio), return_info=return_info, **kwargs)

    def forward(self, logmel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Reconstruct a log-Mel spectrogram. Returns ``(logmel_hat, info)``."""
        latents, info = self.encode_mel(logmel, return_info=True)
        return self.decode_mel(latents, skip_bottleneck=True), info

    # -- discrete token API (used by the language model) ------------------

    def tokenize(self, audio: torch.Tensor) -> torch.Tensor:
        """Waveform ``[B, 1, N]`` -> codes ``[B, num_quantizers, T'*F']``, band-first."""
        _, info = self.encode_audio(audio, return_info=True)
        return info[self.bottleneck.tokens_id]

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes ``[B, num_quantizers, T'*F']`` -> log-Mel spectrogram ``[B, 1, F, T]``."""
        latents = self.bottleneck.decode_tokens(codes)
        if latents.ndim == 3 and not getattr(self.encoder, "flatten", False):
            latents = self._unflatten_grid(latents)
        return self.mel_decoder(latents)

    def decode_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes -> waveform ``[B, 1, N]`` (mel decoder followed by the vocoder)."""
        return self.decode_by_vocoder(self.decode_codes(codes))

    # -- vocoder ----------------------------------------------------------

    @property
    def vocoder(self) -> nn.Module:
        """The frozen BigVGAN vocoder, loaded on first access.

        Held outside the module tree (plain attribute, not a submodule) so that
        it never appears in ``state_dict()`` or in the optimizer's parameters.
        """
        if self._vocoder is None:
            self._vocoder = load_vocoder(self._vocoder_path, num_mels=self.mel.num_mels)
        return self._vocoder

    def set_vocoder_path(self, path: Optional[str]) -> None:
        """Point at a different vocoder checkpoint, discarding any loaded one.

        Lets a CLI flag override whatever the model config specified.
        """
        self._vocoder_path = path
        self._vocoder = None

    @torch.no_grad()
    def decode_by_vocoder(self, logmel: torch.Tensor) -> torch.Tensor:
        """Log-Mel spectrogram -> waveform ``[B, 1, N]``."""
        return vocode(self.vocoder, logmel)

    # -- band-first flattening -------------------------------------------

    def _flatten_grid(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B, C, F', T']`` -> ``[B, C, T'*F']``, bands of one frame consecutive."""
        if latents.ndim != 4:
            return latents
        b, c, f, t = latents.shape
        if f != self.num_bands:
            raise ValueError(f"encoder produced {f} bands but num_bands={self.num_bands}")
        return latents.permute(0, 1, 3, 2).contiguous().view(b, c, t * f)

    def _unflatten_grid(self, latents: torch.Tensor) -> torch.Tensor:
        """``[B, C, T'*F']`` -> ``[B, C, F', T']``, inverse of :meth:`_flatten_grid`."""
        if latents.ndim != 3:
            return latents
        b, c, n = latents.shape
        if n % self.num_bands != 0:
            raise ValueError(f"token count {n} is not a multiple of num_bands={self.num_bands}")
        t = n // self.num_bands
        return latents.view(b, c, t, self.num_bands).permute(0, 1, 3, 2).contiguous()
