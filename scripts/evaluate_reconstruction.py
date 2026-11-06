#!/usr/bin/env python3
"""Evaluate BandTok reconstruction quality (the paper's Table II).

The paper reports a **Mel distance** and an **STFT distance** over 1000 random
segments of the MUSDB18-HQ test set. Both are spectral distances between the
original waveform and the waveform BandTok reconstructs, so they are computed
with the ported auraloss modules in :mod:`bandtok.losses.spectral`::

    audio -> log-Mel -> encode -> quantize -> decode -> log-Mel_hat
                                                     -> frozen BigVGAN -> audio_hat

Three reconstruction numbers are reported, and they are *not* interchangeable:

``mel_distance``
    :class:`~bandtok.losses.spectral.MelSTFTLoss` between ``audio`` and
    ``audio_hat``. This is the "Mel distance" of the codec literature the paper
    follows (spectral convergence + L1 on log Mel-scaled STFT magnitudes) and it
    therefore *includes* the vocoder's own error. Needs the vocoder.
``stft_distance``
    :class:`~bandtok.losses.spectral.STFTLoss` between ``audio`` and
    ``audio_hat`` on the linear-frequency STFT magnitude. Needs the vocoder. The
    exact fft/hop/window settings are written to the JSON, because the value is
    meaningless without them.
``logmel_l1``
    Plain L1 between ``log-Mel`` and ``log-Mel_hat``, i.e. the tokenizer's own
    training objective (``L_mel``) measured on held-out audio. It is computed
    directly in the Mel domain and needs **no vocoder**, so it is the only
    reconstruction number available when the BigVGAN weights cannot be loaded.

Also reported: codebook usage and token entropy over the evaluated segments, and
the token rate / bitrate implied by the tokenizer's geometry.

Example::

    python scripts/evaluate_reconstruction.py \\
        --model-config configs/tokenizer/bandtok.json \\
        --ckpt runs/bandtok/checkpoints/last.pt \\
        --data-config configs/data/valid_musdb_test.json \\
        --num-segments 1000 --batch-size 4 \\
        --output results/bandtok_musdb_test.json

The vocoder is optional: if BigVGAN cannot be loaded (no local weights and no
network access) the waveform-domain metrics are skipped with a warning and the
Mel-domain metrics are still reported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandtok.config import load_data_config, load_model_config  # noqa: E402
from bandtok.data.dataset import create_dataloader_from_config  # noqa: E402
from bandtok.losses.spectral import MelSTFTLoss, STFTLoss  # noqa: E402
from bandtok.models.factory import create_tokenizer_from_config  # noqa: E402
from bandtok.models.tokenizer import BandTok  # noqa: E402
from bandtok.training.distributed import set_seed  # noqa: E402
from bandtok.utils.checkpoint import load_model_weights  # noqa: E402


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True, help="tokenizer model config (JSON)")
    parser.add_argument("--ckpt", required=True, help="tokenizer checkpoint")
    parser.add_argument("--data-config", required=True, help="evaluation dataset config (JSON)")
    parser.add_argument("--num-segments", type=int, default=1000, help="segments to average over")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    parser.add_argument("--output", default=None, help="write the results to this JSON file")
    parser.add_argument(
        "--vocoder-path",
        default=None,
        help="local BigVGAN directory or Hugging Face id; overrides the model config",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def resolve_device(device: Optional[str]) -> torch.device:
    """Pick the evaluation device, defaulting to CUDA when it is available."""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TokenStatistics:
    """Codebook usage and token entropy accumulated over the evaluated segments.

    Counts are kept as a ``[num_quantizers, codebook_size]`` histogram, so the
    same accumulator covers BandTok (one shared codebook) and the BandTok-1D
    ablation (one histogram row per residual codebook).
    """

    def __init__(self, num_quantizers: int, codebook_size: int) -> None:
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.counts = torch.zeros(num_quantizers, codebook_size, dtype=torch.float64)

    def update(self, codes: torch.Tensor) -> None:
        """Accumulate a ``[B, num_quantizers, L]`` (or ``[B, L]``) batch of codes."""
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        codes = codes.detach().to("cpu", torch.long)
        for index in range(codes.size(1)):
            self.counts[index] += torch.bincount(
                codes[:, index].reshape(-1), minlength=self.codebook_size
            ).to(torch.float64)

    @staticmethod
    def _entropy_bits(counts: torch.Tensor) -> float:
        """Plug-in (maximum-likelihood) entropy of a count vector, in bits."""
        total = float(counts.sum())
        if total <= 0:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        return float(-(probs * probs.log2()).sum())

    def summary(self) -> Dict[str, Any]:
        used = (self.counts > 0).sum(dim=1)
        per_codebook_usage = (used.double() / self.codebook_size).tolist()
        per_codebook_entropy = [self._entropy_bits(row) for row in self.counts]
        return {
            "num_tokens": int(self.counts.sum()),
            "codebook_size": self.codebook_size,
            "num_quantizers": self.num_quantizers,
            # Fraction of the num_quantizers * codebook_size (codebook, entry)
            # slots that the evaluated segments touched at least once.
            "codebook_usage": float((self.counts > 0).sum()) / float(self.counts.numel()),
            # Mean over codebooks of the per-codebook plug-in token entropy.
            "token_entropy_bits": float(sum(per_codebook_entropy) / len(per_codebook_entropy)),
            "max_entropy_bits": math.log2(self.codebook_size),
            "per_codebook_usage": per_codebook_usage,
            "per_codebook_entropy_bits": per_codebook_entropy,
        }


def crop_to_common_length(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trim two tensors to their shortest extent along the last axis.

    The vocoder emits ``num_frames * hop_size`` samples, which overshoots the
    input segment by up to one hop, and a segment whose frame count is not a
    multiple of ``spatial_compression`` decodes to slightly fewer Mel frames.
    """
    length = min(a.size(-1), b.size(-1))
    return a[..., :length], b[..., :length]


def try_load_vocoder(tokenizer: BandTok, device: torch.device) -> Tuple[bool, Optional[str]]:
    """Load the frozen vocoder, returning ``(available, reason_if_not)``.

    Loading pulls the BigVGAN weights from a local directory or the Hugging Face
    Hub. On an offline machine that fails, and the waveform-domain metrics are
    simply skipped rather than aborting the whole evaluation. Note that the Hub
    client retries a few times before giving up, so on a machine with no route to
    the network the attempt can block for minutes; ``HF_HUB_OFFLINE=1`` (or a
    local ``--vocoder-path``) avoids the wait.
    """
    try:
        tokenizer.vocoder.to(device)
        return True, None
    except Exception as exc:  # no local weights, no network, wrong num_mels, ...
        reason = f"{type(exc).__name__}: {exc}"
        print(
            "[bandtok] WARNING: could not load the BigVGAN vocoder "
            f"({reason.splitlines()[0]})\n"
            "[bandtok] WARNING: skipping the waveform-domain metrics (mel_distance, "
            "stft_distance); logmel_l1 is still reported"
        )
        return False, reason


def iterate_segments(loader: torch.utils.data.DataLoader, num_segments: int):
    """Yield ``(audio, num_segments_seen)`` until ``num_segments`` are produced.

    The loader is re-iterated as many times as needed: MUSDB18-HQ's test split
    has only 50 tracks, so 1000 random segments means about 20 passes over the
    manifest, each taking a fresh random crop of every track.
    """
    seen = 0
    while seen < num_segments:
        produced = 0
        for batch in loader:
            audio = batch[0] if isinstance(batch, (list, tuple)) else batch
            audio = audio[: num_segments - seen]
            seen += audio.size(0)
            produced += audio.size(0)
            yield audio, seen
            if seen >= num_segments:
                return
        if produced == 0:
            raise RuntimeError("the dataloader yielded no segments; check the data config")


@torch.no_grad()
def evaluate(
    tokenizer: BandTok,
    loader: torch.utils.data.DataLoader,
    num_segments: int,
    device: torch.device,
    mel_loss: Optional[MelSTFTLoss],
    stft_loss: Optional[STFTLoss],
) -> Tuple[Dict[str, float], TokenStatistics, int]:
    """Average the reconstruction metrics over ``num_segments`` segments.

    Returns ``(metrics, token_statistics, num_segments_seen)``. Metrics are
    accumulated as segment-weighted sums so that a short final batch does not
    bias the mean.
    """
    stats = TokenStatistics(tokenizer.num_quantizers, tokenizer.codebook_size)
    totals: Dict[str, float] = {"logmel_l1": 0.0, "mel_distance": 0.0, "stft_distance": 0.0}
    seen = 0

    for audio, seen in iterate_segments(loader, num_segments):
        audio = audio.to(device)
        batch_size = audio.size(0)

        logmel = tokenizer.mel(audio)
        latents, info = tokenizer.encode_mel(logmel, return_info=True)
        logmel_hat = tokenizer.decode_mel(latents, skip_bottleneck=True)

        target, recon = crop_to_common_length(logmel, logmel_hat)
        totals["logmel_l1"] += float((target - recon).abs().mean()) * batch_size

        codes = info.get(tokenizer.bottleneck.tokens_id)
        if codes is not None:
            stats.update(codes)

        if mel_loss is not None and stft_loss is not None:
            audio_hat = tokenizer.decode_by_vocoder(logmel_hat)
            reference, estimate = crop_to_common_length(audio, audio_hat)
            totals["mel_distance"] += float(mel_loss(estimate, reference)) * batch_size
            totals["stft_distance"] += float(stft_loss(estimate, reference)) * batch_size

        if (seen - batch_size) // 100 != seen // 100 or seen >= num_segments:
            print(f"[bandtok] {seen}/{num_segments} segments", flush=True)

    if seen == 0:
        raise RuntimeError("the dataloader yielded no segments; check the data config")
    return {key: value / seen for key, value in totals.items()}, stats, seen


def print_summary(results: Dict[str, Any]) -> None:
    """Print the small summary table that goes into the paper's Table II."""
    mel = results["mel_domain"]
    wave = results["waveform_domain"]
    tokens = results["tokens"]
    rate = results["rate"]

    rows: List[Tuple[str, str]] = [
        ("segments", f"{results['num_segments']}"),
        ("logmel_l1 (Mel domain, no vocoder)", f"{mel['logmel_l1']:.4f}"),
    ]
    if wave["available"]:
        rows += [
            ("mel_distance (MelSTFTLoss, waveform)", f"{wave['mel_distance']:.4f}"),
            ("stft_distance (STFTLoss, waveform)", f"{wave['stft_distance']:.4f}"),
        ]
    else:
        rows += [
            ("mel_distance (MelSTFTLoss, waveform)", "n/a (no vocoder)"),
            ("stft_distance (STFTLoss, waveform)", "n/a (no vocoder)"),
        ]
    rows += [
        ("codebook_usage", f"{tokens['codebook_usage'] * 100:.2f}% of {tokens['codebook_size']}"),
        ("token_entropy", f"{tokens['token_entropy_bits']:.3f} / {tokens['max_entropy_bits']:.3f} bits"),
        ("token_rate", f"{rate['token_rate']:.2f} tokens/s"),
        ("bitrate", f"{rate['bitrate_kbps']:.3f} kbps"),
    ]

    width = max(len(name) for name, _ in rows)
    print("\n[bandtok] reconstruction evaluation")
    print("[bandtok] " + "-" * (width + 24))
    for name, value in rows:
        print(f"[bandtok] {name.ljust(width)}  {value}")
    print("[bandtok] " + "-" * (width + 24))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    set_seed(args.seed, rank_offset=False)
    device = resolve_device(args.device)

    model_config = load_model_config(args.model_config)
    if model_config["model_type"] != "bandtok":
        raise ValueError(f"expected a tokenizer config, got model_type {model_config['model_type']!r}")
    if args.vocoder_path:
        model_config["model"]["vocoder_path"] = args.vocoder_path

    tokenizer = create_tokenizer_from_config(model_config)
    load_model_weights(tokenizer, args.ckpt, strict=False, name="tokenizer")
    tokenizer = tokenizer.to(device).eval()
    tokenizer.requires_grad_(False)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    frames = tokenizer.mel.num_frames(sample_size)
    if frames % tokenizer.spatial_compression != 0:
        print(
            f"[bandtok] WARNING: sample_size={sample_size} gives {frames} Mel frames, which is not a "
            f"multiple of spatial_compression={tokenizer.spatial_compression}; the reconstruction is "
            "shorter than the target and both are cropped to the common length"
        )

    loader = create_dataloader_from_config(
        load_data_config(args.data_config),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=sample_rate,
        sample_size=sample_size,
        audio_channels=model_config.get("audio_channels", 1),
        shuffle=True,  # the paper evaluates *random* segments of the test set
        distributed=False,
        seed=args.seed,
        drop_last=False,
    )

    have_vocoder, vocoder_error = try_load_vocoder(tokenizer, device)

    # Spectral distances are evaluated at the resolution of the model's own Mel
    # front end, so the numbers line up with the training objective. The mel
    # filterbank inside MelSTFTLoss is librosa's default (fmin=0, fmax=sr/2),
    # which is also what LogMelSpectrogram uses for BandTok.
    stft_settings = {
        "fft_size": tokenizer.mel.n_fft,
        "hop_size": tokenizer.mel.hop_size,
        "win_length": tokenizer.mel.win_size,
        "window": "hann_window",
        "w_sc": 1.0,
        "w_log_mag": 1.0,
        "w_lin_mag": 0.0,
        "mag_distance": "L1",
        "reduction": "mean",
    }
    mel_settings = dict(stft_settings, n_mels=tokenizer.mel.num_mels, sample_rate=sample_rate, scale="mel")

    mel_loss: Optional[MelSTFTLoss] = None
    stft_loss: Optional[STFTLoss] = None
    if have_vocoder:
        mel_loss = MelSTFTLoss(
            sample_rate=sample_rate,
            fft_size=stft_settings["fft_size"],
            hop_size=stft_settings["hop_size"],
            win_length=stft_settings["win_length"],
            n_mels=tokenizer.mel.num_mels,
            device=device,
        ).to(device)
        stft_loss = STFTLoss(
            fft_size=stft_settings["fft_size"],
            hop_size=stft_settings["hop_size"],
            win_length=stft_settings["win_length"],
        ).to(device)

    print(
        f"[bandtok] evaluating {args.num_segments} segments of "
        f"{sample_size / sample_rate:.2f}s on {device}"
    )
    metrics, stats, seen = evaluate(
        tokenizer, loader, args.num_segments, device, mel_loss, stft_loss
    )

    results: Dict[str, Any] = {
        "model_config": os.path.abspath(args.model_config),
        "ckpt": os.path.abspath(args.ckpt),
        "data_config": os.path.abspath(args.data_config),
        "device": str(device),
        "seed": args.seed,
        "num_segments": seen,
        "requested_num_segments": args.num_segments,
        "sample_rate": sample_rate,
        "sample_size": sample_size,
        "segment_seconds": sample_size / sample_rate,
        "mel_domain": {
            "logmel_l1": metrics["logmel_l1"],
            "description": "L1 between the input and reconstructed log-Mel spectrogram (no vocoder)",
            "mel_settings": {
                "n_fft": tokenizer.mel.n_fft,
                "num_mels": tokenizer.mel.num_mels,
                "hop_size": tokenizer.mel.hop_size,
                "win_size": tokenizer.mel.win_size,
                "fmin": tokenizer.mel.fmin,
                "fmax": tokenizer.mel.fmax,
            },
        },
        "waveform_domain": {
            "available": have_vocoder,
            "reason": vocoder_error,
            "mel_distance": metrics["mel_distance"] if have_vocoder else None,
            "stft_distance": metrics["stft_distance"] if have_vocoder else None,
            "mel_distance_module": "bandtok.losses.spectral.MelSTFTLoss",
            "stft_distance_module": "bandtok.losses.spectral.STFTLoss",
            "mel_settings": mel_settings,
            "stft_settings": stft_settings,
            "description": (
                "spectral distances between the original waveform and "
                "vocoder(decoder(quantize(encoder(log-Mel)))); includes the vocoder's own error"
            ),
        },
        "tokens": dict(stats.summary(), num_bands=tokenizer.num_bands),
        "rate": {
            "token_rate": tokenizer.token_rate,
            "frame_rate": tokenizer.frame_rate,
            "bitrate_kbps": tokenizer.bitrate / 1000.0,
        },
    }

    print_summary(results)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"[bandtok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
