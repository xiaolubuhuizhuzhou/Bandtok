#!/usr/bin/env python
"""Tokenize audio with BandTok and resynthesize it.

This is the tokenizer's inference path -- the one used for the paper's
reconstruction comparisons: audio -> log-Mel -> tokens -> log-Mel -> BigVGAN.

    python scripts/reconstruct.py --model-config configs/tokenizer/bandtok.json \\
        --ckpt ckpts/bandtok.pt --input-dir /data/musdb/test --output-dir recon \\
        --max-seconds 10 --save-tokens

Inputs may be given as individual files (``--input``, repeatable), a directory
(``--input-dir``) or a JSONL manifest (``--manifest``, the same format
``bandtok.data.dataset`` reads).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio

from bandtok.config import load_model_config
from bandtok.data.dataset import load_manifest
from bandtok.models.factory import create_model_from_config
from bandtok.models.tokenizer import BandTok
from bandtok.utils.checkpoint import load_model_weights


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True, help="tokenizer model config (JSON)")
    parser.add_argument("--ckpt", required=True, help="trained tokenizer weights")
    parser.add_argument("--output-dir", required=True, help="directory for the reconstructed wavs")

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", action="append", default=None, help="audio file to reconstruct (repeatable)")
    inputs.add_argument("--input-dir", default=None, help="directory scanned recursively for audio files")
    inputs.add_argument("--manifest", default=None, help="JSONL manifest (file, directory or glob)")

    parser.add_argument(
        "--extensions",
        default="wav,flac,mp3",
        help="comma-separated extensions used with --input-dir (default: wav,flac,mp3)",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="files encoded together (default: 1)")
    parser.add_argument("--max-seconds", type=float, default=None, help="truncate every input to N seconds")
    parser.add_argument(
        "--save-tokens",
        action="store_true",
        help="also write the token grid as a .npy of shape [num_quantizers, num_tokens]",
    )
    parser.add_argument(
        "--vocoder-path",
        default=None,
        help="local BigVGAN directory or HF id (default: $BANDTOK_VOCODER_PATH, else the config's)",
    )
    parser.add_argument("--device", default=None, help="torch device (default: cuda when available)")
    return parser


# -- inputs ---------------------------------------------------------------


def collect_inputs(args: argparse.Namespace) -> List[str]:
    """Resolve the input flags into a list of audio paths."""
    if args.input:
        paths = list(args.input)
    elif args.input_dir:
        suffixes = {"." + ext.strip().lstrip(".").lower() for ext in args.extensions.split(",") if ext.strip()}
        paths = sorted(
            os.path.join(dirpath, name)
            for dirpath, _, filenames in os.walk(args.input_dir)
            for name in filenames
            if Path(name).suffix.lower() in suffixes
        )
    else:
        paths = [entry["path"] for entry in load_manifest(args.manifest)]

    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input(s) do not exist, first: {missing[0]}")
    if not paths:
        raise ValueError("no input audio files found")
    return paths


def load_audio(path: str, sample_rate: int, max_seconds: Optional[float] = None) -> torch.Tensor:
    """Load one file as mono ``[1, N]`` float audio at ``sample_rate``."""
    audio, source_rate = torchaudio.load(path)
    audio = audio.to(torch.float32)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    if max_seconds is not None:
        audio = audio[:, : max(1, int(max_seconds * sample_rate))]
    return audio


def align_audio(tokenizer: BandTok, audio: torch.Tensor, name: str) -> torch.Tensor:
    """Trim ``audio`` so its Mel frame count is a multiple of ``spatial_compression``.

    The encoder halves the time axis three times, so an unaligned input comes back
    from the decoder shorter than it went in. :meth:`BandTok.aligned_sample_size`
    gives the largest usable length, and rejects a file too short to encode at all.
    """
    num_samples = int(audio.shape[-1])
    try:
        aligned = tokenizer.aligned_sample_size(num_samples)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc
    if aligned != num_samples:
        print(
            f"[bandtok] {name}: trimming {num_samples} -> {aligned} samples so the "
            f"{tokenizer.mel.num_frames(aligned)} Mel frames are a multiple of "
            f"spatial_compression={tokenizer.spatial_compression}"
        )
    return audio[..., :aligned]


# -- reconstruction -------------------------------------------------------


@torch.inference_mode()
def reconstruct_batch(
    tokenizer: BandTok, audio: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Reconstruct a batch of aligned mono audio ``[B, 1, N]``.

    Returns ``(waveforms [B, 1, N], codes [B, num_quantizers, L], mel_l1)`` where
    ``mel_l1`` is the per-item L1 distance between the input and the reconstructed
    log-Mel spectrogram.
    """
    logmel = tokenizer.mel(audio)
    codes = tokenizer.tokenize(audio)
    logmel_hat = tokenizer.decode_codes(codes)
    mel_l1 = (logmel - logmel_hat).abs().flatten(1).mean(dim=1)
    return tokenizer.decode_by_vocoder(logmel_hat), codes, [float(value) for value in mel_l1]


def _unique_stem(stem: str, used: Dict[str, int]) -> str:
    """Disambiguate file stems that repeat across input directories."""
    count = used.get(stem, 0)
    used[stem] = count + 1
    return stem if count == 0 else f"{stem}-{count}"


def main() -> None:
    args = build_argparser().parse_args()

    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_config = load_model_config(args.model_config)
    if model_config["model_type"] != "bandtok":
        raise ValueError(
            f"--model-config must be a 'bandtok' tokenizer config, got {model_config['model_type']!r}; "
            "use scripts/generate.py for an LM config"
        )

    tokenizer: BandTok = create_model_from_config(model_config)
    load_model_weights(tokenizer, args.ckpt, strict=False, name="tokenizer")
    if args.vocoder_path:
        # The flag has to win over a path baked into the tokenizer config.
        os.environ["BANDTOK_VOCODER_PATH"] = args.vocoder_path
        tokenizer.set_vocoder_path(args.vocoder_path)
    tokenizer = tokenizer.to(device).eval()
    tokenizer.requires_grad_(False)

    paths = collect_inputs(args)
    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"[bandtok] reconstructing {len(paths)} file(s) at {tokenizer.sample_rate} Hz on {device}; "
        f"{tokenizer.token_rate:.1f} tokens/s, {tokenizer.bitrate / 1000:.2f} kbps"
    )

    used_stems: Dict[str, int] = {}
    distances: List[float] = []
    pending: List[Tuple[str, torch.Tensor]] = []

    def flush() -> None:
        """Reconstruct and write everything buffered in ``pending``."""
        if not pending:
            return
        batch = torch.cat([audio.unsqueeze(0) for _, audio in pending], dim=0).to(device)
        waveforms, codes, mel_l1 = reconstruct_batch(tokenizer, batch)
        for index, (stem, _) in enumerate(pending):
            wav_path = os.path.join(args.output_dir, f"{stem}_recon.wav")
            wav = waveforms[index].detach().float().clamp(-1.0, 1.0).cpu()
            torchaudio.save(wav_path, wav, tokenizer.sample_rate)
            if args.save_tokens:
                token_grid = codes[index].detach().cpu().numpy().astype(np.int32)
                np.save(os.path.join(args.output_dir, f"{stem}_tokens.npy"), token_grid)
            distances.append(mel_l1[index])
            print(f"[bandtok] {stem}: mel L1 {mel_l1[index]:.4f}, {codes.shape[-1]} tokens -> {wav_path}")
        pending.clear()

    for path in paths:
        audio = align_audio(tokenizer, load_audio(path, tokenizer.sample_rate, args.max_seconds), path)
        # Only equal-length inputs share a batch, so batching never pads and gives
        # exactly the same result as --batch-size 1.
        if pending and audio.shape[-1] != pending[0][1].shape[-1]:
            flush()
        pending.append((_unique_stem(Path(path).stem, used_stems), audio))
        if len(pending) == args.batch_size:
            flush()
    flush()

    print(
        f"[bandtok] {len(distances)} file(s), mean mel L1 "
        f"{sum(distances) / max(1, len(distances)):.4f}"
    )


if __name__ == "__main__":
    main()
