#!/usr/bin/env python3
"""Reconstruct the DAC-style eval set with the BandTok tokenizer.

librosa mono load at 44.1 kHz (no peak normalization), full-file log-Mel,
encode -> quantize -> decode_mel, BigVGAN-v2 vocoder, 16-bit PCM output mirroring
the input directory structure. The paper's Table II numbers are computed on these
outputs with codec_evaluation.py (DAC losses at 24 kHz).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import librosa
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandtok.config import load_model_config  # noqa: E402
from bandtok.models.factory import create_tokenizer_from_config  # noqa: E402
from bandtok.utils.checkpoint import load_model_weights  # noqa: E402


def _write_pcm16(path: Path, wav_gen: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_int16 = (wav_gen.squeeze().cpu() * 32767.0).numpy().astype("int16")
    sf.write(str(path), wav_int16, sample_rate, subtype="PCM_16")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocoder-path", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_config = load_model_config(args.model_config)
    if args.vocoder_path:
        model_config["model"]["vocoder_path"] = args.vocoder_path

    tokenizer = create_tokenizer_from_config(model_config)
    load_model_weights(tokenizer, args.ckpt, strict=False, name="tokenizer")
    tokenizer = tokenizer.to(device).eval()
    tokenizer.requires_grad_(False)
    tokenizer.vocoder.to(device)

    sr_target = model_config["sample_rate"]
    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    wavs = sorted(p for p in input_root.rglob("*.wav"))
    print(f"[reconstruct_dacset] {len(wavs)} files")

    for i, wav_path in enumerate(wavs, 1):
        rel = wav_path.relative_to(input_root)

        # Mono, resampled, NOT normalized.
        wav, sr = librosa.load(str(wav_path), sr=sr_target, mono=True)
        audio = torch.from_numpy(wav).float().unsqueeze(0).to(device)  # [1, T]

        with torch.no_grad():
            logmel = tokenizer.mel(audio)  # [1, 1, F, T]
            latents = tokenizer.encode_mel(logmel)  # quantized
            recon_mel = tokenizer.decode_mel(latents, skip_bottleneck=True)
            _write_pcm16(output_root / rel, tokenizer.decode_by_vocoder(recon_mel), sr_target)

        if i % 100 == 0 or i == len(wavs):
            print(f"[reconstruct_dacset] {i}/{len(wavs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
