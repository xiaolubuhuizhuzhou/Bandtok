#!/usr/bin/env python3
"""Average a `codec_evaluation.py` metrics.csv over the DAC eval set.

Mel/STFT are plain means over all pairs -- those are the paper's Table II columns.
SI-SDR gets a second, filtered mean: 51 of the 1000 MUSDB references are digital
silence, where a scale-invariant ratio is either NaN (bit-exact zero reconstruction)
or a meaningless +80 dB (reconstruction that merely rounds to a few LSB), so the
all-pairs SI-SDR mean is not comparable between models. `sisdr-24k_clean` is the
mean over the non-silent references only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Set

import soundfile as sf

MEL_STFT_COLS = ["mel-24k", "stft-24k", "mel-44k", "stft-44k"]


def silent_refs(input_root: Path, cache: Path) -> Set[str]:
    """Relative paths of references that are digital silence, cached across runs."""
    if cache.exists():
        return set(json.loads(cache.read_text()))

    silent: List[str] = []
    wavs = sorted(p for p in input_root.rglob("*.wav"))
    for wav_path in wavs:
        audio, _ = sf.read(str(wav_path), dtype="float32")
        if not audio.any():
            silent.append(str(wav_path.relative_to(input_root)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sorted(silent), indent=1) + "\n")
    print(f"[aggregate] {len(silent)}/{len(wavs)} references are digital silence -> {cache}")
    return set(silent)


def aggregate(metrics_csv: Path, silent: Set[str]) -> Dict[str, float]:
    rows = list(csv.DictReader(metrics_csv.open()))
    out: Dict[str, float] = {}
    for col in MEL_STFT_COLS:
        values = [float(r[col]) for r in rows]
        out[col] = sum(values) / len(values)

    clean = [
        float(r["sisdr-24k"])
        for r in rows
        if r["rel_path"] not in silent and not math.isnan(float(r["sisdr-24k"]))
    ]
    out["sisdr-24k_clean"] = sum(clean) / len(clean)
    out["n_pairs"] = len(rows)
    out["n_sisdr_clean"] = len(clean)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", nargs="+", help="one or more metrics.csv to average")
    parser.add_argument(
        "--input-dir",
        required=True,
        help="directory of DAC-eval-set reference wavs",
    )
    parser.add_argument("--cache", default="results/dacset_silent_refs.json")
    parser.add_argument("--json-out", default=None, help="write {label: metrics} here")
    parser.add_argument("--label", action="append", default=None, help="label per csv, in order")
    args = parser.parse_args()

    silent = silent_refs(Path(args.input_dir), Path(args.cache))
    labels = args.label or [str(Path(p).parent) for p in args.metrics_csv]
    if len(labels) != len(args.metrics_csv):
        raise ValueError(f"{len(labels)} labels for {len(args.metrics_csv)} csv files")

    results = {label: aggregate(Path(path), silent) for label, path in zip(labels, args.metrics_csv)}
    for label, metrics in results.items():
        cols = "  ".join(f"{c}={metrics[c]:.4f}" for c in MEL_STFT_COLS)
        print(
            f"{label}: {cols}  sisdr-24k_clean={metrics['sisdr-24k_clean']:.2f} "
            f"(n={metrics['n_pairs']}, sisdr n={metrics['n_sisdr_clean']})"
        )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=1) + "\n")
        print(f"[aggregate] -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
