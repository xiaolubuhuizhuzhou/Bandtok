#!/usr/bin/env python3
"""Remove near-silent files from a JSONL manifest, offline.

The training dataset skips crops whose mean absolute amplitude is below
``silence_threshold`` (see ``AudioManifestDataset``). Files that are quiet
throughout are skipped *every* time they are sampled, wasting a full network
read per attempt, epoch after epoch. This script performs that check once:
it reads a few evenly spaced crops per file and drops entries where every
sampled crop is below the threshold.

Kept lines are written verbatim, so any manifest format the dataset accepts
(including the ``[id, {...}]`` pair format) passes through unchanged.
Dropped entries are recorded in ``<output>.dropped.jsonl`` for inspection.

Example::

    python scripts/filter_silent_manifest.py \
        --manifest /data/fsd/fsd_sao_prompt.jsonl \
        --output data/manifests/fsd_sao_prompt.silencefiltered.jsonl \
        --num-workers 96
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandtok.data.dataset import _parse_record, resolve_manifest_paths  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="manifest file, directory or glob (JSONL)")
    parser.add_argument("--output", required=True, help="filtered JSONL manifest to write")
    parser.add_argument("--root", default=None, help="root joined with relative manifest paths")
    parser.add_argument("--sample-rate", type=int, default=44100, help="training sample rate")
    parser.add_argument("--sample-size", type=int, default=65024, help="training crop length in samples")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="mean absolute amplitude below which a crop counts as silent "
        "(must match the dataset's silence_threshold)",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=10,
        help="crops probed per file; a file is dropped only if ALL of them are silent",
    )
    parser.add_argument("--num-workers", type=int, default=64, help="reader threads")
    return parser.parse_args(argv)


def crop_amplitudes(path: str, crop_frames: int, num_frames: int, max_crops: int, threshold: float) -> float:
    """Max mean-abs amplitude over evenly spaced crops, early-exiting once one passes."""
    max_offset = max(0, num_frames - crop_frames)
    num_crops = min(max_crops, max(1, num_frames // crop_frames))
    if num_crops == 1:
        offsets = [max_offset // 2]
    else:
        offsets = [round(i * max_offset / (num_crops - 1)) for i in range(num_crops)]

    loudest = 0.0
    for offset in offsets:
        data, _ = sf.read(path, start=offset, stop=offset + crop_frames, dtype="float32", always_2d=True)
        if data.size == 0:
            continue
        amplitude = float(np.abs(data).mean())
        loudest = max(loudest, amplitude)
        if loudest >= threshold:
            break  # loud enough; no need to read further crops
    return loudest


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    lines: List[str] = []
    for manifest_file in resolve_manifest_paths(args.manifest):
        with open(manifest_file, "r", encoding="utf-8") as handle:
            lines.extend(line.rstrip("\n") for line in handle if line.strip())
    print(f"[filter] {len(lines)} lines from {args.manifest}")

    counters = {"kept": 0, "dropped": 0, "short": 0, "noaudio": 0, "errors": 0, "done": 0}
    lock = threading.Lock()
    start = time.time()

    def check(line: str) -> Tuple[bool, str, Optional[str], float]:
        """Return (keep, line, dropped_path, amplitude)."""
        try:
            entry = _parse_record(json.loads(line))
        except Exception:
            entry = None
        key = "kept"
        keep, dropped_path, amplitude = True, None, 0.0
        try:
            if entry is None:
                key = "noaudio"  # header lines etc.; the dataset ignores them too
            else:
                path = entry["path"]
                if args.root and not os.path.isabs(path):
                    path = os.path.join(args.root, path)
                source_rate, num_frames = entry.get("sample_rate"), entry.get("num_frames")
                if not source_rate or not num_frames:
                    info = sf.info(path)
                    source_rate, num_frames = int(info.samplerate), int(info.frames)
                crop_frames = max(1, int(args.sample_size * source_rate / args.sample_rate))
                if num_frames < crop_frames:
                    # The dataset already drops these from metadata at init, at no
                    # runtime cost; keep the line so the manifest stays general.
                    key = "short"
                else:
                    amplitude = crop_amplitudes(path, crop_frames, num_frames, args.max_crops, args.threshold)
                    if amplitude < args.threshold:
                        keep, dropped_path, key = False, path, "dropped"
        except Exception as exc:
            key = "errors"  # unreadable here; keep it and let training decide
            print(f"[filter] error on {entry['path'] if entry else line[:80]}: {exc}", file=sys.stderr)
        with lock:
            counters[key] += 1
            counters["done"] += 1
            if counters["done"] % 20000 == 0:
                rate = counters["done"] / max(1e-6, time.time() - start)
                print(
                    f"[filter] {counters['done']}/{len(lines)} ({rate:.0f} files/s, "
                    f"dropped {counters['dropped']})",
                    flush=True,
                )
        return keep, line, dropped_path, amplitude

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    dropped_output = args.output + ".dropped.jsonl"
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool, open(
        args.output, "w", encoding="utf-8"
    ) as out, open(dropped_output, "w", encoding="utf-8") as dropped_out:
        for keep, line, dropped_path, amplitude in pool.map(check, lines):
            if keep:
                out.write(line + "\n")
            else:
                dropped_out.write(json.dumps({"path": dropped_path, "max_crop_amplitude": amplitude}) + "\n")

    elapsed = time.time() - start
    print(
        f"[filter] wrote {counters['kept'] + counters['short'] + counters['noaudio'] + counters['errors']} "
        f"lines to {args.output} in {elapsed / 60:.1f} min\n"
        f"[filter]   dropped (all crops silent): {counters['dropped']} -> {dropped_output}\n"
        f"[filter]   kept short entries: {counters['short']}, non-audio lines: {counters['noaudio']}, "
        f"read errors (kept): {counters['errors']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
