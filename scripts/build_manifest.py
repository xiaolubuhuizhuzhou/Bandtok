#!/usr/bin/env python3
"""Build a JSONL manifest for :class:`bandtok.data.dataset.AudioManifestDataset`.

Example::

    python scripts/build_manifest.py \
        --audio-dir /data/mtg_jamendo/wav \
        --captions /data/mtg_jamendo/captions.jsonl \
        --relative-to /data/mtg_jamendo/wav \
        --output /data/mtg_jamendo/train.jsonl \
        --extensions wav,flac,mp3 --num-workers 16

Each output line looks like::

    {"path": "0/973751.wav", "caption": "...", "duration": 46.68}

Captions may come from a JSONL file (any of ``path``/``file_path``/``id`` plus
any of ``caption``/``prompt``/``text``) or from a two-column CSV
(``path,caption``). Keys are matched against the full path, the path relative to
``--audio-dir`` and the file stem, in that order.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

PATH_KEYS = ("path", "file_path", "filepath", "audio_path", "audio_file", "filename", "id", "idx")
CAPTION_KEYS = ("caption", "prompt", "text", "description")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", required=True, help="directory scanned recursively for audio files")
    parser.add_argument("--output", required=True, help="output JSONL manifest")
    parser.add_argument("--captions", default=None, help="optional JSONL or CSV file with captions")
    parser.add_argument(
        "--extensions",
        default="wav,flac,mp3",
        help="comma-separated list of audio extensions (default: wav,flac,mp3)",
    )
    parser.add_argument(
        "--relative-to",
        default=None,
        help="store paths relative to this directory instead of absolute paths",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="threads used to probe durations")
    parser.add_argument(
        "--skip-duration",
        action="store_true",
        help="do not probe durations (the dataset will then probe files at load time)",
    )
    return parser.parse_args(argv)


def find_audio_files(audio_dir: str, extensions: List[str]) -> List[str]:
    """Recursively list audio files under ``audio_dir``, sorted by path."""
    suffixes = {("." + ext.lstrip(".")).lower() for ext in extensions}
    files: List[str] = []
    for dirpath, _, filenames in os.walk(audio_dir):
        for name in filenames:
            if Path(name).suffix.lower() in suffixes:
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def load_captions(captions_path: str) -> Dict[str, str]:
    """Load a caption lookup table from a JSONL or CSV file."""
    captions: Dict[str, str] = {}
    if captions_path.lower().endswith(".csv"):
        with open(captions_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                if row[0].strip().lower() in PATH_KEYS:  # header line
                    continue
                captions[row[0].strip()] = row[1].strip()
        return captions

    with open(captions_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, (list, tuple)) and len(record) == 2 and isinstance(record[1], dict):
                key, record = str(record[0]), record[1]
            else:
                key = None
            if not isinstance(record, dict):
                continue
            caption = next((record[k] for k in CAPTION_KEYS if isinstance(record.get(k), str)), None)
            if caption is None:
                continue
            for path_key in PATH_KEYS:
                value = record.get(path_key)
                if isinstance(value, str) and value:
                    captions[value] = caption
                    captions[Path(value).stem] = caption
            if key is not None:
                captions.setdefault(key, caption)
    return captions


def lookup_caption(captions: Dict[str, str], abs_path: str, audio_dir: str) -> str:
    """Find the caption for one file, trying the usual key spellings."""
    if not captions:
        return ""
    candidates = [abs_path, os.path.relpath(abs_path, audio_dir), Path(abs_path).name, Path(abs_path).stem]
    for candidate in candidates:
        if candidate in captions:
            return captions[candidate]
    return ""


def probe_duration(path: str) -> Optional[float]:
    """Return the duration of ``path`` in seconds, or ``None`` on failure."""
    try:
        import soundfile as sf

        info = sf.info(path)
        if info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import torchaudio

        info = torchaudio.info(path)
        if info.sample_rate > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass
    return None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not os.path.isdir(args.audio_dir):
        raise ValueError(f"--audio-dir {args.audio_dir!r} is not a directory")

    extensions = [ext for ext in args.extensions.split(",") if ext.strip()]
    files = find_audio_files(args.audio_dir, extensions)
    print(f"Found {len(files)} audio files under {args.audio_dir}")
    if not files:
        raise ValueError(f"No audio files with extensions {extensions} under {args.audio_dir!r}")

    captions = load_captions(args.captions) if args.captions else {}
    if args.captions:
        print(f"Loaded {len(captions)} caption keys from {args.captions}")

    durations: List[Optional[float]] = [None] * len(files)
    if not args.skip_duration:
        workers = max(1, args.num_workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            durations = list(pool.map(probe_duration, files))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    written = skipped = 0
    total_seconds = 0.0
    captioned = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for path, duration in zip(files, durations):
            if not args.skip_duration and duration is None:
                print(f"  skipping unreadable file: {path}", file=sys.stderr)
                skipped += 1
                continue
            stored_path = os.path.relpath(path, args.relative_to) if args.relative_to else os.path.abspath(path)
            record: Dict[str, object] = {"path": stored_path}
            caption = lookup_caption(captions, path, args.audio_dir)
            if caption:
                record["caption"] = caption
                captioned += 1
            if duration is not None:
                record["duration"] = round(duration, 3)
                total_seconds += duration
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} entries to {args.output}")
    print(f"  with captions: {captioned}")
    print(f"  skipped files: {skipped}")
    print(f"  total hours:   {total_seconds / 3600.0:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
