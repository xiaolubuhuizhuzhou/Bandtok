#!/usr/bin/env python3
"""Convert a manifest's compressed audio (opus/mp3/...) to wav, offline.

Compressed formats cannot be seek-read: loading a 1.5 s training crop decodes
the file from the start (measured ~1.3 s per fma mp3 vs ~0.15 s for a wav
partial read on network storage). Converting once makes every future epoch cheap.

For each manifest entry the source file is transcoded to 16-bit PCM wav
(native sample rate and channel count preserved), mirroring its path under
``--output-audio-dir``, and a new manifest is written with ``file_path`` and
``audio_meta`` updated. All other record fields (prompts, ids, the
``[id, {...}]`` pair format) pass through unchanged. Existing wav entries are
copied into the manifest as-is. Already-converted files are skipped, so the
script is resumable.

Example::

    python scripts/convert_manifest_to_wav.py \
        --manifest /path/meta_pt2/musdb18_pt2.jsonl \
        --source-root /path/to/mtg_jamendo/ops \
        --output-audio-dir /path/to/mtg_jamendo_pt2_wav \
        --output-manifest data/manifests/mtg_jamendo_pt2_wav.jsonl \
        --num-workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Tuple

import soundfile as sf

PATH_KEYS = ("path", "file_path", "filepath", "audio_path", "audio_file", "filename")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="input JSONL manifest")
    parser.add_argument("--source-root", required=True, help="prefix stripped from source paths")
    parser.add_argument("--output-audio-dir", required=True, help="directory the wav files are written to")
    parser.add_argument("--output-manifest", required=True, help="updated JSONL manifest to write")
    parser.add_argument("--num-workers", type=int, default=32, help="parallel ffmpeg processes")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary")
    return parser.parse_args(argv)


def record_path(record: Any) -> Tuple[Optional[dict], Optional[str], Optional[str]]:
    """Return ``(record_dict, path_key, path)`` for a manifest line's record."""
    if isinstance(record, (list, tuple)) and len(record) == 2 and isinstance(record[1], dict):
        record = record[1]
    if not isinstance(record, dict):
        return None, None, None
    for key in PATH_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            return record, key, value
    return record, None, None


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    with open(args.manifest, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle if line.strip()]
    print(f"[convert] {len(lines)} lines from {args.manifest}")

    counters = {"converted": 0, "skipped_existing": 0, "already_wav": 0, "failed": 0, "done": 0}
    lock = threading.Lock()
    start = time.time()

    def convert(line: str) -> Optional[str]:
        """Return the (possibly rewritten) output line, or ``None`` to drop it."""
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return line
        record, key, src = record_path(parsed)
        if record is None or key is None:
            return line  # header lines etc.
        outcome = "converted"
        try:
            if src.lower().endswith(".wav"):
                outcome = "already_wav"
                return line

            relative = os.path.relpath(src, args.source_root)
            if relative.startswith(".."):
                raise ValueError(f"{src} is not under --source-root {args.source_root}")
            dst = os.path.join(args.output_audio_dir, os.path.splitext(relative)[0] + ".wav")
            os.makedirs(os.path.dirname(dst), exist_ok=True)

            info = None
            if os.path.exists(dst):
                try:
                    info = sf.info(dst)
                    outcome = "skipped_existing"
                except Exception:
                    info = None  # partial file from an interrupted run; redo it
            if info is None:
                tmp = dst + ".tmp.wav"
                subprocess.run(
                    [args.ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", src, "-c:a", "pcm_s16le", tmp],
                    check=True,
                    capture_output=True,
                )
                os.replace(tmp, dst)
                info = sf.info(dst)

            record[key] = dst
            record["audio_meta"] = {
                "sample_rate": int(info.samplerate),
                "num_frames": int(info.frames),
                "channels": int(info.channels),
            }
            return json.dumps(parsed, ensure_ascii=False)
        except Exception as exc:
            outcome = "failed"
            detail = exc.stderr.decode(errors="replace").strip() if hasattr(exc, "stderr") and exc.stderr else exc
            print(f"[convert] FAILED {src}: {detail}", file=sys.stderr)
            return None  # a file that cannot be decoded here would fail in training too
        finally:
            with lock:
                counters[outcome] += 1
                counters["done"] += 1
                if counters["done"] % 500 == 0:
                    rate = counters["done"] / max(1e-6, time.time() - start)
                    remaining = (len(lines) - counters["done"]) / max(1e-6, rate)
                    print(
                        f"[convert] {counters['done']}/{len(lines)} ({rate:.1f} files/s, "
                        f"~{remaining / 60:.0f} min left, failed {counters['failed']})",
                        flush=True,
                    )

    os.makedirs(args.output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_manifest)) or ".", exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as pool, open(
        args.output_manifest, "w", encoding="utf-8"
    ) as out:
        for output_line in pool.map(convert, lines):
            if output_line is not None:
                out.write(output_line + "\n")

    elapsed = time.time() - start
    print(
        f"[convert] wrote {args.output_manifest} in {elapsed / 60:.1f} min\n"
        f"[convert]   converted: {counters['converted']}, resumed/skipped: {counters['skipped_existing']}, "
        f"already wav: {counters['already_wav']}, failed (dropped): {counters['failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
