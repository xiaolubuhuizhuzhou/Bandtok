"""Manifest-based audio dataset for BandTok.

A dataset is described by one JSONL manifest per split. Every line is a JSON
object with the keys

    {"path": "relative/or/absolute/audio.wav",
     "caption": "a text description of the clip",   # optional
     "duration": 187.4}                             # optional, seconds

``path`` is resolved against a per-dataset ``root`` when it is relative.
``caption`` feeds the text conditioner of the music language model.
``duration``, when present, is used to filter short files and to build the
``seconds_total`` conditioning value without opening the audio file.

Alternative manifest layouts are also accepted: alias key
names (``file_path``/``prompt``/...), nested ``audio_meta`` blocks and lines of
the form ``[file_id, {...}]`` are all understood. See ``_parse_record``.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchaudio import transforms as T

__all__ = [
    "AudioManifestDataset",
    "collate_batch",
    "create_dataloader_from_config",
    "load_manifest",
]

# Accepted key aliases, in priority order.
PATH_KEYS: Tuple[str, ...] = ("path", "file_path", "filepath", "audio_path", "audio_file", "filename")
CAPTION_KEYS: Tuple[str, ...] = ("caption", "prompt", "text", "description")
DURATION_KEYS: Tuple[str, ...] = ("duration", "duration_sec", "seconds", "seconds_total", "length_sec")

Info = Dict[str, Any]


class SkipEntry(Exception):
    """A dataset entry that is loadable but unusable (e.g. silent or too short).

    Raised by ``_load_entry`` so ``__getitem__`` can report *why* an item was
    skipped, distinct from a genuine decode/read error.
    """


# ---------------------------------------------------------------------------
# manifest parsing
# ---------------------------------------------------------------------------


def _first_present(record: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    """Return the value of the first key of ``keys`` present in ``record``."""
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _parse_record(record: Any) -> Optional[Info]:
    """Convert one decoded JSONL line into a normalised entry.

    Returns ``None`` for lines that carry no audio path (for example the
    ``{"feat_root": ...}`` header line of the pair-format manifests).
    """
    # Pair format: ["<file_id>", {"file_path": ..., "audio_meta": {...}}]
    file_id: Optional[str] = None
    if isinstance(record, (list, tuple)):
        if len(record) != 2 or not isinstance(record[1], dict):
            return None
        file_id, record = str(record[0]), record[1]

    if not isinstance(record, dict):
        return None

    path = _first_present(record, PATH_KEYS)
    if not isinstance(path, str) or not path:
        return None

    caption = _first_present(record, CAPTION_KEYS)
    caption = caption if isinstance(caption, str) else ""

    duration = _first_present(record, DURATION_KEYS)
    sample_rate = record.get("sample_rate")
    num_frames = record.get("num_frames")

    # Nested block: {"audio_meta": {"sample_rate": .., "num_frames": .., "channels": ..}}
    audio_meta = record.get("audio_meta")
    if isinstance(audio_meta, dict):
        sample_rate = sample_rate or audio_meta.get("sample_rate")
        num_frames = num_frames or audio_meta.get("num_frames")

    if duration is None and sample_rate and num_frames:
        duration = float(num_frames) / float(sample_rate)

    entry: Info = {
        "path": path,
        "caption": caption,
        "duration": float(duration) if duration is not None else None,
        "sample_rate": int(sample_rate) if sample_rate else None,
        "num_frames": int(num_frames) if num_frames else None,
    }
    if file_id is not None:
        entry["id"] = file_id
    # Optional fixed crop offset (in source-rate frames), honoured by the loader.
    start_idx = record.get("start_idx")
    if isinstance(start_idx, int) and start_idx > 0:
        entry["start_idx"] = start_idx
    return entry


def resolve_manifest_paths(manifest: str) -> List[str]:
    """Expand a manifest argument into a sorted list of JSONL files.

    ``manifest`` may be a single file, a directory (all files inside are used)
    or a glob pattern such as ``"meta/*"``.
    """
    if os.path.isdir(manifest):
        matches = sorted(glob.glob(os.path.join(manifest, "*")))
    else:
        matches = sorted(glob.glob(str(manifest)))
    matches = [m for m in matches if os.path.isfile(m)]
    if not matches:
        raise ValueError(f"No manifest file matches {manifest!r}")
    return matches


def load_manifest(
    manifest: str,
    root: Optional[str] = None,
    dataset_id: Optional[str] = None,
) -> List[Info]:
    """Read one manifest (file, directory or glob) into a list of entries.

    Relative ``path`` values are joined with ``root``.
    """
    entries: List[Info] = []
    for manifest_file in resolve_manifest_paths(manifest):
        with open(manifest_file, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.warn(f"{manifest_file}:{line_no}: malformed JSON ({exc}); skipping")
                    continue
                entry = _parse_record(record)
                if entry is None:
                    continue
                if root and not os.path.isabs(entry["path"]):
                    entry["path"] = os.path.join(root, entry["path"])
                if dataset_id is not None:
                    entry["dataset_id"] = dataset_id
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------


class AudioManifestDataset(Dataset):
    """Random-crop audio dataset driven by JSONL manifests.

    Each item is a fixed-length crop of one track, returned as
    ``(audio, info)`` where ``audio`` has shape ``[audio_channels, sample_size]``
    and ``info`` carries the metadata used for conditioning:

    ``path``, ``caption``, ``seconds_start`` (crop offset in seconds),
    ``seconds_total`` (duration of the whole track in seconds),
    ``padding_mask`` (``1`` on valid samples, ``0`` on zero padding) and
    ``timestamps`` (relative ``(start, end)`` position inside the track).

    Args:
        manifests: list of ``{"id": str, "manifest": str, "root": str}`` dicts.
        sample_rate: target sample rate; audio is resampled when needed.
        sample_size: crop length in samples.
        audio_channels: number of channels of the returned tensor (1 or 2).
        random_crop: sample the crop offset at random, else take a centre crop.
        drop_short: drop tracks shorter than ``sample_size`` when the duration
            is known, instead of zero padding them.
        force_channels: ``"mono"``, ``"stereo"`` or ``"none"``.
        peak_normalize: divide each crop by its peak amplitude.
        silence_threshold: crops whose mean absolute amplitude is below this
            value are treated as silent and replaced by another item.
        max_retries: number of alternative items tried after a decode failure.
    """

    def __init__(
        self,
        manifests: Sequence[Dict[str, Any]],
        sample_rate: int = 44100,
        sample_size: int = 65024,
        audio_channels: int = 1,
        random_crop: bool = True,
        drop_short: bool = True,
        force_channels: str = "mono",
        peak_normalize: bool = True,
        silence_threshold: float = 1e-3,
        max_retries: int = 10,
    ) -> None:
        super().__init__()

        if force_channels not in ("mono", "stereo", "none"):
            raise ValueError(f"force_channels must be 'mono', 'stereo' or 'none', got {force_channels!r}")
        if audio_channels not in (1, 2):
            raise ValueError(f"audio_channels must be 1 or 2, got {audio_channels}")
        if sample_size <= 0:
            raise ValueError(f"sample_size must be positive, got {sample_size}")

        self.sample_rate = int(sample_rate)
        self.sample_size = int(sample_size)
        self.audio_channels = int(audio_channels)
        self.random_crop = bool(random_crop)
        self.drop_short = bool(drop_short)
        self.force_channels = force_channels
        self.peak_normalize = bool(peak_normalize)
        self.silence_threshold = float(silence_threshold)
        self.max_retries = int(max_retries)

        self.entries: List[Info] = []
        for spec in manifests:
            manifest = spec.get("manifest") or spec.get("meta")
            if not manifest:
                raise ValueError(f"Dataset entry {spec!r} has no 'manifest' key")
            entries = load_manifest(
                manifest,
                root=spec.get("root") or spec.get("path"),
                dataset_id=spec.get("id"),
            )
            print(f"[bandtok.data] {spec.get('id', manifest)}: {len(entries)} entries")
            self.entries.extend(entries)

        if self.drop_short:
            self.entries = self._filter_short(self.entries)

        if not self.entries:
            raise ValueError("Empty dataset: no usable entries found in the given manifests")
        print(f"[bandtok.data] total {len(self.entries)} entries")

        # One resampler per source sample rate, built lazily inside each worker.
        self._resamplers: Dict[int, T.Resample] = {}

    # -- construction helpers ------------------------------------------------

    def _filter_short(self, entries: Iterable[Info]) -> List[Info]:
        """Remove entries whose known duration is shorter than one crop."""
        min_duration = self.sample_size / self.sample_rate
        kept, dropped = [], 0
        for entry in entries:
            duration = entry.get("duration")
            if duration is not None and duration < min_duration:
                dropped += 1
                continue
            kept.append(entry)
        if dropped:
            print(f"[bandtok.data] drop_short: removed {dropped} tracks shorter than {min_duration:.2f}s")
        return kept

    # -- audio helpers -------------------------------------------------------

    def _resample(self, audio: torch.Tensor, source_rate: int) -> torch.Tensor:
        """Resample ``audio`` from ``source_rate`` to ``self.sample_rate``."""
        if source_rate == self.sample_rate:
            return audio
        resampler = self._resamplers.get(source_rate)
        if resampler is None:
            resampler = T.Resample(source_rate, self.sample_rate)
            self._resamplers[source_rate] = resampler
        return resampler(audio)

    def _apply_channels(self, audio: torch.Tensor) -> torch.Tensor:
        """Force the channel layout requested by ``force_channels``."""
        if self.force_channels == "mono":
            audio = audio.mean(dim=0, keepdim=True)
        elif self.force_channels == "stereo":
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
        if audio.shape[0] != self.audio_channels:
            if audio.shape[0] > self.audio_channels:
                audio = audio[: self.audio_channels]
            else:
                audio = audio.repeat(self.audio_channels // audio.shape[0], 1)
        return audio

    @staticmethod
    def _probe(path: str) -> Tuple[int, int]:
        """Return ``(sample_rate, num_frames)`` of an audio file."""
        try:
            meta = torchaudio.info(path)
            return int(meta.sample_rate), int(meta.num_frames)
        except Exception:
            import soundfile as sf

            meta = sf.info(path)
            return int(meta.samplerate), int(meta.frames)

    @staticmethod
    def _read(path: str, frame_offset: int, num_frames: int) -> Tuple[torch.Tensor, int]:
        """Read ``num_frames`` frames starting at ``frame_offset`` as ``[C, N]``."""
        try:
            audio, source_rate = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        except Exception:
            import soundfile as sf

            data, source_rate = sf.read(
                path, start=frame_offset, stop=frame_offset + num_frames, dtype="float32", always_2d=True
            )
            audio = torch.from_numpy(data.T)
        return audio.to(torch.float32), int(source_rate)

    # -- item loading --------------------------------------------------------

    def _load_entry(self, entry: Info) -> Tuple[torch.Tensor, Info]:
        """Load and crop one entry, or raise :class:`SkipEntry` if it is unusable."""
        path = entry["path"]

        source_rate = entry.get("sample_rate")
        num_frames = entry.get("num_frames")
        if not source_rate or not num_frames:
            source_rate, num_frames = self._probe(path)
        if num_frames <= 0:
            raise SkipEntry(f"no audio frames (num_frames={num_frames})")

        # Length of the crop expressed in source-rate frames.
        crop_frames = max(1, int(self.sample_size * source_rate / self.sample_rate))
        max_offset = max(0, num_frames - crop_frames)

        if self.drop_short and num_frames < crop_frames:
            raise SkipEntry(
                f"shorter than one crop ({num_frames / source_rate:.2f}s < "
                f"{self.sample_size / self.sample_rate:.2f}s)"
            )

        start_idx = entry.get("start_idx")
        if isinstance(start_idx, int) and 0 < start_idx <= max_offset:
            offset = start_idx
        elif max_offset == 0:
            offset = 0
        elif self.random_crop:
            offset = random.randint(0, max_offset)
        else:
            offset = max_offset // 2  # deterministic centre crop

        # Relative position of the crop inside the track, in [0, 1].
        full_length = max_offset + crop_frames
        t_start = offset / full_length
        t_end = (offset + crop_frames) / full_length

        # Segment-time conditioning: integer seconds (floor for the offset,
        # ceil for the total length).
        seconds_start = float(math.floor(offset / source_rate))
        seconds_total = float(math.ceil(num_frames / source_rate))

        audio, source_rate = self._read(path, offset, min(crop_frames, num_frames - offset))
        if audio.numel() == 0:
            raise SkipEntry(f"empty read at offset {offset}")
        if not torch.isfinite(audio).all():
            raise SkipEntry("non-finite samples (NaN/Inf)")

        audio = self._resample(audio, source_rate)
        crop_amplitude = audio.abs().mean().item()
        if crop_amplitude < self.silence_threshold:
            raise SkipEntry(
                f"near-silent crop (mean amplitude {crop_amplitude:.2e} < "
                f"silence_threshold {self.silence_threshold:.0e})"
            )
        if self.peak_normalize:
            peak = audio.abs().max()
            if peak <= 0:
                raise SkipEntry("all-zero crop")
            audio = audio / peak

        audio = self._apply_channels(audio)

        # Zero pad (or trim) to the fixed crop length.
        valid = min(audio.shape[-1], self.sample_size)
        chunk = audio.new_zeros([audio.shape[0], self.sample_size])
        chunk[:, :valid] = audio[:, :valid]
        chunk = chunk.clamp(-1.0, 1.0)

        padding_mask = torch.zeros([self.sample_size])
        padding_mask[:valid] = 1.0

        info: Info = {
            "path": path,
            "caption": entry.get("caption", "") or "",
            "seconds_start": seconds_start,
            "seconds_total": seconds_total,
            "padding_mask": padding_mask,
            "timestamps": (t_start, t_end),
        }
        if "dataset_id" in entry:
            info["dataset_id"] = entry["dataset_id"]
        # Kept for backwards compatibility with conditioners reading "prompt".
        info["prompt"] = info["caption"]
        return chunk, info

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Info]:
        for attempt in range(self.max_retries + 1):
            entry = self.entries[index]
            try:
                return self._load_entry(entry)
            except SkipEntry as skip:
                # Loadable but unusable content (silent / too short / non-finite).
                if attempt == 0:
                    # warnings.warn(f"Skipping unusable audio file {entry['path']}: {skip}")
                    continue
            except Exception as exc:  # corrupt or unreadable file
                if attempt == 0:
                    warnings.warn(f"Failed to load {entry['path']}: {type(exc).__name__}: {exc}")
            index = random.randrange(len(self.entries))
        raise RuntimeError(
            f"Could not load a usable item after {self.max_retries + 1} attempts; "
            "check the audio files referenced by the manifest"
        )


def collate_batch(items: Sequence[Tuple[torch.Tensor, Info]]) -> Tuple[torch.Tensor, List[Info]]:
    """Stack ``(audio, info)`` pairs into ``([B, C, N] tensor, [info, ...])``."""
    audios = torch.stack([audio for audio, _ in items])
    infos = [info for _, info in items]
    return audios, infos


# ---------------------------------------------------------------------------
# dataloader
# ---------------------------------------------------------------------------


def _normalise_dataset_specs(dataset_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract the per-dataset ``{id, manifest, root}`` specs from a config."""
    datasets = dataset_config.get("datasets")
    if not datasets:
        raise ValueError("Dataset config must contain a non-empty 'datasets' list")

    specs: List[Dict[str, Any]] = []
    for raw in datasets:
        manifest = raw.get("manifest") or raw.get("meta")
        if not manifest:
            raise ValueError(f"Dataset entry {raw!r} must define 'manifest' (or 'meta')")
        specs.append(
            {
                "id": raw.get("id"),
                "manifest": manifest,
                "root": raw.get("root") or raw.get("path"),
            }
        )
    return specs


def create_dataloader_from_config(
    dataset_config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    sample_rate: int,
    sample_size: int,
    audio_channels: int,
    shuffle: bool = True,
    distributed: bool = True,
    seed: int = 42,
    drop_last: bool = True,
) -> DataLoader:
    """Build a ``DataLoader`` over ``AudioManifestDataset`` from a JSON config.

    Expected config::

        {"dataset_type": "audio_manifest",
         "random_crop": true, "drop_short": true,
         "datasets": [{"id": "mtg_jamendo",
                       "manifest": "/path/meta.jsonl",
                       "root": "/path/audio"}]}

    ``"path"``/``"meta"`` are accepted as aliases of ``"root"``/``"manifest"``,
    and ``manifest`` may be a glob pattern or a directory.

    A ``DistributedSampler`` is used when ``distributed`` is true and the
    default process group is initialised, otherwise plain shuffling.
    """
    dataset_type = dataset_config.get("dataset_type", "audio_manifest")
    if dataset_type not in ("audio_manifest", "audio_dir"):
        raise ValueError(f"Unsupported dataset_type {dataset_type!r}; expected 'audio_manifest'")
    if audio_channels not in (1, 2):
        raise ValueError(f"audio_channels must be 1 or 2, got {audio_channels}")

    dataset = AudioManifestDataset(
        manifests=_normalise_dataset_specs(dataset_config),
        sample_rate=sample_rate,
        sample_size=sample_size,
        audio_channels=audio_channels,
        random_crop=dataset_config.get("random_crop", True),
        drop_short=dataset_config.get("drop_short", True),
        force_channels="mono" if audio_channels == 1 else "stereo",
    )

    use_distributed = distributed and torch.distributed.is_available() and torch.distributed.is_initialized()
    sampler: Optional[DistributedSampler] = None
    if use_distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        # Random 1.5 s crops from network storage are I/O-bound, so a deeper
        # prefetch queue keeps the GPUs fed through slow items (opus/mp3 decodes).
        prefetch_factor=4 if num_workers > 0 else None,
    )
