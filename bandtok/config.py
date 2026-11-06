"""Configuration handling.

Every experiment is described by two JSON files:

* a **model config** — architecture plus the ``training`` block (optimizers,
  loss weights, checkpointing, logging);
* a **data config** — the dataset manifests to read.

JSON keeps configs diff-able and lets a checkpoint embed the exact config it was
trained with. This module only loads, validates and resolves them; the model
factories consume the resulting dicts.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Keys that may hold a path relative to the config file's own directory.
_PATH_KEYS = ("model_config", "ckpt_path", "vocoder_path")

#: repository root, i.e. the directory containing the ``bandtok`` package
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file, with the file name included in any error message."""
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def _resolve_path(value: str, base_dir: Path) -> str:
    """Resolve a possibly relative config path.

    Tried in order: as given (absolute or cwd-relative), relative to the config
    file's directory, then relative to the repository root — so a config can say
    ``configs/tokenizer/bandtok.json`` regardless of how deep it lives or where the
    command was run from. If none of them exist the value is returned expanded, so
    the eventual error names the path the user actually wrote.
    """
    expanded = os.path.expanduser(os.path.expandvars(value))
    if os.path.isabs(expanded):
        return expanded
    for candidate in (Path(expanded), base_dir / expanded, REPO_ROOT / expanded):
        if candidate.exists():
            return str(candidate.resolve())
    return expanded


def _resolve_relative_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Resolve path-valued keys so configs can reference each other portably."""
    for key, value in config.items():
        if isinstance(value, dict):
            _resolve_relative_paths(value, base_dir)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _resolve_relative_paths(item, base_dir)
        elif key in _PATH_KEYS and isinstance(value, str) and value:
            config[key] = _resolve_path(value, base_dir)
    return config


def load_model_config(path: str) -> Dict[str, Any]:
    """Load a model config and validate the keys the trainer relies on."""
    config = load_json(path)
    required = ("model_type", "sample_rate", "sample_size", "model", "training")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"model config {path} is missing required key(s): {', '.join(missing)}")
    if config["model_type"] not in ("bandtok", "bandtok_lm"):
        raise ValueError(
            f"unknown model_type {config['model_type']!r} in {path}; expected 'bandtok' or 'bandtok_lm'"
        )
    return _resolve_relative_paths(config, Path(path).resolve().parent)


def load_data_config(path: str) -> Dict[str, Any]:
    """Load a dataset config."""
    config = load_json(path)
    if "datasets" not in config:
        raise ValueError(f"data config {path} must contain a 'datasets' list")
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """Write a config next to a run's checkpoints, so the run can be re-created."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def get_training_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    return model_config.get("training", {})


def precision_to_dtype(precision: Optional[str]):
    """Map a precision string to the autocast dtype (``None`` means full fp32)."""
    import torch

    mapping = {
        "bf16-mixed": torch.bfloat16,
        "bf16": torch.bfloat16,
        "16-mixed": torch.float16,
        "fp16": torch.float16,
        "32-true": None,
        "fp32": None,
        "32": None,
        None: None,
    }
    if precision not in mapping:
        raise ValueError(
            f"unknown precision {precision!r}; expected one of {sorted(k for k in mapping if k)}"
        )
    return mapping[precision]
