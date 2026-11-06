"""Checkpoint saving and loading.

A training checkpoint holds everything needed to resume: model, optimizer and
scheduler state, the step counter, and the config the run was launched with.
``export_model`` writes a smaller inference-only file.

Checkpoints whose ``state_dict`` keys carry a wrapper prefix (DDP's ``module.``,
or a wrapper module name) are also accepted.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def _unwrap_state_dict(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Pull the tensor dict out of the various checkpoint layouts we accept."""
    for key in ("model", "state_dict"):
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break
    return state_dict


def _strip_prefixes(state_dict: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """Drop a leading wrapper prefix (``module.``, ``autoencoder.``, ...) if present.

    Wrapper modules and DDP both prefix parameter names; the prefix is removed
    only when doing so increases the number of keys the model recognizes.
    """
    model_keys = set(model.state_dict().keys())
    if any(key in model_keys for key in state_dict):
        return state_dict

    candidates = sorted({key.split(".", 1)[0] for key in state_dict if "." in key})
    best: Tuple[int, Optional[str]] = (0, None)
    for prefix in candidates:
        stripped = {
            key[len(prefix) + 1 :]: value for key, value in state_dict.items() if key.startswith(prefix + ".")
        }
        matches = len(model_keys & set(stripped))
        if matches > best[0]:
            best = (matches, prefix)

    if best[1] is not None:
        prefix = best[1]
        print(f"[bandtok] stripping checkpoint key prefix {prefix!r}")
        return {
            key[len(prefix) + 1 :]: value for key, value in state_dict.items() if key.startswith(prefix + ".")
        }
    return state_dict


def load_state_dict_file(path: str) -> Dict[str, Any]:
    """Load a ``.ckpt``/``.pt``/``.safetensors`` file to CPU."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_weights(
    model: nn.Module, path: str, strict: bool = False, name: str = "model"
) -> Tuple[List[str], List[str]]:
    """Load weights into ``model``, reporting what did and did not match.

    Returns the ``(missing, unexpected)`` key lists. Shape-mismatched entries are
    dropped with a warning rather than raising, which is what lets a checkpoint
    from a differently-configured run partially initialize a model.
    """
    state_dict = _strip_prefixes(_unwrap_state_dict(load_state_dict_file(path)), model)

    model_state = model.state_dict()
    dropped = [
        key for key, value in state_dict.items() if key in model_state and value.shape != model_state[key].shape
    ]
    for key in dropped:
        print(f"[bandtok] {name}: dropping {key} (shape {tuple(state_dict[key].shape)} != {tuple(model_state[key].shape)})")
        state_dict.pop(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    print(
        f"[bandtok] loaded {name} weights from {path}: "
        f"{len(state_dict)} tensors, {len(missing)} missing, {len(unexpected)} unexpected"
    )
    if missing:
        print(f"[bandtok] {name}: first missing keys: {missing[:5]}")
    if unexpected:
        print(f"[bandtok] {name}: first unexpected keys: {unexpected[:5]}")
    return missing, unexpected


def save_checkpoint(
    path: str,
    modules: Dict[str, nn.Module],
    step: int,
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    schedulers: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a resumable checkpoint atomically.

    Args:
        modules: modules to save, keyed by name. ``"model"`` is the model that
            inference will load; a tokenizer run also saves ``"discriminator"``,
            which is needed to resume adversarial training but not to run the model.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "modules": {name: module.state_dict() for name, module in modules.items()},
        # Kept at the top level too, so inference code can load a training
        # checkpoint directly without knowing about the "modules" layout.
        "model": modules["model"].state_dict(),
        "step": step,
        "config": config,
    }
    if optimizers:
        payload["optimizers"] = {name: opt.state_dict() for name, opt in optimizers.items()}
    if schedulers:
        payload["schedulers"] = {
            name: sched.state_dict() for name, sched in schedulers.items() if sched is not None
        }
    if extra:
        payload.update(extra)

    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    modules: Dict[str, nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    schedulers: Optional[Dict[str, Any]] = None,
    strict: bool = True,
) -> int:
    """Restore a training checkpoint in place and return the step to resume from."""
    payload = load_state_dict_file(path)

    saved_modules = payload.get("modules")
    if saved_modules is None:
        # Inference-only checkpoint: only the main model is present.
        saved_modules = {"model": _unwrap_state_dict(payload)}

    for name, module in modules.items():
        if name not in saved_modules:
            print(f"[bandtok] checkpoint has no state for {name!r}; leaving it at its current values")
            continue
        state_dict = _strip_prefixes(saved_modules[name], module)
        module.load_state_dict(state_dict, strict=strict)

    if optimizers and "optimizers" in payload:
        for name, opt in optimizers.items():
            if name in payload["optimizers"]:
                opt.load_state_dict(payload["optimizers"][name])
            else:
                print(f"[bandtok] no saved state for optimizer {name!r}; starting it fresh")
    if schedulers and "schedulers" in payload:
        for name, sched in schedulers.items():
            if sched is not None and name in payload["schedulers"]:
                sched.load_state_dict(payload["schedulers"][name])

    step = int(payload.get("step", 0))
    print(f"[bandtok] resumed from {path} at step {step}")
    return step


def export_model(
    model: nn.Module, path: str, config: Optional[Dict[str, Any]] = None, use_safetensors: bool = False
) -> None:
    """Write an inference-only checkpoint (weights plus the config)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if use_safetensors:
        from safetensors.torch import save_file

        save_file(state_dict, path)
    else:
        torch.save({"model": state_dict, "config": config}, path)
    print(f"[bandtok] exported model to {path}")
