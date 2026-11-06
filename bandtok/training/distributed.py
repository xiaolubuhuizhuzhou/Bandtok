"""Distributed helpers.
"""

import datetime
import os
from typing import Optional

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_rank() == 0


def print_once(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed(timeout_minutes: int = 30) -> torch.device:
    """Initialize the process group if launched under ``torchrun``.

    Returns the device this process should use.
    """
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not is_distributed():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Surface NCCL failures instead of hanging forever on a dead peer.
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=datetime.timedelta(minutes=timeout_minutes),
        )
        print_once(f"[bandtok] initialized {backend} process group with {get_world_size()} ranks")

    return device


def cleanup_distributed() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a tensor across ranks (returns it unchanged in a single process)."""
    if not is_distributed():
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / get_world_size()


def set_seed(seed: int, rank_offset: bool = True) -> None:
    """Seed the RNGs, offsetting per rank so ranks draw different crops."""
    import random

    import numpy as np

    if rank_offset:
        seed += get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module, whether or not it is DDP-wrapped."""
    return model.module if hasattr(model, "module") else model


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def format_parameter_count(model: torch.nn.Module, name: Optional[str] = None) -> str:
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    prefix = f"{name}: " if name else ""
    return f"{prefix}{total / 1e6:.1f}M parameters ({trainable / 1e6:.1f}M trainable)"
