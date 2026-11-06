"""Optimizers and learning-rate schedules.
"""

import warnings
from typing import Any, Dict, Iterable, Optional

import torch


class InverseLR(torch.optim.lr_scheduler._LRScheduler):
    """Inverse decay schedule with optional exponential warmup.

    ``inv_gamma`` is the number of steps after which the learning rate has decayed
    to ``(1/2) ** power`` of its original value.

    Args:
        optimizer: wrapped optimizer.
        inv_gamma: inverse multiplicative decay factor.
        power: exponent of the decay.
        warmup: exponential warmup factor in ``[0, 1)``; 0 disables warmup. With
            ``warmup=0.999`` the learning rate reaches ~63% of its base value
            after ~1000 steps.
        final_lr: floor for the learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        inv_gamma: float = 1.0,
        power: float = 1.0,
        warmup: float = 0.0,
        final_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0.0 <= warmup < 1:
            raise ValueError(f"warmup must be in [0, 1), got {warmup}")
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("use `get_last_lr()` to read the current learning rate")
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult) for base_lr in self.base_lrs]


def create_optimizer_from_config(optimizer_config: Dict[str, Any], parameters: Iterable) -> torch.optim.Optimizer:
    """Build an optimizer, e.g. ``{"type": "AdamW", "config": {"lr": 2e-4}}``.

    ``parameters`` may be a flat iterable of tensors or a list of torch
    param-group dicts. A group may carry ``"lr_scale"``, which is resolved here
    to ``lr_scale * base_lr`` — this is how a module (e.g. the frequency warp's
    width logits) requests a smaller step without the config having to repeat
    the base learning rate.
    """
    optimizer_type = optimizer_config["type"]
    if not hasattr(torch.optim, optimizer_type):
        raise ValueError(f"torch.optim has no optimizer {optimizer_type!r}")
    optimizer_fn = getattr(torch.optim, optimizer_type)
    config = optimizer_config.get("config", {})

    parameters = list(parameters)
    if parameters and isinstance(parameters[0], dict):
        base_lr = config.get("lr")
        for group in parameters:
            scale = group.pop("lr_scale", None)
            if scale is None:
                continue
            if base_lr is None:
                raise ValueError(
                    "a parameter group requests lr_scale but the optimizer config has no base 'lr'"
                )
            group["lr"] = base_lr * scale
    return optimizer_fn(parameters, **config)


def create_scheduler_from_config(
    scheduler_config: Optional[Dict[str, Any]], optimizer: torch.optim.Optimizer
):
    """Build a scheduler; ``None`` config means a constant learning rate."""
    if scheduler_config is None:
        return None
    scheduler_type = scheduler_config["type"]
    if scheduler_type == "InverseLR":
        scheduler_fn = InverseLR
    elif hasattr(torch.optim.lr_scheduler, scheduler_type):
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_type)
    else:
        raise ValueError(f"unknown scheduler type {scheduler_type!r}")
    return scheduler_fn(optimizer, **scheduler_config.get("config", {}))


def build_optimizers_and_schedulers(
    optimizer_configs: Dict[str, Any], parameter_groups: Dict[str, Iterable]
):
    """Build one optimizer (and scheduler) per named parameter group.

    Args:
        optimizer_configs: ``{group_name: {"optimizer": ..., "scheduler": ...}}``.
        parameter_groups: ``{group_name: parameters}``. The tokenizer has two
            groups (``autoencoder`` and ``discriminator``), the LM has one (``lm``).

    Returns:
        ``(optimizers, schedulers)``, both dicts keyed by group name.
    """
    optimizers, schedulers = {}, {}
    for name, parameters in parameter_groups.items():
        if name not in optimizer_configs:
            raise ValueError(
                f"training.optimizer_configs is missing the {name!r} group "
                f"(found: {sorted(optimizer_configs)})"
            )
        group_config = optimizer_configs[name]
        optimizers[name] = create_optimizer_from_config(group_config["optimizer"], parameters)
        schedulers[name] = create_scheduler_from_config(group_config.get("scheduler"), optimizers[name])
    return optimizers, schedulers
