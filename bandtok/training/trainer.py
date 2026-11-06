"""The BandTok trainer.
"""

import contextlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from ..config import precision_to_dtype
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .distributed import (
    barrier,
    format_parameter_count,
    get_local_rank,
    get_world_size,
    is_distributed,
    is_main_process,
    print_once,
    unwrap_model,
)
from .logger import MetricsLogger, format_metric
from .optim import build_optimizers_and_schedulers


@dataclass
class StepOutput:
    """What one training step produced.

    Attributes:
        loss: scalar loss to back-propagate.
        group: which optimizer group to step (e.g. ``"autoencoder"``).
        metrics: scalars to log.
    """

    loss: torch.Tensor
    group: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class TrainingTask(ABC):
    """A model plus the definition of one training step.

    Subclasses maintain two views of the same modules:

    * :attr:`modules` -- unwrapped modules to checkpoint, keyed by name, always
      including ``"model"``;
    * :attr:`trainable_modules` -- the modules to *call*, keyed by optimizer group.
      After :meth:`wrap_distributed` these are DDP wrappers, so calling them is
      what triggers gradient synchronization.
    """

    #: unwrapped modules to save; must contain "model"
    modules: Dict[str, nn.Module]
    #: modules to call during a step, keyed by optimizer group name
    trainable_modules: Dict[str, nn.Module]

    @property
    def model(self) -> nn.Module:
        """The module inference will load."""
        return self.modules["model"]

    @abstractmethod
    def parameter_groups(self) -> Dict[str, Iterable]:
        """Parameters per optimizer group."""

    @abstractmethod
    def train_step(self, batch: Any, step: int) -> StepOutput:
        """Compute the loss for one micro-batch."""

    def active_group(self, step: int) -> str:
        """Which optimizer group this step updates. Single-group tasks ignore it."""
        return next(iter(self.parameter_groups()))

    def validation_step(self, batch: Any) -> Dict[str, float]:
        return {}

    def on_demo(self, batch: Any, step: int, logger: MetricsLogger) -> None:
        """Optional qualitative logging (audio, spectrograms)."""

    def to(self, device: torch.device) -> None:
        for module in self.modules.values():
            module.to(device)

    def wrap_distributed(self, device: torch.device) -> None:
        """Wrap each trainable module in DDP."""
        device_ids = [get_local_rank()] if device.type == "cuda" else None
        for name, module in self.trainable_modules.items():
            if not any(p.requires_grad for p in module.parameters()):
                continue
            self.trainable_modules[name] = DistributedDataParallel(
                module, device_ids=device_ids, output_device=device_ids[0] if device_ids else None
            )

    def train(self) -> None:
        for module in self.modules.values():
            module.train()

    def eval(self) -> None:
        for module in self.modules.values():
            module.eval()


class Trainer:
    """Drives a :class:`TrainingTask` over a dataloader.

    Args:
        task: the training task.
        train_loader: training dataloader; its sampler is re-seeded per epoch.
        training_config: the model config's ``training`` block.
        run_dir: directory for checkpoints and logs.
        device: device for this process.
        val_loader: optional validation dataloader.
        max_steps: stop after this many optimizer steps (``-1`` for no limit).
        accum_batches: micro-batches per optimizer step.
        logger: metric logger; one is created if omitted.
        config: full config, embedded into checkpoints.
    """

    def __init__(
        self,
        task: TrainingTask,
        train_loader: torch.utils.data.DataLoader,
        training_config: Dict[str, Any],
        run_dir: str,
        device: torch.device,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        max_steps: int = -1,
        accum_batches: int = 1,
        logger: Optional[MetricsLogger] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task = task
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.training_config = training_config
        self.run_dir = run_dir
        self.device = device
        self.max_steps = max_steps
        self.accum_batches = max(1, accum_batches)

        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        checkpoint_config = training_config.get("checkpoint", {})
        self.checkpoint_every = int(checkpoint_config.get("every_n_train_steps", 10000))
        self.keep_last_k = int(checkpoint_config.get("keep_last_k", -1))
        # Numbered checkpoints are kept only at these milestones; everything else
        # a run needs lives in ``last.pt``, which every save rewrites. Set to 0 to
        # go back to numbering every save and letting ``keep_last_k`` alone bound
        # the directory.
        self.keep_every = int(checkpoint_config.get("keep_every_n_steps", 100000))

        self.val_every = int(training_config.get("val_every", 0) or 0)
        self.val_steps = int(training_config.get("val_steps", 50))
        self.demo_every = int(training_config.get("demo", {}).get("demo_every", 0) or 0)
        self.clip_grad_norm = float(training_config.get("clip_grad_norm", 0.0) or 0.0)

        self.amp_dtype = precision_to_dtype(training_config.get("precision"))
        self.use_scaler = self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.task.to(device)
        self.optimizers, self.schedulers = build_optimizers_and_schedulers(
            training_config["optimizer_configs"], self.task.parameter_groups()
        )
        if is_distributed():
            self.task.wrap_distributed(device)

        self.logger = logger or MetricsLogger(
            log_dir=run_dir, log_every=int(training_config.get("logging", {}).get("log_every", 20))
        )
        self.step = 0

        for name, module in self.task.trainable_modules.items():
            print_once(f"[bandtok] {format_parameter_count(unwrap_model(module), name)}")

    # -- helpers ----------------------------------------------------------

    def _autocast(self):
        if self.amp_dtype is None or self.device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _module_for_group(self, group: str) -> nn.Module:
        return self.task.trainable_modules[group]

    def _optimizer_step(self, group: str) -> Optional[float]:
        """Unscale, clip, step and advance the scheduler for one group."""
        optimizer = self.optimizers[group]
        grad_norm = None

        if self.use_scaler:
            self.scaler.unscale_(optimizer)
        if self.clip_grad_norm > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._module_for_group(group).parameters(), self.clip_grad_norm
                )
            )

        if self.use_scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        # Zero EVERY group's gradients, not just the stepped one. In adversarial
        # training the generator's backward also writes gradients into the
        # discriminator's parameters (the adversarial and feature-matching terms
        # flow through it); leaving those in place would contaminate the next
        # discriminator update with the exact opposite of its own objective.
        for group_optimizer in self.optimizers.values():
            group_optimizer.zero_grad(set_to_none=True)

        scheduler = self.schedulers.get(group)
        if scheduler is not None:
            # Each group's schedule advances only on the steps where it is updated,
            # matching the alternating GAN schedule of the reference implementation.
            scheduler.step()
        return grad_norm

    def resume(self, path: Optional[str]) -> None:
        """Resume from a checkpoint, or from ``last.pt`` in the run directory."""
        if path is None:
            candidate = os.path.join(self.checkpoint_dir, "last.pt")
            path = candidate if os.path.exists(candidate) else None
        if path is None:
            return
        self.step = load_checkpoint(
            path, self.task.modules, optimizers=self.optimizers, schedulers=self.schedulers, strict=False
        )

    def save(self, tag: Optional[str] = None) -> None:
        """Write ``last.pt``, plus a numbered copy at milestone steps.

        ``last.pt`` is rewritten on every save, so a crash still costs at most
        ``every_n_train_steps``. A numbered ``step_N.pt`` is written once per
        ``keep_every_n_steps`` window (or whenever ``tag`` is given explicitly, as
        the final save does), which is what keeps a long run from filling the disk
        with 2 GB checkpoints.
        """
        if not is_main_process():
            return
        names = ["last"]
        if tag is not None:
            names.insert(0, tag)
        elif self.keep_every <= 0 or self._is_milestone(self.step):
            # keep_every_n_steps <= 0 means "number every save", the old behaviour.
            names.insert(0, f"step_{self.step}")
        for name in names:
            save_checkpoint(
                os.path.join(self.checkpoint_dir, f"{name}.pt"),
                self.task.modules,
                step=self.step,
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                config=self.config,
            )
        self._prune_checkpoints()

    def _is_milestone(self, step: int) -> bool:
        """Whether ``step``'s numbered checkpoint is protected from pruning.

        A milestone is the *first* save at or after each ``keep_every_n_steps``
        boundary, not a step that divides it exactly: ``every_n_train_steps`` does
        not have to divide ``keep_every_n_steps`` (several configs save every
        12000 steps), and demanding exact division would silently keep nothing.
        """
        if self.keep_every <= 0:
            return False
        return step % self.keep_every < max(1, self.checkpoint_every)

    def _prune_checkpoints(self) -> None:
        """Trim numbered checkpoints that no milestone and no ``keep_last_k`` protects.

        Milestones are never deleted. With milestones enabled the only prunable
        files are leftovers from before -- or from a run that set
        ``keep_every_n_steps: 0`` -- so ``keep_last_k`` still bounds those.
        """
        if self.keep_last_k <= 0:
            return

        def step_of(name: str) -> int:
            return int(name.split("_")[1].split(".")[0])

        prunable: List[str] = sorted(
            (
                name
                for name in os.listdir(self.checkpoint_dir)
                if name.startswith("step_") and not self._is_milestone(step_of(name))
            ),
            key=step_of,
        )
        for name in prunable[: -self.keep_last_k]:
            os.remove(os.path.join(self.checkpoint_dir, name))

    # -- loops ------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run the task's validation step over up to ``val_steps`` batches."""
        if self.val_loader is None:
            return {}

        self.task.eval()
        totals: Dict[str, float] = {}
        count = 0
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= self.val_steps:
                break
            with self._autocast():
                metrics = self.task.validation_step(self._to_device(batch))
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1

        self.task.train()
        if not count:
            return {}

        means = {f"val/{key}": value / count for key, value in totals.items()}
        if is_main_process():
            body = "  ".join(f"{key}={format_metric(value)}" for key, value in sorted(means.items()))
            print(f"[validation step {self.step}] {body}", flush=True)
            if self.logger.wandb is not None:
                self.logger.wandb.log(means, step=self.step)
            if self.logger.tensorboard is not None:
                for key, value in means.items():
                    self.logger.tensorboard.add_scalar(key, value, self.step)
        return means

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(item) for item in batch)
        return batch

    def fit(self) -> None:
        """Run the training loop until ``max_steps`` (or forever)."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # An empty loader would otherwise spin the epoch loop forever without ever
        # taking a step. This happens easily under DDP: with `drop_last=True`, a
        # dataset smaller than world_size * batch_size leaves every rank with nothing.
        try:
            batches_per_epoch = len(self.train_loader)
        except TypeError:  # iterable-style dataset with no length
            batches_per_epoch = None
        if batches_per_epoch == 0:
            raise ValueError(
                "the training dataloader is empty: the dataset is too small for "
                f"batch_size x world_size ({self.train_loader.batch_size} x {get_world_size()}) "
                "with drop_last enabled. Use a smaller batch size, fewer processes, or more data."
            )

        self.task.train()
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

        print_once(
            f"[bandtok] training from step {self.step} "
            f"(accum={self.accum_batches}, precision={self.training_config.get('precision', 'fp32')})"
        )
        start_time = time.time()
        epoch = 0
        micro_step = 0
        demo_batch = None
        group = self.task.active_group(self.step)

        while self.max_steps < 0 or self.step < self.max_steps:
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for batch in self.train_loader:
                batch = self._to_device(batch)
                if demo_batch is None:
                    demo_batch = batch

                with self._autocast():
                    output = self.task.train_step(batch, self.step)
                group = output.group

                loss = output.loss / self.accum_batches
                if self.use_scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                self.logger.update(output.metrics)
                micro_step += 1

                if micro_step % self.accum_batches != 0:
                    continue

                grad_norm = self._optimizer_step(group)
                self.step += 1

                if grad_norm is not None:
                    self.logger.update({"grad_norm": grad_norm})
                if self.logger.should_flush(self.step):
                    lrs = {
                        f"lr/{name}": opt.param_groups[0]["lr"] for name, opt in self.optimizers.items()
                    }
                    self.logger.flush(self.step, prefix="train/", extra=lrs)

                if self.demo_every and self.step % self.demo_every == 0:
                    self.task.eval()
                    try:
                        self.task.on_demo(demo_batch, self.step, self.logger)
                    except Exception as exc:  # a broken demo must not kill a long run
                        print_once(f"[bandtok] demo failed at step {self.step}: {type(exc).__name__}: {exc}")
                    self.task.train()

                if self.val_every and self.step % self.val_every == 0:
                    self.validate()

                if self.checkpoint_every and self.step % self.checkpoint_every == 0:
                    self.save()
                    barrier()

                if self.max_steps >= 0 and self.step >= self.max_steps:
                    break

            epoch += 1

        self.save(tag=f"step_{self.step}")
        elapsed = (time.time() - start_time) / 3600
        print_once(f"[bandtok] finished at step {self.step} after {elapsed:.2f} h")
        self.logger.close()
