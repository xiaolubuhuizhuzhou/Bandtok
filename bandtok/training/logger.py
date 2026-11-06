"""Metric logging.
"""

import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional

import torch

from .distributed import is_main_process


def format_metric(value: float) -> str:
    """Format a metric for the console.

    Learning rates during warmup are far below 1e-4, so a fixed 4-decimal format
    would print them all as 0.0000; small magnitudes switch to scientific notation.
    """
    if value != 0.0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


class MetricsLogger:
    """Accumulate scalars and flush their means to the configured backends."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_every: int = 20,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        project: str = "bandtok",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log_every = max(1, log_every)
        self.log_dir = log_dir
        # Raw per-step values (floats, or detached scalar tensors synced at flush).
        self._values: Dict[str, list] = defaultdict(list)
        self._last_flush_step = 0
        self._last_flush_time = time.time()

        self.wandb = None
        self.tensorboard = None
        if not is_main_process():
            return

        if use_wandb:
            try:
                import wandb

                # A short init timeout so an unreachable W&B server (e.g. an
                # offline training node accidentally left in "online" mode) fails
                # fast and disables logging, instead of stalling the whole run.
                # WANDB_MODE / WANDB_PROJECT / WANDB_DIR from the environment are
                # honoured by wandb.init as usual.
                settings = wandb.Settings(init_timeout=30)
                self.wandb = wandb
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", project),
                    name=run_name,
                    config=config,
                    dir=log_dir,
                    settings=settings,
                )
                print(f"[bandtok] W&B logging enabled (mode={wandb.run.settings.mode}, run={wandb.run.name})")
            except Exception as exc:  # keep training alive if logging is misconfigured
                print(f"[bandtok] W&B logging disabled ({type(exc).__name__}: {exc})")
                self.wandb = None

        if use_tensorboard and log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tensorboard = SummaryWriter(log_dir=log_dir)
            except Exception as exc:
                print(f"[bandtok] TensorBoard logging disabled ({type(exc).__name__}: {exc})")
                self.tensorboard = None

    def update(self, metrics: Dict[str, Any]) -> None:
        """Accumulate a step's scalars.

        GPU tensors are kept as (detached) tensors here and only converted to
        Python floats at flush time: an ``.item()`` per metric per step would
        synchronize with the GPU a dozen times per iteration and stop the CPU
        from queueing the next step's kernels ahead of time.
        """
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach()
            elif not isinstance(value, (int, float)):
                continue
            self._values[key].append(value)

    def should_flush(self, step: int) -> bool:
        return step % self.log_every == 0 and bool(self._values)

    def flush(self, step: int, prefix: str = "", extra: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Average, log and reset the accumulated metrics."""
        means: Dict[str, float] = {}
        for key, values in self._values.items():
            tensors = [v.float() for v in values if isinstance(v, torch.Tensor)]
            total = sum(float(v) for v in values if not isinstance(v, torch.Tensor))
            if tensors:
                # One sync per key per flush window instead of one per step.
                total += float(torch.stack(tensors).sum())
            means[key] = total / len(values)
        if extra:
            means.update({k: float(v) for k, v in extra.items() if isinstance(v, (int, float))})
        self._values.clear()

        if not means:
            return means

        now = time.time()
        steps_done = max(1, step - self._last_flush_step)
        steps_per_sec = steps_done / max(1e-6, now - self._last_flush_time)
        self._last_flush_step, self._last_flush_time = step, now

        if is_main_process():
            body = "  ".join(f"{key}={format_metric(value)}" for key, value in sorted(means.items()))
            print(f"[{prefix}step {step}] {body}  ({steps_per_sec:.2f} it/s)", flush=True)

            payload = {f"{prefix}{key}": value for key, value in means.items()}
            payload[f"{prefix}steps_per_sec"] = steps_per_sec
            if self.wandb is not None:
                self.wandb.log(payload, step=step)
            if self.tensorboard is not None:
                for key, value in payload.items():
                    self.tensorboard.add_scalar(key, value, step)

        return means

    def log_audio(self, key: str, audio: torch.Tensor, step: int, sample_rate: int) -> None:
        """Log an audio sample ``[C, N]`` or ``[N]`` (rank 0, W&B/TensorBoard only)."""
        if not is_main_process():
            return
        audio = audio.detach().float().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if self.wandb is not None:
            self.wandb.log(
                {key: self.wandb.Audio(audio.transpose(0, 1).numpy(), sample_rate=sample_rate)}, step=step
            )
        if self.tensorboard is not None:
            self.tensorboard.add_audio(key, audio, step, sample_rate=sample_rate)

    def log_image(self, key: str, image: torch.Tensor, step: int) -> None:
        """Log a ``[H, W]`` or ``[C, H, W]`` image, e.g. a spectrogram."""
        if not is_main_process():
            return
        image = image.detach().float().cpu()
        if self.tensorboard is not None:
            self.tensorboard.add_image(key, image if image.ndim == 3 else image.unsqueeze(0), step)
        if self.wandb is not None:
            array = image.squeeze().numpy()
            self.wandb.log({key: self.wandb.Image(array)}, step=step)

    def close(self) -> None:
        if self.tensorboard is not None:
            self.tensorboard.close()
        if self.wandb is not None:
            self.wandb.finish()
