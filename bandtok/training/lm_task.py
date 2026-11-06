"""Language-model training.
"""

from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.lm import BandTokLM
from .distributed import unwrap_model
from .logger import MetricsLogger
from .trainer import StepOutput, TrainingTask


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 10) -> Tuple[float, float]:
    """Return ``(top-1, top-k)`` accuracy over a flat set of predictions."""
    if targets.numel() == 0:
        return 0.0, 0.0
    k = min(k, logits.size(-1))
    top_k = logits.topk(k, dim=-1).indices
    correct = top_k == targets.unsqueeze(-1)
    return float(correct[..., 0].float().mean()), float(correct.any(dim=-1).float().mean())


class LanguageModelTask(TrainingTask):
    """Next-token training for :class:`BandTokLM`.

    Args:
        model: the LM wrapper (its tokenizer is expected to be frozen).
        cfg_dropout_prob: probability of dropping the conditioning prefix.
    """

    def __init__(self, model: BandTokLM, cfg_dropout_prob: float = 0.1) -> None:
        self.bandtok_lm = model
        self.cfg_dropout_prob = cfg_dropout_prob

        model.tokenizer.eval()
        model.tokenizer.requires_grad_(False)

        # Only the LM trains; the tokenizer and the T5 encoder stay frozen.
        self.modules: Dict[str, nn.Module] = {"model": model.lm}
        self.trainable_modules: Dict[str, nn.Module] = {"lm": model.lm}

    def parameter_groups(self) -> Dict[str, Iterable]:
        return {"lm": [p for p in self.bandtok_lm.lm.parameters() if p.requires_grad]}

    def to(self, device: torch.device) -> None:
        self.bandtok_lm.to(device)
        if self.bandtok_lm.conditioner is not None:
            self.bandtok_lm.conditioner.set_device(device)

    def train(self) -> None:
        self.bandtok_lm.lm.train()
        self.bandtok_lm.tokenizer.eval()  # frozen, keep BatchNorm/EMA buffers still

    def eval(self) -> None:
        self.bandtok_lm.eval()

    # -- batch preparation ------------------------------------------------

    @torch.no_grad()
    def _tokenize(self, audio: torch.Tensor, infos: List[Dict[str, Any]]):
        """Audio -> ``(codes [B, 1, L], padding_mask [B, L])``, flattened."""
        tokenizer = self.bandtok_lm.tokenizer
        codes = tokenizer.tokenize(audio)
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        batch_size, num_codebooks, num_positions = codes.shape

        masks = []
        for info in infos:
            mask = info["padding_mask"]
            masks.append(mask if mask.ndim == 1 else mask[0])
        padding_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True).to(audio.device)
        padding_mask = (
            F.interpolate(padding_mask.unsqueeze(1).float(), size=num_positions, mode="nearest")
            .squeeze(1)
            .bool()
        )

        codes = codes.long()
        if num_codebooks > 1:
            # BandTok-1D: give each residual codebook its own vocabulary range and
            # interleave the codebooks within each time step.
            offsets = (
                torch.arange(num_codebooks, device=codes.device).view(1, num_codebooks, 1)
                * self.bandtok_lm.codebook_size
            )
            codes = codes + offsets
            codes = codes.permute(0, 2, 1).reshape(batch_size, 1, num_positions * num_codebooks)
            padding_mask = (
                padding_mask.unsqueeze(-1)
                .expand(-1, -1, num_codebooks)
                .reshape(batch_size, num_positions * num_codebooks)
            )

        return codes, padding_mask

    # -- steps ------------------------------------------------------------

    def train_step(self, batch: Any, step: int) -> StepOutput:
        audio, infos = batch
        codes, padding_mask = self._tokenize(audio, infos)

        condition_tensors = None
        if self.bandtok_lm.conditioner is not None:
            condition_tensors = self.bandtok_lm.conditioning_from_info(infos)

        # Route the forward through the (possibly DDP-wrapped) LM module so that
        # gradients are synchronized, while conditioning/CFG logic stays here.
        lm_module = self.trainable_modules["lm"]
        wrapper = unwrap_model(lm_module)
        conditioning_inputs = self.bandtok_lm.get_conditioning_inputs(condition_tensors or {})
        prepend_cond = conditioning_inputs["prepend_cond"]

        if self.cfg_dropout_prob > 0.0 and prepend_cond is not None:
            null_embed = torch.zeros_like(prepend_cond) + 1e-4
            drop = torch.bernoulli(
                torch.full((prepend_cond.shape[0], 1, 1), self.cfg_dropout_prob, device=prepend_cond.device)
            ).bool()
            prepend_cond = torch.where(drop, null_embed, prepend_cond)

        sequence = codes[:, 0]
        sequence, padding_mask = wrapper.add_eos(sequence, padding_mask)
        output = lm_module(sequence, prepend_cond=prepend_cond, padding_mask=padding_mask)

        loss = output.loss
        metrics: Dict[str, Any] = {
            "loss": loss,
            "perplexity": torch.exp(loss.detach()),
        }

        with torch.no_grad():
            # Align logits with their targets: position i predicts token i+1.
            logits = output.logits[:, -sequence.size(1) - 1 : -1]
            valid = padding_mask.bool()
            top1, top10 = top_k_accuracy(logits[valid].float(), sequence[valid], k=10)
            metrics["top1_acc"] = top1
            metrics["top10_acc"] = top10

        return StepOutput(loss=loss, group="lm", metrics=metrics)

    @torch.no_grad()
    def validation_step(self, batch: Any) -> Dict[str, float]:
        audio, infos = batch
        codes, padding_mask = self._tokenize(audio, infos)
        condition_tensors = (
            self.bandtok_lm.conditioning_from_info(infos) if self.bandtok_lm.conditioner else None
        )
        output = self.bandtok_lm.compute_logits(
            codes, condition_tensors=condition_tensors, cfg_dropout_prob=0.0, padding_mask=padding_mask
        )
        loss = float(output.loss)
        return {"loss": loss, "perplexity": float(torch.exp(torch.tensor(loss)))}

    @torch.no_grad()
    def on_demo(self, batch: Any, step: int, logger: MetricsLogger) -> None:
        """Log audio reconstructed from teacher-forced tokens as a sanity check.

        Full sampling is left to ``scripts/generate.py``: it is slow enough that
        running it inside the training loop would stall the run.
        """
        audio, infos = batch
        audio, infos = audio[:1], infos[:1]
        codes, _ = self._tokenize(audio, infos)
        recon = self.bandtok_lm.tokenizer.decode_tokens(codes)
        logger.log_audio("demo/audio_real", audio[0], step, self.bandtok_lm.sample_rate)
        logger.log_audio("demo/audio_tokenized", recon[0], step, self.bandtok_lm.sample_rate)


def create_lm_task(model_config: Dict[str, Any], model: BandTokLM) -> LanguageModelTask:
    """Build a :class:`LanguageModelTask` from an LM model config."""
    training_config = model_config["training"]
    return LanguageModelTask(
        model=model,
        cfg_dropout_prob=float(training_config.get("cfg_dropout_prob", 0.1)),
    )
