"""Conditioning for the BandTok language model.

Two conditions are used, both prepended to the audio-token sequence as a prefix:

* **caption** — a frozen T5 text encoder embeds the text prompt;
* **segment-time** — the crop's start time and the full track duration are
  embedded as continuous numbers. Captions describe whole tracks while training
  crops are short segments, so these let the model tell an opening segment from a
  middle one under the same caption.

Provenance:
  * The conditioner interface and the T5/number conditioners are adapted from
    stable-audio-tools (MIT, LICENSES/LICENSE.stable-audio-tools.txt), which in
    turn credits audiocraft (MIT, LICENSES/LICENSE.audiocraft.txt).
  * ``NumberEmbedder`` and its learned Fourier-feature time embedding come from
    audio-diffusion-pytorch by archinet.ai (MIT), vendored via stable-audio-tools.
"""

import logging
import os
import typing as tp
import warnings
from math import pi

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """Random-Fourier-feature embedding of a continuous scalar."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


class NumberEmbedder(nn.Module):
    """Embed scalars (or tensors of scalars) into ``features`` dimensions."""

    def __init__(self, features: int, dim: int = 256) -> None:
        super().__init__()
        self.features = features
        self.embedding = TimePositionalEmbedding(dim=dim, out_features=features)

    def forward(self, x: tp.Union[tp.List[float], torch.Tensor]) -> torch.Tensor:
        if not torch.is_tensor(x):
            device = next(self.embedding.parameters()).device
            x = torch.tensor(x, device=device)
        shape = x.shape
        embedding = self.embedding(x.reshape(-1))
        return embedding.view(*shape, self.features)


class Conditioner(nn.Module):
    """Base class: maps raw metadata to ``(embeddings, mask)``.

    ``embeddings`` is ``[B, T_cond, output_dim]`` and ``mask`` is ``[B, T_cond]``
    with 1 for valid positions.
    """

    def __init__(self, dim: int, output_dim: int, project_out: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def set_device(self, device: tp.Any) -> None:
        raise NotImplementedError

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError


class NumberConditioner(Conditioner):
    """Embed a list of floats, normalized to ``[min_val, max_val]``.
    """

    def __init__(self, output_dim: int, min_val: float = 0.0, max_val: float = 1.0) -> None:
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim)
        self.device = next(self.embedder.parameters()).device

    def set_device(self, device: tp.Any) -> None:
        self.to(device)
        self.device = device

    def forward(self, floats: tp.List[float]) -> tp.List[torch.Tensor]:
        self.device = next(self.embedder.parameters()).device

        values = torch.tensor([float(x) for x in floats], device=self.device)
        values = values.clamp(self.min_val, self.max_val)
        normalized = (values - self.min_val) / (self.max_val - self.min_val)
        normalized = normalized.to(next(self.embedder.parameters()).dtype)

        embeds = self.embedder(normalized).unsqueeze(1)
        return [embeds, torch.ones(embeds.shape[0], 1, device=self.device)]


class T5Conditioner(Conditioner):
    """Frozen T5 text encoder over the caption.

    Args:
        output_dim: width expected by the LM's prefix projection.
        t5_model_name: Hugging Face model id.
        t5_model_path: optional local directory, for offline training.
        max_length: caption tokens; captions are padded/truncated to this length,
            so the conditioning prefix has a fixed size.
        enable_grad: fine-tune T5 along with the LM.
    """

    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }

    def __init__(
        self,
        output_dim: int,
        t5_model_name: str = "t5-base",
        t5_model_path: tp.Optional[str] = None,
        max_length: int = 125,
        enable_grad: bool = False,
        project_out: bool = False,
    ) -> None:
        if t5_model_name not in self.T5_MODEL_DIMS:
            raise ValueError(
                f"unknown T5 model {t5_model_name!r}; expected one of {sorted(self.T5_MODEL_DIMS)}"
            )
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)

        from transformers import AutoTokenizer, T5EncoderModel

        self.max_length = max_length
        self.enable_grad = enable_grad
        self.device = "cpu"

        source = t5_model_name
        local_files_only = False
        if t5_model_path:
            expanded = os.path.expanduser(t5_model_path)
            if not os.path.exists(expanded):
                raise FileNotFoundError(f"local T5 path does not exist: {expanded}")
            source, local_files_only = expanded, True

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
                # Keep T5 in fp32: its embedding gradients overflow easily in fp16/bf16.
                model = (
                    T5EncoderModel.from_pretrained(source, local_files_only=local_files_only)
                    .float()
                    .train(enable_grad)
                    .requires_grad_(enable_grad)
                )
        finally:
            logging.disable(previous_level)

        if self.enable_grad:
            self.model = model
        else:
            # Hidden from the module tree so the frozen encoder stays out of
            # state_dict() and out of the optimizer.
            self.__dict__["model"] = model

    def set_device(self, device: tp.Any) -> None:
        self.to(device)
        self.model.to(device=device, dtype=torch.float32)
        self.device = device

    def forward(self, texts: tp.List[str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device).to(torch.bool)

        self.model.train(self.enable_grad)
        with torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask


class MultiConditioner(nn.Module):
    """Apply several conditioners to a batch of dataset info dicts.

    Args:
        conditioners: mapping from metadata key to conditioner.
        default_keys: fallback metadata keys, e.g. ``{"caption": "prompt"}``.
    """

    def __init__(
        self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Optional[tp.Dict[str, str]] = None
    ) -> None:
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys or {}

    def set_device(self, device: tp.Any) -> None:
        for module in self.conditioners.values():
            module.set_device(device)

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]]) -> tp.Dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            metadata_key = key
            inputs = []
            for item in batch_metadata:
                if metadata_key not in item:
                    metadata_key = self.default_keys.get(key, metadata_key)
                    if metadata_key not in item:
                        raise ValueError(
                            f"conditioner key {key!r} not found in batch metadata "
                            f"(available: {sorted(item)})"
                        )
                value = item[metadata_key]
                # Collate functions sometimes wrap scalars in a 1-element list.
                if isinstance(value, (list, tuple)) and len(value) == 1:
                    value = value[0]
                inputs.append(value)
            output[key] = conditioner(inputs)
        return output


def create_multi_conditioner_from_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """Build a :class:`MultiConditioner` from the ``conditioning`` config block."""
    conditioners: tp.Dict[str, Conditioner] = {}
    cond_dim = config["cond_dim"]

    for conditioner_config in config["configs"]:
        cond_id = conditioner_config["id"]
        conditioner_type = conditioner_config["type"]
        conditioner_kwargs = dict(conditioner_config.get("config", {}))
        conditioner_kwargs.setdefault("output_dim", cond_dim)

        if conditioner_type == "t5":
            conditioners[cond_id] = T5Conditioner(**conditioner_kwargs)
        elif conditioner_type == "number":
            conditioners[cond_id] = NumberConditioner(**conditioner_kwargs)
        else:
            raise ValueError(
                f"unknown conditioner type {conditioner_type!r} for {cond_id!r}; expected 't5' or 'number'"
            )

    return MultiConditioner(conditioners, default_keys=config.get("default_keys", {}))
