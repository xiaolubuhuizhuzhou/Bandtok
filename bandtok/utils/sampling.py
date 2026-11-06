"""Token sampling utilities.

Adapted from stable-audio-tools (https://github.com/Stability-AI/stable-audio-tools),
MIT License -- see LICENSES/LICENSE.stable-audio-tools.txt -- which in turn follows
audiocraft (https://github.com/facebookresearch/audiocraft), MIT License.
"""

from typing import Optional

import torch


def multinomial(
    input: torch.Tensor, num_samples: int, replacement: bool = False, *, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """``torch.multinomial`` for inputs with arbitrarily many leading dimensions.

    Candidates live on the last dimension. For ``num_samples == 1`` this uses the
    Gumbel/exponential trick, which avoids materializing a 2D view.
    """
    if num_samples == 1:
        q = torch.empty_like(input).exponential_(1, generator=generator)
        return torch.argmax(input / q, dim=-1, keepdim=True).to(torch.int64)

    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    return output_.reshape(*list(input.shape[:-1]), -1)


def sample_top_k(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Sample from the ``k`` most likely candidates on the last dimension."""
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs = probs * (probs >= min_value_top_k).float()
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return multinomial(probs, num_samples=1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest candidate set with cumulative mass > ``p``."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort = probs_sort * (~mask).float()
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    next_token = multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def normalize_probs(probs: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Repair a probability row that became numerically invalid.

    Long autoregressive rollouts in low precision can produce rows that are all
    zeros or contain NaNs; falling back to the argmax keeps generation alive
    instead of raising inside ``multinomial``.
    """
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    bad_rows = probs_sum <= 0
    probs = probs / probs_sum.clamp_min(1e-12)
    if bad_rows.any():
        fallback_idx = torch.argmax(logits, dim=-1, keepdim=True)
        fallback = torch.zeros_like(probs)
        fallback.scatter_(1, fallback_idx, 1.0)
        probs = torch.where(bad_rows, fallback, probs)
    return probs
