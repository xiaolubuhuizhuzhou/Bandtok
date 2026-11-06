# BandTok quantizer: a single shared codebook updated with EMA statistics.
#
# The Bottleneck / DiscreteBottleneck base classes are adapted from
# stable-audio-tools (https://github.com/Stability-AI/stable-audio-tools), MIT License
# -- see LICENSES/LICENSE.stable-audio-tools.txt
#
# Codebook statistics are gathered across ranks so that every replica keeps an
# identical codebook. The collectives are no-ops when training on one process.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import random
from dataclasses import dataclass


def _distributed() -> bool:
    """True when a process group is up and there is more than one rank."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum a tensor across ranks in place; a no-op in single-process training."""
    if _distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def vector_quantize(inputs, codebook):
    embedding_size = codebook.size(1)
    inputs_flatten = inputs.reshape(-1, embedding_size)
    codebook_sqr = torch.sum(codebook ** 2, dim=1)
    inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
    # Compute the distances to the codebook
    distances = torch.addmm(codebook_sqr + inputs_sqr,
                            inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

    _, indices_flatten = torch.min(distances, dim=1)
    codes_flatten = torch.index_select(codebook, dim=0,
                                       index=indices_flatten)
    codes = codes_flatten.view_as(inputs)
    return codes, indices_flatten, distances

def mse_loss_with_mask(input, target, mask):
    loss = torch.nn.functional.mse_loss(input, target, reduction='none')
    loss = loss.mean(dim=-1)
    loss = loss * mask
    return loss.sum() / mask.sum()


@dataclass
class VectorQuantizeConfig:
    d_model: int = 128
    quantize_vocab_size: int = 32768
    quantize_ema_decay: float = 0.99
    quantize_restart_interval: int = 100
    quantize_commit_coefficient: float = 0.25
    quantize_loss_scale: int = 10
    denorm: bool = False
    pre_norm_type: str = 'bn'



class VectorQuantize(nn.Module):
    """A single codebook whose entries are maintained by EMA statistics.

    Nearest-neighbour lookup follows the straightforward implementation in
    https://github.com/karpathy/deep-vector-quantization. Two details matter for
    BandTok:

    * latents are normalized before the lookup (BatchNorm by default), which keeps
      the scale of the encoder output and the codebook comparable;
    * codebook entries are updated with exponential moving averages of the assigned
      latents rather than by a codebook loss, which stabilizes a large codebook and
      avoids noisy updates for rarely used codes. Only a commitment loss is returned.
      Setting ``quantize_ema_decay=0`` switches to the classic VQ-VAE codebook loss.

    Codes that stop being used are periodically re-seeded from live encoder outputs
    (``quantize_restart_interval``), which keeps codebook usage high.
    """

    def __init__(self,config=VectorQuantizeConfig(), **kwargs):
        super().__init__()
        self.config: VectorQuantizeConfig = config
        if self.config.pre_norm_type == 'bn':
            self.pre_quantize_norm = nn.BatchNorm1d(self.config.d_model)
        elif not self.config.pre_norm_type:
            self.pre_quantize_norm = nn.Identity()
        else:
            self.pre_quantize_norm = nn.LayerNorm(self.config.d_model, elementwise_affine=False, bias=False)
        self.quantize_ema_count = 0

        self.codebook = nn.Embedding(self.config.quantize_vocab_size, self.config.d_model)
        self.codebook.weight.data.uniform_(-0.2, 0.2)
        # With EMA updates the codebook is maintained by statistics, not by the
        # optimizer. Without EMA (the paper's "w/ codebook loss" ablation) the
        # codebook is trained by the codebook loss and therefore needs gradients.
        self.codebook.weight.requires_grad = not self.config.quantize_ema_decay
        self.register_buffer("ema_count", torch.ones(self.config.quantize_vocab_size, dtype=torch.float))
        self.register_buffer("ema_weight", self.codebook.weight.data.clone().float())

    @torch.amp.autocast("cuda", enabled=False, dtype=torch.float32)
    def forward(self, hidden_states, attention_mask=None):

        channel_first = (hidden_states.ndim == 3
                     and hidden_states.shape[1] == self.config.d_model
                     and hidden_states.shape[-1] != self.config.d_model)
        if channel_first:
            hidden_states = hidden_states.transpose(1, 2).contiguous()  # [B,T,D]
        B, T, D = hidden_states.shape
        assert D == self.config.d_model, (hidden_states.shape, self.config.d_model)

        batch_size = hidden_states.size()[0]
        if attention_mask is None:
            attention_mask = torch.ones([hidden_states.size(0), hidden_states.size(1)]).long().to(hidden_states.device)
        if self.config.denorm:
            mean = hidden_states.mean(dim=-1, keepdim=True)
            var = hidden_states.var(dim=-1, unbiased=False, keepdim=True)
        if self.config.pre_norm_type == 'bn':
            hidden_states = self.pre_quantize_norm(hidden_states.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()
        else:
            hidden_states = self.pre_quantize_norm(hidden_states)

        with torch.amp.autocast("cuda", enabled=False, dtype=torch.float32):
            hidden_states = hidden_states.float()
            hidden_quantized, indices_flat, distances = vector_quantize(hidden_states, self.codebook.weight)
        quantized_token_ids = indices_flat.reshape(batch_size, hidden_quantized.shape[1])
        if self.training:
            encodings = torch.nn.functional.one_hot(indices_flat, self.config.quantize_vocab_size).float()
            # Padded positions are excluded from the codebook statistics (below);
            # they still take part in the straight-through gradient.
            encodings = encodings * attention_mask.reshape(-1, 1)
            n = torch.sum(encodings, dim=0)
            _all_reduce_sum(n)
            self.num_active_codes = n.nonzero().shape[0]
            if self.config.quantize_ema_decay:
                hidden_flat = hidden_states.detach().float().reshape(-1, hidden_states.shape[-1])
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    dw = torch.matmul(encodings.t(), hidden_flat)
                _all_reduce_sum(dw)
                self.ema_count = self.ema_count * self.config.quantize_ema_decay + (
                        1 - self.config.quantize_ema_decay) * n
                total_count = torch.sum(self.ema_count)
                self.ema_count = (self.ema_count + 1e-5) / (
                        total_count + self.config.quantize_vocab_size * 1e-5) * total_count
                self.ema_weight = self.ema_weight * self.config.quantize_ema_decay + (
                        1 - self.config.quantize_ema_decay) * dw
                self.codebook.weight.data = self.ema_weight / self.ema_count.unsqueeze(1)
                self.quantize_loss = self.config.quantize_loss_scale * self.config.quantize_commit_coefficient * mse_loss_with_mask(
                    hidden_states, hidden_quantized.detach(), attention_mask)
                self.quantize_ema_count += 1
                if self.config.quantize_restart_interval is not None and self.quantize_ema_count % self.config.quantize_restart_interval == 0:
                    rank = dist.get_rank() if _distributed() else 0
                    world_size = dist.get_world_size() if _distributed() else 1
                    segment_vocab_size = self.config.quantize_vocab_size // world_size
                    start_idx = segment_vocab_size * rank
                    ema_count_segment = self.ema_count[start_idx: start_idx + segment_vocab_size]
                    threshold = 1 * (
                                self.config.quantize_ema_decay ** self.config.quantize_restart_interval)
                    update_indices = (ema_count_segment < threshold).nonzero()[:, 0] + start_idx
                    num_update = update_indices.shape[0]
                    mask_flat = attention_mask.reshape(-1) > 0
                    hidden_selected = hidden_flat[mask_flat]
                    # Never revive more codes than there are encoder vectors to seed them with.
                    if num_update > len(hidden_selected):
                        update_indices = update_indices[:len(hidden_selected)]
                        num_update = update_indices.shape[0]
                    hidden_update = hidden_selected[random.sample(range(len(hidden_selected)), num_update)]
                    if _distributed():
                        # Each rank revives its own slice of the codebook; share the
                        # replacements so every replica applies the same update.
                        num_update = torch.as_tensor([num_update], dtype=torch.long,
                                                        device=hidden_states.device)
                        num_update_list = [torch.as_tensor([0], dtype=torch.long, device=hidden_states.device)
                                            for _
                                            in range(world_size)]
                        dist.all_gather(num_update_list, num_update)
                        update_indices_list = [
                            torch.zeros(num.item(), dtype=torch.long, device=hidden_states.device) for num in
                            num_update_list]
                        dist.all_gather(update_indices_list, update_indices)
                        update_indices = torch.cat(update_indices_list)
                        hidden_update_list = [
                            torch.zeros(num.item(), hidden_flat.shape[-1], dtype=hidden_update.dtype,
                                        device=hidden_states.device) for num in num_update_list]
                        dist.all_gather(hidden_update_list, hidden_update)
                        hidden_update = torch.cat(hidden_update_list)
                    self.codebook.weight.data[update_indices] = hidden_update
                    self.ema_count[update_indices] = 1
                    self.ema_weight[update_indices] = hidden_update
            else:
                loss = self.config.quantize_loss_scale * (
                        self.config.quantize_commit_coefficient * mse_loss_with_mask(hidden_states,
                                                                                        hidden_quantized.detach(),
                                                                                        attention_mask) + mse_loss_with_mask(
                    hidden_quantized, hidden_states.detach(), attention_mask))
                self.quantize_loss = loss

            hidden_states = hidden_states + (hidden_quantized - hidden_states).detach()
            loss = self.quantize_loss
        else:
            hidden_states = hidden_quantized
            loss = None
        if self.config.denorm:
            eps = self.pre_quantize_norm.eps
            std_eps = (var + eps) ** 0.5
            hidden_states = hidden_states * std_eps + mean

        if channel_first:
            hidden_states = hidden_states.transpose(1,2).contiguous()
        return hidden_states, quantized_token_ids, loss

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)


class BandVectorQuantize(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312

    BandTok uses a single codebook (``n_codebooks=1``) shared by every Mel band.
    Setting ``n_codebooks > 1`` gives the residual (BandTok-1D) variant.
    ``quantize_ema_decay=0`` disables the EMA update and trains the codebook with
    an explicit codebook loss instead, which is the paper's "w/ codebook loss"
    ablation.
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 8,
        codebook_size: int = 32768,
        pre_norm_type: str = 'bn',
        quantizer_dropout: float = 0.0,
        quantize_ema_decay: float = 0.99,
        quantize_commit_coefficient: float = 0.25,
        quantize_restart_interval: int = 100,
    ):
        super().__init__()

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(VectorQuantizeConfig(
                    d_model=input_dim,
                    quantize_vocab_size=codebook_size,
                    pre_norm_type=pre_norm_type,
                    quantize_ema_decay=quantize_ema_decay,
                    quantize_commit_coefficient=quantize_commit_coefficient,
                    quantize_restart_interval=quantize_restart_interval,
                ))
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        tuple
            ``(z_q, codes, commitment_loss)``:

            z_q : Tensor[B x D x T]
                Quantized continuous representation of the input
            codes : Tensor[B x N x T]
                Codebook indices, one row per codebook
            commitment_loss : Tensor[1]
                Commitment loss, summed over the codebooks in use. There is no
                separate codebook loss: the codebook is updated from EMA
                statistics unless ``quantize_ema_decay`` is 0, in which case the
                codebook loss is folded into each quantizer's own loss.
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_indices = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and i >= n_quantizers:
                break

            z_q_i, indices_i, commitment_loss_i = quantizer(
                residual
            )
            if commitment_loss_i is None:
                commitment_loss_i = 0.
            # Create mask to apply quantizer dropout
            mask = (
                torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            )
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            # Sum losses
            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_indices.append(indices_i)

        codes = torch.stack(codebook_indices, dim=1)

        return z_q, codes, commitment_loss

    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            # z_q_i = self.quantizers[i].out_proj(z_p_i)
            # z_q = z_q + z_q_i
            z_q = z_q + z_p_i
        return z_q, torch.cat(z_p, dim=1), codes


# copied from https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/bottleneck.py under MIT license
class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError

class DiscreteBottleneck(Bottleneck):
    def __init__(self, num_quantizers, codebook_size, tokens_id):
        super().__init__(is_discrete=True)

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.tokens_id = tokens_id

    def decode_tokens(self, codes, **kwargs):
        raise NotImplementedError

class BandVQBottleneck(DiscreteBottleneck):
    def __init__(self, quantize_on_decode=False, noise_augment_dim=0, **quantizer_kwargs):
        super().__init__(num_quantizers = quantizer_kwargs["n_codebooks"], codebook_size = quantizer_kwargs["codebook_size"], tokens_id = "codes")

        self.quantizer = BandVectorQuantize(**quantizer_kwargs)
        self.num_quantizers = quantizer_kwargs["n_codebooks"]
        self.quantize_on_decode = quantize_on_decode
        self.noise_augment_dim = noise_augment_dim

    def encode(self, x, return_info=False):
        info = {}

        info["pre_quantizer"] = x

        if self.quantize_on_decode:
            return (x, info) if return_info else x

        z, codes, commitment_loss = self.quantizer(x)

        output = {
            "z": z,
            "codes": codes,
            "vq/commitment_loss": commitment_loss
        }

        output["vq/commitment_loss"] /= self.num_quantizers

        info.update(output)

        if return_info:
            return output["z"], info

        return output["z"]

    def decode(self, x):

        if self.quantize_on_decode:
            x = self.quantizer(x)[0]

        if self.noise_augment_dim > 0:
            noise = torch.randn(x.shape[0], self.noise_augment_dim,
                                x.shape[-1]).type_as(x)
            x = torch.cat([x, noise], dim=1)

        return x

    def decode_tokens(self, codes, **kwargs):
        latents, _, _ = self.quantizer.from_codes(codes)

        return self.decode(latents, **kwargs)
