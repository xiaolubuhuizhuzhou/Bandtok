"""BandTok autoregressive language model.

The LM predicts a flattened time-frequency token grid. Tokens are laid out
band-first: the ``F' = 16`` Mel-band tokens of one frame come first, then time
advances. Standard 1D RoPE would place two tokens of the same Mel band in
adjacent frames ``F'`` positions apart, which destroys the local time-frequency
structure that makes the grid worth modelling in the first place. BandTok
therefore uses **2D RoPE**: Interleaved-MRoPE from Qwen3-VL, with the attention
head dimension split into token / time / Mel-band components, so temporal and
spectral locality survive flattening (see :func:`make_2d_positions`).

The sequence fed to the backbone is

    [ caption embeddings | segment-time embeddings | <BOS> | audio tokens | <EOS> ]

where the conditioning prefix is projected into the model's embedding space and
prepended (MusicGen-style prefix conditioning). During training the prefix is
replaced by a near-null embedding with some probability, which is what makes
classifier-free guidance possible at inference time.

Provenance:
  * The backbone is a Hugging Face ``LlamaForCausalLM``; :func:`forward_1d` and
    :func:`forward_2d` are adapted from ``LlamaModel.forward`` and 2D RoPE reuses
    ``Qwen3VLTextRotaryEmbedding``. Transformers is Apache-2.0 licensed --
    see LICENSES/LICENSE.apache-2.0.txt
  * The conditioning-prefix and classifier-free-guidance structure follows
    stable-audio-tools (MIT, LICENSES/LICENSE.stable-audio-tools.txt) and
    audiocraft/MusicGen (MIT, LICENSES/LICENSE.audiocraft.txt).
"""

import contextlib
import math
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import trange
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
# These two symbols moved across transformers versions: ``DynamicCache`` lives in
# ``cache_utils`` and ``create_causal_mask`` in ``masking_utils``, but both were
# (and in some versions still are) re-exported from ``modeling_llama``. Import
# from the canonical modules first and fall back to the re-export, so the LM
# builds across the transformers versions in use on different nodes.
try:
    from transformers.cache_utils import DynamicCache
except ImportError:  # very old transformers
    from transformers.models.llama.modeling_llama import DynamicCache
try:
    from transformers.masking_utils import create_causal_mask
except ImportError:  # transformers predating masking_utils
    from transformers.models.llama.modeling_llama import create_causal_mask

from ..utils.sampling import multinomial, normalize_probs, sample_top_k, sample_top_p
from .conditioners import MultiConditioner, create_multi_conditioner_from_config
from .tokenizer import BandTok


def make_2d_positions(text_len: int, num_frames: int, num_bands: int) -> torch.Tensor:
    """Build the three position axes used by 2D RoPE.

    Args:
        text_len: length of the conditioning prefix, including ``<BOS>``.
        num_frames: number of audio frames (time steps) in the grid.
        num_bands: number of Mel-band tokens per frame.

    Returns:
        ``[3, text_len + num_frames * num_bands + 1]`` long tensor. Row 0 is the
        token axis (global sequence position, shared by text and audio); row 1 is
        the time axis (sequential over text, then the frame index repeated across
        that frame's bands); row 2 is the Mel-band axis (zero for text, ``1..F'``
        within each frame). The trailing position is for the ``<EOS>`` token.
    """
    num_audio_tokens = num_frames * num_bands
    time_pos = torch.arange(num_frames).repeat_interleave(num_bands) + text_len
    band_pos = torch.arange(num_bands).repeat(num_frames) + 1

    pos_ids = torch.arange(text_len + num_audio_tokens + 1).unsqueeze(0).repeat(3, 1)
    pos_ids[2, :] = 0
    pos_ids[1, text_len : text_len + num_audio_tokens] = time_pos
    pos_ids[1, -1] = num_frames + text_len
    pos_ids[2, text_len : text_len + num_audio_tokens] = band_pos
    return pos_ids


@contextlib.contextmanager
def grid_geometry(model: nn.Module, text_token_len: int, num_bands: int):
    """Tell :func:`forward_2d` where the audio grid starts and how wide it is.

    ``LlamaModel.forward`` is reached both directly and through
    ``GenerationMixin.generate``, and ``generate`` rejects keyword arguments that
    the model's signature does not name. Stashing the geometry on the module for
    the duration of a call is therefore the one mechanism that works on both paths.
    """
    previous = getattr(model, "_bandtok_grid", None)
    model._bandtok_grid = (text_token_len, num_bands)
    try:
        yield
    finally:
        model._bandtok_grid = previous


def forward_1d(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=None,
    **kwargs,
) -> BaseModelOutputWithPast:
    """``LlamaModel.forward`` with plain 1D RoPE over the flattened sequence.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def forward_2d(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    cache_position=None,
    use_cache=None,
    text_token_len: int = 128,
    num_bands: int = 16,
    **kwargs,
) -> BaseModelOutputWithPast:
    """``LlamaModel.forward`` with 2D (token / time / Mel-band) RoPE.

    The audio grid's origin and width come from :func:`grid_geometry` when it is
    active, and otherwise from the ``text_token_len`` / ``num_bands`` arguments.
    ``text_token_len`` counts the conditioning prefix including ``<BOS>``.
    """
    text_token_len, num_bands = getattr(self, "_bandtok_grid", None) or (text_token_len, num_bands)

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    seq_len = hidden_states.size(1)
    past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
    total_len = past_len + seq_len

    # During training the sequence ends with <EOS>, which is not part of the grid.
    audio_len = total_len - text_token_len - (1 if self.training else 0)

    num_frames = max(1, math.ceil(audio_len / num_bands))
    pos_ids = make_2d_positions(text_token_len, num_frames, num_bands)
    pos_ids = pos_ids[:, past_len : past_len + seq_len]
    pos_ids = pos_ids.unsqueeze(1).repeat(1, hidden_states.size(0), 1).to(hidden_states.device)
    position_embeddings = self.rotary_emb(hidden_states, pos_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


def build_rope_module(rope_2d: Dict[str, Any]) -> Optional[nn.Module]:
    """Construct the 2D RoPE module ``rope_2d`` describes, or ``None`` for plain 1D RoPE.
    """
    if not rope_2d:
        return None

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

    return Qwen3VLTextRotaryEmbedding(types.SimpleNamespace(**rope_2d))


class AudioLanguageModel(nn.Module):
    """Llama backbone over audio tokens, with a projected conditioning prefix.

    Args:
        backbone: a ``LlamaForCausalLM`` whose ``vocab_size`` covers the audio
            codebook plus the special tokens (``<EOS>`` at ``vocab_size - 2``,
            ``<BOS>`` at ``vocab_size - 1``).
        num_quantizers: codebooks per frame position (1 for BandTok).
        codebook_size: entries per codebook.
        prepend_cond_dim: width of the conditioning embeddings to project.
        rope_2d: Interleaved-MRoPE settings; an empty dict selects 1D RoPE.
        num_bands: Mel-band token positions per frame, used by 2D RoPE.
    """

    def __init__(
        self,
        backbone: LlamaForCausalLM,
        num_quantizers: int,
        codebook_size: int,
        prepend_cond_dim: int = 0,
        rope_2d: Optional[Dict[str, Any]] = None,
        num_bands: int = 16,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.embed_dim = backbone.config.hidden_size
        self.rope_2d = rope_2d or {}
        self.num_bands = num_bands

        rotary = build_rope_module(self.rope_2d)
        if rotary is not None:
            self.backbone.model.rotary_emb = rotary
            self.backbone.model.forward = types.MethodType(forward_2d, self.backbone.model)
        else:
            self.backbone.model.forward = types.MethodType(forward_1d, self.backbone.model)

        if prepend_cond_dim > 0:
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, self.embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            )

    @property
    def bos_token_id(self) -> int:
        return self.backbone.config.vocab_size - 1

    @property
    def eos_token_id(self) -> int:
        return self.backbone.config.vocab_size - 2

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone.model.embed_tokens(tokens)

    def add_eos(
        self, sequence: torch.Tensor, padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append ``<EOS>`` after the last valid token of every sequence.

        Sequences that are valid all the way to the end grow by one position;
        shorter ones get ``<EOS>`` written into the first padding slot.
        """
        new_sequence, new_padding_mask = [], []
        eos = self.eos_token_id
        for s, p in zip(sequence, padding_mask):
            if p[-1] == 1:
                p = F.pad(p, pad=(0, 1), value=1)
                s = F.pad(s, pad=(0, 1), value=eos)
            else:
                valid = int(p.sum())
                s[valid : valid + 1] = eos
                p[valid : valid + 1] = 1
            new_sequence.append(s)
            new_padding_mask.append(p)

        new_sequence = torch.nn.utils.rnn.pad_sequence(new_sequence, batch_first=True)
        new_padding_mask = torch.nn.utils.rnn.pad_sequence(new_padding_mask, batch_first=True)
        return new_sequence, new_padding_mask

    def forward(
        self,
        sequence: torch.Tensor,
        prepend_cond: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
    ):
        """Run the backbone over ``[prefix | BOS | audio tokens]``.

        Args:
            sequence: ``[B, L]`` audio token ids (already flattened band-first).
            prepend_cond: ``[B, T_cond, prepend_cond_dim]`` conditioning to project
                and prepend.
            padding_mask: ``[B, L]``, 1 for real tokens. Padded positions are
                excluded from attention and from the loss.
            compute_loss: build shifted labels so the backbone returns a loss.

        Returns:
            A ``CausalLMOutputWithPast``.
        """
        if sequence.ndim == 3:
            sequence = sequence.squeeze(1)
        batch_size = sequence.size(0)
        device = sequence.device

        if prepend_cond is None:
            raise ValueError("BandTok's LM is prefix-conditioned; prepend_cond is required")

        prefix = self.to_prepend_embed(prepend_cond)
        bos = self.embed_tokens(
            torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)
        )
        prefix = torch.cat([prefix, bos], dim=1)
        prefix_len = prefix.size(1)

        x = torch.cat([prefix, self.embed_tokens(sequence)], dim=1)

        attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=device)
        labels = None
        if compute_loss:
            labels = torch.cat(
                [torch.full((batch_size, prefix_len), -100, dtype=torch.long, device=device), sequence],
                dim=-1,
            )
        if padding_mask is not None:
            padding_mask = padding_mask.bool()
            attention_mask[:, prefix_len:][~padding_mask] = 0
            if labels is not None:
                labels[:, prefix_len:][~padding_mask] = -100

        position_ids = torch.arange(x.size(1), device=device).unsqueeze(0).expand(batch_size, -1)

        with grid_geometry(self.backbone.model, prefix_len, self.num_bands):
            return self.backbone(
                inputs_embeds=x,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
            )


class BandTokLM(nn.Module):
    """BandTok tokenizer + autoregressive LM, with conditioning and CFG.

    Args:
        tokenizer: frozen BandTok tokenizer providing the discrete token grid.
        lm: the :class:`AudioLanguageModel`.
        conditioner: multi-conditioner producing caption / segment-time embeddings.
        sample_rate: audio sample rate in Hz.
        prepend_cond_ids: conditioner ids concatenated into the prefix, in order.
    """

    def __init__(
        self,
        tokenizer: BandTok,
        lm: AudioLanguageModel,
        sample_rate: int,
        conditioner: Optional[MultiConditioner] = None,
        prepend_cond_ids: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        if not getattr(tokenizer, "is_discrete", False):
            raise ValueError("the LM requires a discrete tokenizer")

        self.tokenizer = tokenizer
        self.lm = lm
        self.conditioner = conditioner
        self.sample_rate = sample_rate
        self.prepend_cond_ids = prepend_cond_ids or []

        self.num_quantizers = tokenizer.num_quantizers
        self.codebook_size = tokenizer.codebook_size
        self.num_bands = tokenizer.num_bands

    # -- conditioning -----------------------------------------------------

    def get_conditioning_inputs(self, cond: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        """Concatenate the configured conditioners into a single prefix."""
        prepend_cond = None
        prepend_cond_mask = None
        if self.prepend_cond_ids:
            missing = [key for key in self.prepend_cond_ids if key not in cond]
            if missing:
                raise KeyError(f"conditioner(s) {missing} missing from conditioning inputs")
            prepend_cond = torch.cat([cond[key][0] for key in self.prepend_cond_ids], dim=1)
            prepend_cond_mask = torch.cat([cond[key][1] for key in self.prepend_cond_ids], dim=1)
        return {"prepend_cond": prepend_cond, "prepend_cond_mask": prepend_cond_mask}

    def conditioning_from_info(self, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the conditioners over a batch of dataset info dicts."""
        if self.conditioner is None:
            raise ValueError("no conditioner configured")
        return self.conditioner(infos)

    # -- training ---------------------------------------------------------

    def compute_logits(
        self,
        codes: torch.Tensor,
        condition_tensors: Optional[Dict[str, Any]] = None,
        cfg_dropout_prob: float = 0.0,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Teacher-forced forward pass.

        Args:
            codes: ``[B, num_quantizers, L]`` token grid from the tokenizer.
            condition_tensors: output of the conditioner.
            cfg_dropout_prob: probability of replacing the whole conditioning
                prefix with a near-null embedding, which is what enables
                classifier-free guidance at inference time (MusicGen-style).
            padding_mask: ``[B, L]`` token validity mask.

        Returns:
            A ``CausalLMOutputWithPast`` whose ``loss`` is the next-token
            cross-entropy over audio positions only.
        """
        conditioning_inputs = self.get_conditioning_inputs(condition_tensors or {})
        prepend_cond = conditioning_inputs["prepend_cond"]

        if cfg_dropout_prob > 0.0 and prepend_cond is not None:
            null_embed = torch.zeros_like(prepend_cond) + 1e-4
            dropout_mask = torch.bernoulli(
                torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)
            ).to(torch.bool)
            prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

        if codes.ndim == 3:
            if codes.size(1) != 1:
                raise ValueError(
                    f"expected a single codebook after flattening, got {codes.size(1)}; "
                    "multi-codebook tokens must be flattened by the tokenizer"
                )
            codes = codes[:, 0]

        if padding_mask is not None:
            codes, padding_mask = self.lm.add_eos(codes, padding_mask)

        return self.lm(codes, prepend_cond=prepend_cond, padding_mask=padding_mask)

    # -- generation -------------------------------------------------------

    def _sample_next_token(
        self,
        sequence: torch.Tensor,
        prepend_cond: torch.Tensor,
        cfg_scale: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        suppress_eos: bool = False,
        keep_segment_time_cond: bool = False,
        num_caption_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample one token, optionally with classifier-free guidance.

        With ``cfg_scale != 1.0`` the batch is duplicated: the first half keeps the
        conditioning prefix, the second half gets a null prefix, and the logits are
        combined as ``uncond + scale * (cond - uncond)``.
        """
        generated = sequence
        if cfg_scale != 1.0:
            sequence = torch.cat([sequence, sequence], dim=0)
            if keep_segment_time_cond:
                # Drop only the caption, keep the segment-time conditions, so that
                # guidance acts on the text and not on the timing.
                if num_caption_tokens is None:
                    raise ValueError("keep_segment_time_cond requires num_caption_tokens")
                caption_cond = prepend_cond[:, :num_caption_tokens, :]
                time_cond = prepend_cond[:, num_caption_tokens:, :]
                uncond = torch.cat([torch.zeros_like(caption_cond), time_cond], dim=1)
            else:
                uncond = torch.zeros_like(prepend_cond)
            prepend_cond = torch.cat([prepend_cond, uncond], dim=0)

        out = self.lm(sequence, prepend_cond=prepend_cond, compute_loss=False)
        logits = out.logits

        if cfg_scale != 1.0:
            cond_logits, uncond_logits = logits.chunk(2, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale

        logits = logits[:, -1, :]
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9).float()

        if repetition_penalty != 1.0 and generated.size(-1) > 0:
            # Transformers' RepetitionPenaltyLogitsProcessor: every already-generated
            # token has its logit divided by the penalty when positive, multiplied
            # when negative. Applied before the temperature/top-k/top-p warpers,
            # matching Hugging Face's processor-then-warper order.
            prev = generated
            score = logits.gather(1, prev)
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits = logits.scatter(1, prev, score)

        if suppress_eos:
            logits[:, self.lm.eos_token_id] = -float("inf")

        if temperature > 0:
            probs = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
            probs = normalize_probs(probs, logits)
            if top_p > 0.0:
                next_token = sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, k=min(int(top_k), probs.size(-1)))
            else:
                next_token = multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token.unsqueeze(1)

    @torch.no_grad()
    def generate(
        self,
        conditioning: Optional[List[Dict[str, Any]]] = None,
        conditioning_tensors: Optional[Dict[str, Any]] = None,
        max_gen_len: Optional[int] = None,
        cfg_scale: float = 1.0,
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        keep_segment_time_cond: bool = False,
        num_caption_tokens: Optional[int] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """Sample a token sequence.

        Args:
            conditioning: list of per-item condition dicts (``caption``,
                ``seconds_start``, ``seconds_total``), one per batch element.
            conditioning_tensors: pre-computed conditioner output, used instead of
                ``conditioning`` when given.
            max_gen_len: number of audio tokens to sample. Should be a multiple of
                ``num_bands`` so the grid stays rectangular.
            cfg_scale: classifier-free-guidance scale; 1.0 disables guidance and
                takes the faster cached ``generate`` path.
            repetition_penalty: Hugging Face-style repetition penalty over the
                tokens generated so far; 1.0 disables it.
            min_new_tokens: suppress ``<EOS>`` until this many tokens are sampled.
            keep_segment_time_cond: keep segment-time conditions in the negative
                branch, guiding only on the caption.

        Returns:
            ``[B, L]`` sampled token ids, with a trailing ``<EOS>`` removed.
        """
        device = next(self.lm.parameters()).device
        if conditioning_tensors is None:
            if conditioning is None:
                raise ValueError("provide either conditioning or conditioning_tensors")
            self.conditioner.set_device(device)
            conditioning_tensors = self.conditioner(conditioning)

        prepend_cond = self.get_conditioning_inputs(conditioning_tensors)["prepend_cond"]
        if prepend_cond is None:
            raise ValueError("BandTok's LM requires prefix conditioning")

        if max_gen_len is None:
            max_gen_len = self.num_bands * 128
        if max_gen_len % self.num_bands != 0:
            raise ValueError(
                f"max_gen_len={max_gen_len} is not a multiple of num_bands={self.num_bands}; "
                "the token grid would be ragged"
            )

        batch_size = prepend_cond.size(0)
        eos_token_id = self.lm.eos_token_id

        if cfg_scale == 1.0:
            # No guidance: let Transformers handle KV caching. `generate` rejects
            # extra forward kwargs, so the grid geometry is stashed on the module.
            projected = self.lm.to_prepend_embed(prepend_cond)
            bos = self.lm.embed_tokens(
                torch.full((batch_size, 1), self.lm.bos_token_id, dtype=torch.long, device=device)
            )
            inputs_embeds = torch.cat([projected, bos], dim=1)
            gen_config = GenerationConfig(
                max_length=inputs_embeds.size(1) + max_gen_len,
                do_sample=temperature > 0,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p if top_p > 0 else None,
                repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
                min_new_tokens=min_new_tokens if min_new_tokens > 0 else None,
                eos_token_id=eos_token_id,
            )
            with grid_geometry(self.lm.backbone.model, inputs_embeds.size(1), self.num_bands):
                out = self.lm.backbone.generate(
                    inputs_embeds=inputs_embeds, generation_config=gen_config
                )
        else:
            sequence = torch.zeros((batch_size, 0), device=device, dtype=torch.long)
            steps = trange(int(max_gen_len), disable=not progress, desc="sampling")
            for step in steps:
                next_token = self._sample_next_token(
                    sequence=sequence,
                    prepend_cond=prepend_cond,
                    cfg_scale=cfg_scale,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    suppress_eos=step < min_new_tokens,
                    keep_segment_time_cond=keep_segment_time_cond,
                    num_caption_tokens=num_caption_tokens,
                )
                sequence = torch.cat([sequence, next_token[:, 0]], dim=-1)
                if (next_token[:, 0, 0] == eos_token_id).all():
                    break
            out = sequence

        if out.size(-1) > 0 and (out[:, -1] == eos_token_id).all():
            out = out[:, :-1]
        return out

    @torch.no_grad()
    def generate_audio(self, codes: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens and decode them to audio.

        Returns:
            ``(audio [B, 1, N], codes [B, L])``.
        """
        if codes is None:
            codes = self.generate(**kwargs)
        return self.decode_codes_to_audio(codes), codes

    @torch.no_grad()
    def decode_codes_to_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """Turn sampled tokens into audio, dropping special/invalid ids."""
        dec_codes = codes
        if dec_codes.ndim == 1:
            dec_codes = dec_codes.unsqueeze(0)

        invalid = (dec_codes < 0) | (dec_codes >= self.codebook_size)
        num_invalid = int(invalid.sum())
        if num_invalid:
            print(f"[bandtok.lm] replacing {num_invalid} special/out-of-range token(s) with 0 before decoding")
            dec_codes = dec_codes.masked_fill(invalid, 0)

        seq_len = dec_codes.size(-1)
        valid_len = (seq_len // self.num_bands) * self.num_bands
        if valid_len == 0:
            raise ValueError(f"cannot decode {seq_len} tokens: fewer than num_bands={self.num_bands}")
        if valid_len != seq_len:
            print(f"[bandtok.lm] truncating {seq_len} tokens to {valid_len} to fill whole frames")
            dec_codes = dec_codes[..., :valid_len]

        if dec_codes.ndim == 2:
            dec_codes = dec_codes.unsqueeze(1)  # [B, 1, L]
        return self.tokenizer.decode_tokens(dec_codes)


def create_lm_from_config(config: Dict[str, Any], tokenizer: BandTok) -> BandTokLM:
    """Build a :class:`BandTokLM` from a model config and a loaded tokenizer."""
    model_config = config["model"]
    sample_rate = config["sample_rate"]

    lm_config = model_config.get("lm")
    if lm_config is None:
        raise ValueError("model config must contain a 'lm' section")

    conditioner = None
    if model_config.get("conditioning") is not None:
        conditioner = create_multi_conditioner_from_config(model_config["conditioning"])

    lm_type = lm_config.get("type", "llama")
    if lm_type not in ("llama", "llama_simple"):
        raise ValueError(f"unknown lm type {lm_type!r}; expected 'llama'")

    backbone_config = dict(lm_config["config"])
    prepend_cond_dim = backbone_config.pop("prepend_cond_dim", 0)
    rope_2d = backbone_config.pop("rope_2d", {})
    if "depth" in backbone_config:
        raise ValueError(
            "'depth' is not a Llama config field; "
            "use 'num_hidden_layers' instead"
        )

    backbone = LlamaForCausalLM(config=LlamaConfig(**backbone_config))

    lm = AudioLanguageModel(
        backbone=backbone,
        num_quantizers=tokenizer.num_quantizers,
        codebook_size=tokenizer.codebook_size,
        prepend_cond_dim=prepend_cond_dim,
        rope_2d=rope_2d,
        num_bands=tokenizer.num_bands,
    )

    return BandTokLM(
        tokenizer=tokenizer,
        lm=lm,
        sample_rate=sample_rate,
        conditioner=conditioner,
        prepend_cond_ids=lm_config.get("prepend_cond_ids", []),
    )
