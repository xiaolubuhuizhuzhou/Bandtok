#!/usr/bin/env python
"""Text-to-music generation with a trained BandTok language model.

Single prompt:

    python scripts/generate.py --model-config configs/lm/bandtok.json \\
        --ckpt runs/bandtok_lm/checkpoints/last.pt --output-dir out \\
        --prompt "a slow lo-fi hip hop beat with a warm rhodes chord loop"

A batch of prompts, eight GPUs (each rank takes every eighth prompt):

    torchrun --nproc_per_node 8 scripts/generate.py \\
        --model-config configs/lm/bandtok.json \\
        --ckpt runs/bandtok_lm/checkpoints/last.pt \\
        --output-dir out --prompts-file prompts.jsonl --duration 10

The defaults are the paper's inference settings: classifier-free guidance 2.0,
top-k 250, temperature 1.0.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio

from bandtok.config import load_model_config, precision_to_dtype
from bandtok.models.factory import create_model_from_config
from bandtok.models.lm import BandTokLM
from bandtok.models.tokenizer import BandTok
from bandtok.training.distributed import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    print_once,
    set_seed,
    setup_distributed,
)
from bandtok.utils.checkpoint import load_model_weights, load_state_dict_file


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True, help="LM model config (JSON)")
    parser.add_argument("--ckpt", required=True, help="trained LM weights (a training or exported checkpoint)")
    parser.add_argument("--output-dir", required=True, help="directory for the generated wavs and the manifest")

    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", action="append", default=None, help="caption to generate (repeatable)")
    prompts.add_argument(
        "--prompts-file",
        default=None,
        help="JSONL with one {caption, seconds_start, seconds_total, name} object per line",
    )
    prompts.add_argument(
        "--prompts-yaml",
        default=None,
        help="YAML mapping of name -> {caption, seconds_start, seconds_total}",
    )

    parser.add_argument(
        "--cfg-scale", type=float, default=2.0, help="classifier-free guidance scale (paper: 2.0)"
    )
    parser.add_argument("--top-k", type=int, default=250, help="top-k sampling cutoff, 0 disables (paper: 250)")
    parser.add_argument("--top-p", type=float, default=0.0, help="nucleus sampling cutoff, 0 disables")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="sampling temperature, 0 is greedy (paper: 1.0)"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Hugging Face-style repetition penalty over generated tokens, 1.0 disables",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="suppress <EOS> until this many tokens have been sampled",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="length to generate in seconds (default: 10)")
    parser.add_argument(
        "--num-samples-per-prompt", type=int, default=1, help="samples drawn per prompt (default: 1)"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="prompts sampled together (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-segment-time-cond",
        action="store_true",
        help="guide on the caption only, keeping the segment-time conditions in the negative branch",
    )
    parser.add_argument(
        "--precision",
        default=None,
        help="autocast precision on CUDA (default: the config's training.precision)",
    )
    parser.add_argument(
        "--vocoder-path",
        default=None,
        help="local BigVGAN directory or HF id (default: $BANDTOK_VOCODER_PATH, else the config's)",
    )
    return parser


# -- prompts --------------------------------------------------------------


def _safe_name(name: str) -> str:
    """Make ``name`` usable as a file name."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_") or "prompt"


def _normalize_prompt(record: Dict[str, Any], index: int, duration: float) -> Dict[str, Any]:
    """Fill in the conditioning fields a prompt record may leave out."""
    caption = record.get("caption", record.get("prompt", ""))
    if not isinstance(caption, str):
        raise ValueError(f"prompt {index}: 'caption' must be a string, got {caption!r}")
    return {
        "name": _safe_name(str(record["name"])) if record.get("name") else f"{index:04d}",
        "caption": caption,
        # The paper conditions on where the segment sits inside the track; a bare
        # prompt is treated as a track that is exactly as long as the generation.
        "seconds_start": float(record.get("seconds_start", 0.0)),
        "seconds_total": float(record.get("seconds_total", duration)),
    }


def load_prompts(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Read the prompts from whichever of the three input flags was given."""
    if args.prompt:
        records: List[Dict[str, Any]] = [{"caption": caption} for caption in args.prompt]
    elif args.prompts_file:
        records = []
        with open(args.prompts_file, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{args.prompts_file}:{line_no}: expected a JSON object")
                records.append(record)
    else:
        import yaml

        with open(args.prompts_yaml, "r", encoding="utf-8") as handle:
            mapping = yaml.safe_load(handle)
        if not isinstance(mapping, dict):
            raise ValueError(f"{args.prompts_yaml} must be a mapping of name -> prompt fields")
        records = []
        for name, record in mapping.items():
            record = {"caption": record} if isinstance(record, str) else dict(record)
            record.setdefault("name", name)
            records.append(record)

    prompts = [_normalize_prompt(record, index, args.duration) for index, record in enumerate(records)]
    if not prompts:
        raise ValueError("no prompts found")

    seen: Dict[str, int] = {}
    for prompt in prompts:
        count = seen.get(prompt["name"], 0)
        seen[prompt["name"]] = count + 1
        if count:
            prompt["name"] = f"{prompt['name']}-{count}"
    return prompts


# -- geometry -------------------------------------------------------------


def duration_to_num_tokens(tokenizer: BandTok, seconds: float) -> Tuple[int, int]:
    """Convert a duration in seconds to ``(num_tokens, num_frames)``.

    ``BandTokLM.generate`` rejects a token count that does not fill whole frames,
    so the duration is rounded to the nearest latent frame (~10.77 Hz for the
    paper's config) before being multiplied by the tokens per frame.
    """
    if seconds <= 0:
        raise ValueError(f"--duration must be positive, got {seconds}")
    num_frames = max(1, round(seconds * tokenizer.frame_rate))
    return num_frames * tokenizer.num_bands * tokenizer.num_quantizers, num_frames


# -- checkpoint -----------------------------------------------------------


def _checkpoint_tensors(path: str) -> Dict[str, Any]:
    """Pull the tensor dict out of a checkpoint, mirroring ``checkpoint.py``."""
    payload = load_state_dict_file(path)
    for key in ("model", "state_dict"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def load_lm_weights(model: BandTokLM, path: str) -> str:
    """Load ``path`` into the LM, detecting which module the weights belong to.

    A training checkpoint written by ``train.py`` holds ``task.modules["model"]``,
    which for an LM run *is* ``model.lm`` -- so its keys start with ``backbone.``.
    A checkpoint exported from the whole wrapper instead has ``lm.``/``tokenizer.``
    keys, and a checkpoint saved from an outer wrapper prefixes those keys with the
    module attribute name (``model.lm.`` ...). The key sets are disjoint, so
    counting how many keys each module recognizes -- after stripping a wrapper
    prefix where that helps -- identifies the layout. Returns ``"lm"`` or
    ``"bandtok_lm"``.
    """
    keys = set(_checkpoint_tensors(path))

    def hits(target: torch.nn.Module) -> int:
        target_keys = set(target.state_dict())
        direct = len(keys & target_keys)
        # A wrapper module or DDP prefixes every key; count matches after
        # stripping each candidate first segment and keep the best.
        stripped = 0
        for prefix in {key.split(".", 1)[0] for key in keys if "." in key}:
            candidate = {key[len(prefix) + 1 :] for key in keys if key.startswith(prefix + ".")}
            stripped = max(stripped, len(candidate & target_keys))
        return max(direct, stripped)

    lm_hits = hits(model.lm)
    wrapper_hits = hits(model)

    if wrapper_hits > lm_hits:
        layout, target = "bandtok_lm", model
        print_once(f"[bandtok] {path}: whole-wrapper layout ({wrapper_hits} matching keys)")
    else:
        layout, target = "lm", model.lm
        print_once(f"[bandtok] {path}: LM-module layout ({lm_hits} matching keys)")

    load_model_weights(target, path, strict=False, name=layout)
    return layout


def apply_vocoder_path(tokenizer: BandTok, vocoder_path: Optional[str]) -> None:
    """Point the tokenizer at a specific BigVGAN checkpoint.

    The explicit flag has to win over a path baked into the tokenizer config.
    ``$BANDTOK_VOCODER_PATH`` is set too, for any other ``load_vocoder`` caller in
    the process.
    """
    if not vocoder_path:
        return
    os.environ["BANDTOK_VOCODER_PATH"] = vocoder_path
    tokenizer.set_vocoder_path(vocoder_path)


# -- generation -----------------------------------------------------------


def prompt_to_info(prompt: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a prompt record into the metadata dict the conditioners read."""
    return {
        "caption": prompt["caption"],
        "prompt": prompt["caption"],  # alias accepted by MultiConditioner.default_keys
        "seconds_start": prompt["seconds_start"],
        "seconds_total": prompt["seconds_total"],
    }


def _batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {batch_size}")
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def _autocast(device: torch.device, amp_dtype: Optional[torch.dtype]):
    """Autocast only where it is useful; fp32 configs and CPU runs get a no-op."""
    if amp_dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def main() -> None:
    args = build_argparser().parse_args()

    device = setup_distributed()
    set_seed(args.seed)

    model_config = load_model_config(args.model_config)
    if model_config["model_type"] != "bandtok_lm":
        raise ValueError(
            f"--model-config must be a 'bandtok_lm' config, got {model_config['model_type']!r}; "
            "use scripts/reconstruct.py for a tokenizer config"
        )
    precision = args.precision if args.precision is not None else model_config["training"].get("precision")
    amp_dtype = precision_to_dtype(precision)

    model = create_model_from_config(model_config)
    load_lm_weights(model, args.ckpt)
    apply_vocoder_path(model.tokenizer, args.vocoder_path)

    model = model.to(device).eval()
    model.requires_grad_(False)
    if model.conditioner is not None:
        model.conditioner.set_device(device)

    tokenizer = model.tokenizer
    max_gen_len, num_frames = duration_to_num_tokens(tokenizer, args.duration)
    print_once(
        f"[bandtok] {args.duration:.2f}s -> {num_frames} frames x {tokenizer.num_bands} band(s) x "
        f"{tokenizer.num_quantizers} codebook(s) = {max_gen_len} tokens\n"
        f"[bandtok] {tokenizer.token_rate:.1f} tokens/s, {tokenizer.bitrate / 1000:.2f} kbps, "
        f"precision={precision or 'fp32'} on {device}"
    )

    prompts = load_prompts(args)
    world_size, rank = get_world_size(), get_rank()
    shard = prompts[rank::world_size] if world_size > 1 else prompts
    print_once(
        f"[bandtok] {len(prompts)} prompt(s) x {args.num_samples_per_prompt} sample(s) "
        f"over {world_size} rank(s)"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    settings = {
        "cfg_scale": args.cfg_scale,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "min_new_tokens": args.min_new_tokens,
        "duration": args.duration,
        "max_gen_len": max_gen_len,
        "seed": args.seed,
        "keep_segment_time_cond": args.keep_segment_time_cond,
        "precision": precision,
        "model_config": os.path.abspath(args.model_config),
        "ckpt": os.path.abspath(args.ckpt),
    }
    # One manifest per rank, so that sharded runs never interleave writes.
    manifest_name = "generation_manifest.jsonl" if world_size == 1 else f"generation_manifest.rank{rank}.jsonl"
    manifest_path = os.path.join(args.output_dir, manifest_name)

    written = 0
    with torch.inference_mode(), open(manifest_path, "w", encoding="utf-8") as manifest:
        for batch in _batches(shard, args.batch_size):
            infos = [prompt_to_info(prompt) for prompt in batch]
            condition_tensors = model.conditioning_from_info(infos)
            # Guiding on the caption alone needs to know where the caption ends
            # inside the concatenated prefix.
            num_caption_tokens = (
                int(condition_tensors["caption"][0].shape[1]) if "caption" in condition_tensors else None
            )

            for sample_index in range(args.num_samples_per_prompt):
                with _autocast(device, amp_dtype):
                    audio, codes = model.generate_audio(
                        conditioning_tensors=condition_tensors,
                        max_gen_len=max_gen_len,
                        cfg_scale=args.cfg_scale,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        min_new_tokens=args.min_new_tokens,
                        keep_segment_time_cond=args.keep_segment_time_cond,
                        num_caption_tokens=num_caption_tokens,
                        progress=is_main_process(),
                    )

                for index, prompt in enumerate(batch):
                    # A single sample per prompt keeps the prompt's own name, so the
                    # file ids line up with external text-audio scorers (CLAP score).
                    if args.num_samples_per_prompt == 1:
                        filename = f"{prompt['name']}.wav"
                    else:
                        filename = f"{prompt['name']}_{sample_index}.wav"
                    wav = audio[index].detach().float().clamp(-1.0, 1.0).cpu()
                    torchaudio.save(os.path.join(args.output_dir, filename), wav, model.sample_rate)
                    manifest.write(
                        json.dumps(
                            {
                                "filename": filename,
                                "caption": prompt["caption"],
                                "seconds_start": prompt["seconds_start"],
                                "seconds_total": prompt["seconds_total"],
                                "sample_index": sample_index,
                                "num_tokens": int(codes.shape[-1]),
                                "num_samples": int(wav.shape[-1]),
                                "sample_rate": model.sample_rate,
                                **settings,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    manifest.flush()
                    written += 1

    print(f"[bandtok] rank {rank}: wrote {written} wav(s) and {manifest_path}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
