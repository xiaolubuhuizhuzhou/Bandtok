#!/usr/bin/env python3
"""Export an inference-only tokenizer checkpoint from a training checkpoint.

A training checkpoint carries the optimizer and scheduler state, the
discriminator, and -- because the frozen BigVGAN auto-registers itself as a
submodule the first time a demo runs -- 667 ``_vocoder.*`` tensors the repo
loads separately anyway. This writes just the codec's own weights, which is
what an LM config's ``pretransform.ckpt_path`` wants: smaller, and it reloads
with ``strict=True``.

    python scripts/export_tokenizer_ckpt.py \
        --model-config configs/tokenizer/bandtok.json \
        --ckpt runs/tokenizer_bandtok/checkpoints/last.pt \
        --out ckpts/bandtok.pt

Exporting also pins the weights: a run's own ``step_*.pt`` files are subject to
``keep_last_k`` pruning, so anything an LM run depends on belongs outside the
run directory.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bandtok.config import load_model_config  # noqa: E402
from bandtok.models.factory import create_tokenizer_from_config  # noqa: E402
from bandtok.utils.checkpoint import export_model, load_model_weights  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--ckpt", required=True, help="training checkpoint to export from")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="export even if the training checkpoint does not cover every model parameter",
    )
    args = parser.parse_args()

    model_config = load_model_config(args.model_config)
    model_type = model_config.get("model_type")
    if model_type != "bandtok":
        raise ValueError(f"expected a tokenizer config (model_type 'bandtok'), got {model_type!r}")
    tokenizer = create_tokenizer_from_config(model_config)

    missing, _ = load_model_weights(tokenizer, args.ckpt, strict=False, name="tokenizer")
    if missing and not args.allow_missing:
        raise SystemExit(
            f"[bandtok] {len(missing)} model parameters are not in {args.ckpt}: {missing[:5]}; "
            "check the config matches the run, or pass --allow-missing"
        )

    # The vocoder is loaded lazily and never touched here, so it stays out of
    # state_dict() and the export is the codec alone.
    export_model(tokenizer, args.out, config=model_config)

    # The point of exporting is a checkpoint that loads exactly; prove it.
    verify = create_tokenizer_from_config(model_config)
    load_model_weights(verify, args.out, strict=True, name="re-loaded export")
    size_gib = os.path.getsize(args.out) / 2**30
    print(f"[bandtok] {args.out}: {size_gib:.2f} GiB, reloads with strict=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
