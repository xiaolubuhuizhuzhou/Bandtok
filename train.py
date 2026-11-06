#!/usr/bin/env python
"""Train the BandTok tokenizer or the BandTok language model.

Single GPU:

    python train.py --model-config configs/tokenizer/bandtok.json \\
                    --data-config configs/data/tokenizer_train.json --run-dir runs/bandtok

Eight GPUs:

    torchrun --nproc_per_node 8 train.py \\
        --model-config configs/tokenizer/bandtok.json \\
        --data-config configs/data/tokenizer_train.json \\
        --run-dir runs/bandtok --batch-size 128

``--batch-size`` is per GPU.
"""

import argparse
import os

# The T5 tokenizer is used before the dataloader forks its workers, which makes
# HuggingFace print a parallelism warning on every run. We do not tokenize inside
# the workers, so disabling it costs nothing.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from bandtok.config import get_training_config, load_data_config, load_model_config, save_config
from bandtok.data.dataset import create_dataloader_from_config
from bandtok.models.factory import create_model_from_config
from bandtok.training.distributed import (
    cleanup_distributed,
    get_world_size,
    print_once,
    set_seed,
    setup_distributed,
)
from bandtok.training.logger import MetricsLogger
from bandtok.training.trainer import Trainer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-config", required=True, help="tokenizer or LM model config (JSON)")
    parser.add_argument("--data-config", required=True, help="training dataset config (JSON)")
    parser.add_argument("--val-data-config", default=None, help="optional validation dataset config")
    parser.add_argument("--run-dir", default=None, help="output directory (default: runs/<config name>)")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size PER GPU")
    parser.add_argument("--num-workers", type=int, default=8, help="dataloader workers per GPU")
    parser.add_argument("--accum-batches", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume training; with no value, resume from <run-dir>/checkpoints/last.pt",
    )
    parser.add_argument(
        "--pretrained", default=None, help="initialize the model from these weights (no optimizer state)"
    )
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--tensorboard", action="store_true", help="log to TensorBoard")
    parser.add_argument("--project", default="bandtok", help="W&B project name")
    parser.add_argument("--run-name", default=None, help="run name for logging")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    device = setup_distributed()
    set_seed(args.seed)

    model_config = load_model_config(args.model_config)
    data_config = load_data_config(args.data_config)
    training_config = get_training_config(model_config)

    run_name = args.run_name or os.path.splitext(os.path.basename(args.model_config))[0]
    run_dir = args.run_dir or os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_loader = create_dataloader_from_config(
        data_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
    )
    val_loader = None
    if args.val_data_config:
        val_loader = create_dataloader_from_config(
            load_data_config(args.val_data_config),
            batch_size=args.batch_size,
            num_workers=max(1, args.num_workers // 2),
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 1),
            shuffle=False,
            drop_last=False,
        )

    model = create_model_from_config(model_config)

    if args.pretrained:
        from bandtok.utils.checkpoint import load_model_weights

        load_model_weights(model, args.pretrained, strict=False, name=model_config["model_type"])

    if model_config["model_type"] == "bandtok":
        from bandtok.training.tokenizer_task import create_tokenizer_task

        task = create_tokenizer_task(model_config, model)
    elif model_config["model_type"] == "bandtok_lm":
        from bandtok.training.lm_task import create_lm_task

        task = create_lm_task(model_config, model)
    else:
        # Never fall through to the LM task: a tokenizer model_type that is not
        # listed here would otherwise fail deep inside LanguageModelTask looking
        # for a `.tokenizer` attribute it does not have.
        raise ValueError(
            f"unknown model_type {model_config['model_type']!r}; expected 'bandtok' or 'bandtok_lm'"
        )

    effective_batch = args.batch_size * get_world_size() * args.accum_batches
    print_once(
        f"[bandtok] {model_config['model_type']} run '{run_name}' in {run_dir}\n"
        f"[bandtok] effective batch size {effective_batch} "
        f"({args.batch_size} per GPU x {get_world_size()} GPU(s) x {args.accum_batches} accum)"
    )

    logger = MetricsLogger(
        log_dir=run_dir,
        log_every=int(training_config.get("logging", {}).get("log_every", 20)),
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        project=args.project,
        run_name=run_name,
        config={"model": model_config, "data": data_config, "args": vars(args)},
    )

    trainer = Trainer(
        task=task,
        train_loader=train_loader,
        val_loader=val_loader,
        training_config=training_config,
        run_dir=run_dir,
        device=device,
        max_steps=args.max_steps,
        accum_batches=args.accum_batches,
        logger=logger,
        config={"model": model_config, "data": data_config},
    )

    save_config(model_config, os.path.join(run_dir, "model_config.json"))
    save_config(data_config, os.path.join(run_dir, "data_config.json"))

    if args.resume:
        trainer.resume(None if args.resume == "auto" else args.resume)

    try:
        trainer.fit()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
