#!/usr/bin/env bash
#
# BandTok training launcher.
#
# Train the tokenizer (the default), on all visible GPUs:
#
#     bash train.sh
#
# Train the language model (needs a trained tokenizer checkpoint first):
#
#     bash train.sh lm
#
# Everything is overridable from the environment, e.g.
#
#     BATCH_SIZE=64 RUN_DIR=runs/my_run bash train.sh tokenizer
#     STAGE=lm MODEL_CONFIG=configs/lm/bandtok_L.json BATCH_SIZE=4 ACCUM_BATCHES=4 bash train.sh
#
# The run directory defaults to runs/<stage>_<model-config-name>, so pointing
# MODEL_CONFIG at a different config (e.g. a tokenizer ablation) gets its own run
# directory automatically; set RUN_DIR to override. For example
#
#     MODEL_CONFIG=configs/tokenizer/ablations/bandtok_codebook_loss.json bash train.sh
#
# writes to runs/tokenizer_bandtok_codebook_loss.
#
# The shipped data configs point at placeholder dataset roots; edit them (or pass
# DATA_CONFIG / VAL_DATA_CONFIG) to point at your copies, see docs/DATA.md.
#
# Multi-node: set the standard torchrun rendezvous variables on every node:
#
#     NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 bash train.sh
set -euo pipefail

cd "$(dirname "$0")"

# Stage: "tokenizer" (default) or "lm". Accept it as $1 or via $STAGE.
STAGE="${1:-${STAGE:-tokenizer}}"

case "${STAGE}" in
  tokenizer|tok|vae)
    STAGE="tokenizer"
    MODEL_CONFIG="${MODEL_CONFIG:-configs/tokenizer/bandtok.json}"
    DATA_CONFIG="${DATA_CONFIG:-configs/data/tokenizer_train.json}"
    BATCH_SIZE="${BATCH_SIZE:-128}"   
    NUM_WORKERS="${NUM_WORKERS:-8}"
    SEED="${SEED:-43}"
    DEFAULT_WANDB_PROJECT="STFTVAE"
    ;;
  lm|ar)
    STAGE="lm"
    MODEL_CONFIG="${MODEL_CONFIG:-configs/lm/bandtok.json}"
    DATA_CONFIG="${DATA_CONFIG:-configs/data/lm_train.json}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    NUM_WORKERS="${NUM_WORKERS:-12}"
    SEED="${SEED:-49}"
    DEFAULT_WANDB_PROJECT="LM"
    ;;
  *)
    echo "usage: bash train.sh [tokenizer|lm]" >&2
    exit 2
    ;;
esac

# Default the run directory to the model config's name, so each config (in
# particular the tokenizer ablations) gets its own run directory without RUN_DIR
# being set by hand. Override by exporting RUN_DIR.
CONFIG_NAME="$(basename "${MODEL_CONFIG}" .json)"
RUN_DIR="${RUN_DIR:-runs/${STAGE}_${CONFIG_NAME}}"

ACCUM_BATCHES="${ACCUM_BATCHES:-1}"
MAX_STEPS="${MAX_STEPS:--1}"

# GPUs / rendezvous.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  DEFAULT_NPROC=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
else
  DEFAULT_NPROC=$(python -c 'import torch; print(torch.cuda.device_count() or 1)')
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC}}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Keep the frozen T5 tokenizer from warning about the dataloader fork.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Weights & Biases. Enabled with WANDB=1 (or WANDB=true). Logs online by default;
# on a network-isolated node either export http_proxy/https_proxy (inherited by
# the workers) or set WANDB_MODE=offline and `wandb sync` the run later.
# The project defaults per stage (STFTVAE for the tokenizer, LM for the LM);
# override with WANDB_PROJECT. Set
# WANDB_API_KEY in the environment to choose the account.
WANDB_ARGS=()
case "${WANDB:-}" in
  1|true|True|yes|on)
    export WANDB_MODE="${WANDB_MODE:-online}"
    export WANDB_PROJECT="${WANDB_PROJECT:-${DEFAULT_WANDB_PROJECT}}"
    WANDB_ARGS+=(--wandb --project "${WANDB_PROJECT}" --run-name "$(basename "${RUN_DIR}")")
    echo "   W&B          : on (mode=${WANDB_MODE}, project=${WANDB_PROJECT})"
    ;;
esac

EXTRA_ARGS=()
[ -n "${VAL_DATA_CONFIG:-}" ] && EXTRA_ARGS+=(--val-data-config "${VAL_DATA_CONFIG}")
[ -n "${RESUME:-}" ] && EXTRA_ARGS+=(--resume "${RESUME}")
[ -n "${PRETRAINED:-}" ] && EXTRA_ARGS+=(--pretrained "${PRETRAINED}")
EXTRA_ARGS+=("${WANDB_ARGS[@]}")

EFFECTIVE=$(( BATCH_SIZE * NPROC_PER_NODE * NNODES * ACCUM_BATCHES ))
echo "======================================================================"
echo " BandTok ${STAGE} training"
echo "   model config : ${MODEL_CONFIG}"
echo "   data config  : ${DATA_CONFIG}"
[ -n "${VAL_DATA_CONFIG:-}" ] && echo "   val config   : ${VAL_DATA_CONFIG}"
echo "   run dir      : ${RUN_DIR}"
echo "   GPUs         : ${NNODES} node(s) x ${NPROC_PER_NODE} = $(( NNODES * NPROC_PER_NODE ))"
echo "   batch size   : ${BATCH_SIZE}/GPU x ${ACCUM_BATCHES} accum -> ${EFFECTIVE} effective"
echo "======================================================================"

torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    train.py \
    --model-config "${MODEL_CONFIG}" \
    --data-config "${DATA_CONFIG}" \
    --run-dir "${RUN_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --accum-batches "${ACCUM_BATCHES}" \
    --num-workers "${NUM_WORKERS}" \
    --max-steps "${MAX_STEPS}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"
