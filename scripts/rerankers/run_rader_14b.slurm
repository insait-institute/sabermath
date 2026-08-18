#!/bin/bash
#SBATCH --job-name=reranker_rader_14b
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=1-0
#SBATCH --output=/scratch/%u/logs/reranker_rader_14b.log
#SBATCH --error=/scratch/%u/logs/reranker_rader_14b.error

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_rader_14b.slurm
#   sbatch scripts/rerankers/run_rader_14b.slurm --n 20   # smoke test first
#
# Plain bi-encoder embedding model (Qwen2.5-14B based reasoning-aware dense
# retriever - confirmed via its paper, not a cross-encoder) - runs through
# the generic SentenceTransformersProcessor path, single GPU, no tensor
# parallelism needed.
set -e
cd "${SABERMATH_REPO:-$HOME/sabermath}"

export SABERMATH_CANONICAL_HOST="${SABERMATH_CANONICAL_HOST:-hala}"
export SABERMATH_CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/maria_drencheva/sabermath}"
if [[ "$(hostname)" != "$SABERMATH_CANONICAL_HOST" ]]; then
  echo "[~] Syncing latest code + any existing progress from $SABERMATH_CANONICAL_HOST (this node's home storage may not be the canonical copy)..."
  rsync -auz --exclude=".git" \
    "$SABERMATH_CANONICAL_HOST:$SABERMATH_CANONICAL_PATH/" "./"
  export NEEDS_REGION_SYNC=1
fi

source scripts/rerankers/_common.sh

# Confirmed the hard way: 14B params defaults to fp32 (~56GiB) via
# sentence-transformers unless told otherwise (see model_kwargs.torch_dtype
# in GENERIC_MODELS, run_rerankers.py) - OOM'd a 140GiB H200 partway through
# a run from PyTorch allocator fragmentation across many variable-shaped
# encode() calls. expandable_segments reduces that fragmentation (this is
# exactly what the OOM error itself recommended).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkenv -n reranker-st-rader -f scripts/rerankers/envs/env_st_biencoders.yml -- \
  python scripts/run_rerankers.py --models rader-14b "$@"

echo "Done."
