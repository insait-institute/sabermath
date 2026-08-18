#!/bin/bash
#SBATCH --job-name=reranker_reason_embed_qwen3_8b
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=1-0
#SBATCH --output=/scratch/%u/logs/reranker_reason_embed_qwen3_8b.log
#SBATCH --error=/scratch/%u/logs/reranker_reason_embed_qwen3_8b.error

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_reason_embed_qwen3_8b.slurm
#   sbatch scripts/rerankers/run_reason_embed_qwen3_8b.slurm --n 20   # smoke test first
#
# Plain bi-encoder embedding model (confirmed via its own model card -
# SentenceTransformer-loadable, separate query/document embeddings scored by
# cosine similarity) - runs through the generic SentenceTransformersProcessor
# path, single GPU, no tensor parallelism needed.
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

# See run_rader_14b.slurm for why: fp32-by-default plus allocator
# fragmentation OOM'd the 14B sibling model partway through a run. Applied
# here too as a precaution, even though 8B is much less likely to hit the
# same wall.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkenv -n reranker-st-reason-embed -f scripts/rerankers/envs/env_st_biencoders.yml -- \
  python scripts/run_rerankers.py --models reason-embed-qwen3-8b "$@"

echo "Done."
