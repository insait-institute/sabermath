#!/bin/bash
#SBATCH --job-name=reranker_diver_retriever_4b
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=1-0
#SBATCH --output=/scratch/%u/logs/reranker_diver_retriever_4b.log
#SBATCH --error=/scratch/%u/logs/reranker_diver_retriever_4b.error

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_diver_retriever_4b.slurm
#   sbatch scripts/rerankers/run_diver_retriever_4b.slurm --n 20   # smoke test first
#
# Plain bi-encoder embedding model (Qwen3-embedding architecture, part of the
# DIVER retrieval system - confirmed via its paper/model card, not a
# cross-encoder) - runs through the generic SentenceTransformersProcessor
# path, single GPU, no tensor parallelism needed.
#
# NOTE the real repo is AQ-MedAI/Diver-Retriever-4B (not "Driver") - see
# GENERIC_MODELS in run_rerankers.py.
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
# here too as a precaution, even though 4B is much less likely to hit the
# same wall.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkenv -n reranker-st-diver -f scripts/rerankers/envs/env_st_biencoders.yml -- \
  python scripts/run_rerankers.py --models diver-retriever-4b "$@"

echo "Done."
