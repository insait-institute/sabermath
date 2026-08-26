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
# --chdir: slurmd otherwise tries to chdir into the *submit-time* working
# directory before the script runs - on a region whose home storage lacks
# this repo (e.g. msp3) that path doesn't exist, so it logs a noisy
# "couldn't chdir ... going to /tmp instead" error. Start in /tmp
# explicitly; the script cd's into the repo itself below.
#SBATCH --chdir=/tmp

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_reason_embed_qwen3_8b.slurm
#   sbatch scripts/rerankers/run_reason_embed_qwen3_8b.slurm --n 20   # smoke test first
#
# Plain bi-encoder embedding model (confirmed via its own model card -
# SentenceTransformer-loadable, separate query/document embeddings scored by
# cosine similarity) - runs through the generic SentenceTransformersProcessor
# path, single GPU, no tensor parallelism needed.
set -e

# On a region whose home storage has NEVER held this repo (msp3 for this
# account), $HOME/sabermath doesn't exist at all, so a bare `cd` died here
# before the rsync self-heal below ever got a chance to populate it
# (confirmed: job 716300 on msp3-1 failed exactly this way). Create the
# directory first - a no-op everywhere the repo already exists - and let
# the self-heal pull fill it.
mkdir -p "${SABERMATH_REPO:-$HOME/sabermath}"
cd "${SABERMATH_REPO:-$HOME/sabermath}"

export SABERMATH_CANONICAL_HOST="${SABERMATH_CANONICAL_HOST:-hala}"
export SABERMATH_CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/ivo_petrov/sabermath}"
if [[ "$(hostname)" != "$SABERMATH_CANONICAL_HOST" ]]; then
  echo "[~] Syncing latest code + any existing progress from $SABERMATH_CANONICAL_HOST (this node's home storage may not be the canonical copy)..."
  set +e
  rsync -auz --exclude=".git" --exclude="*.tmp" --exclude=".*.??????" \
    "$SABERMATH_CANONICAL_HOST:$SABERMATH_CANONICAL_PATH/" "./"
  rsync_rc=$?
  set -e
  if [[ $rsync_rc -ne 0 && $rsync_rc -ne 23 && $rsync_rc -ne 24 ]]; then
    echo "[!!] Pulling the repo from $SABERMATH_CANONICAL_HOST:$SABERMATH_CANONICAL_PATH failed (rsync exit $rsync_rc)."
    echo "[!!] This node's home storage has no usable copy, so the job cannot continue."
    echo "[!!] Check SSH reachability/keys from $(hostname) to $SABERMATH_CANONICAL_HOST,"
    echo "[!!] and that the canonical path exists there (override via SABERMATH_CANONICAL_HOST/_PATH)."
    exit 1
  fi
  if [[ $rsync_rc -eq 23 || $rsync_rc -eq 24 ]]; then
    echo "[~] rsync reported vanished/unreadable source files (exit $rsync_rc) - another job was checkpointing on the canonical host mid-pull; continuing."
  fi
  # Belt-and-braces: even a "successful" rsync against a wrong/empty source
  # leaves an unusable tree - fail loudly now, not at some later source/cd.
  if [[ ! -f scripts/rerankers/_common.sh ]]; then
    echo "[!!] Repo pull completed but scripts/rerankers/_common.sh is still missing in $(pwd) - aborting."
    exit 1
  fi
  export NEEDS_REGION_SYNC=1
fi

source scripts/rerankers/_common.sh

# See run_rader_14b.slurm for why: fp32-by-default plus allocator
# fragmentation OOM'd the 14B sibling model partway through a run. Applied
# here too as a precaution, even though 8B is much less likely to hit the
# same wall.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkenv -n reranker-vllm-reason-embed -f scripts/rerankers/envs/env_vllm_feas.yml -- \
  python scripts/run_rerankers.py --models reason-embed-qwen3-8b "$@"

echo "Done."
