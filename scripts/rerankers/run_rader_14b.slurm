#!/bin/bash
#SBATCH --job-name=reranker_rader_14b
#SBATCH --nodes=1
#SBATCH --partition=non-reserved
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=1-0
#SBATCH --output=/scratch/%u/logs/reranker_rader_14b.log
#SBATCH --error=/scratch/%u/logs/reranker_rader_14b.error
# --chdir: slurmd otherwise tries to chdir into the *submit-time* working
# directory before the script runs - on a region whose home storage lacks
# this repo (e.g. msp3) that path doesn't exist, so it logs a noisy
# "couldn't chdir ... going to /tmp instead" error. Start in /tmp
# explicitly; the script cd's into the repo itself below.
#SBATCH --chdir=/tmp

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_rader_14b.slurm
#   sbatch scripts/rerankers/run_rader_14b.slurm --n 20   # smoke test first
#
# Plain bi-encoder embedding model (Qwen2.5-14B based reasoning-aware dense
# retriever - confirmed via its paper, not a cross-encoder) - runs through
# the generic SentenceTransformersProcessor path, single GPU, no tensor
# parallelism needed.
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

# One-time poisoned-state purge for the 2026-08-20 rader-14b rerun: every
# rader-14b result/checkpoint written BEFORE the sentence-transformers
# chat-template bug was found (see run_rerankers.py's legacy
# _build_rader_biencoder_processor) is invalid. The canonical copy gets
# cleaned once by scripts/rerankers/rerun_rader.sh, but nodes keep stale
# LOCAL copies the pull above can never remove (rsync -u updates, never
# deletes) - and a rerun here would silently RESUME from them. Guarded by
# mtime so it is idempotent across resubmissions: if ANY local rader-14b
# file is newer than the cutoff, it is this rerun's own fresh progress and
# everything is kept.
RADER_PURGE_CUTOFF="2026-08-20 12:00"
if ! { find results/rerankers/.checkpoints/rader-14b -type f -newermt "$RADER_PURGE_CUTOFF" 2>/dev/null; \
       find results/rerankers -maxdepth 1 -name 'rader-14b*.json' -newermt "$RADER_PURGE_CUTOFF" 2>/dev/null; } | grep -q .; then
  echo "[~] Purging pre-fix rader-14b results/checkpoints on this node (nothing newer than $RADER_PURGE_CUTOFF)."
  rm -rf "results/rerankers/.checkpoints/rader-14b"
  rm -f results/rerankers/rader-14b*.json
fi

# Confirmed the hard way: 14B params defaults to fp32 (~56GiB) via
# sentence-transformers unless told otherwise (see model_kwargs.torch_dtype
# in GENERIC_MODELS, run_rerankers.py) - OOM'd a 140GiB H200 partway through
# a run from PyTorch allocator fragmentation across many variable-shaped
# encode() calls. expandable_segments reduces that fragmentation (this is
# exactly what the OOM error itself recommended).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkenv -n reranker-vllm-rader-14b -f scripts/rerankers/envs/env_vllm_feas.yml -- \
  python scripts/run_rerankers.py --models rader-14b "$@"

echo "Done."
