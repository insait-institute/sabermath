#!/bin/bash
#SBATCH --job-name=reranker_rader_diag
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/reranker_rader_diag.log
#SBATCH --error=/scratch/%u/logs/reranker_rader_diag.error
# --chdir: slurmd otherwise tries to chdir into the *submit-time* working
# directory before the script runs - on a region whose home storage lacks
# this repo (e.g. msp3) that path doesn't exist, so it logs a noisy
# "couldn't chdir ... going to /tmp instead" error. Start in /tmp
# explicitly; the script cd's into the repo itself below.
#SBATCH --chdir=/tmp

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_rader_reranker_diag.slurm
#   sbatch scripts/rerankers/run_rader_reranker_diag.slurm --n 20   # smoke test
#
# INPUT-FORMAT diagnostic for rader-reranker-7b, which scores ~0.15 nDCG@10
# below the RaDeR bi-encoders trained on the same data. Sweeps 7 input
# templates on a fixed 60-query subset, because the reranker's Tevatron
# training format (empty title slot + a data-side "Query: " prefix) is NOT
# what RaDeR's own rerank.py rebuilds at inference. Full reasoning in
# experiments/rader-reranker-diag/sweep.py.
#
# This does NOT write to results/rerankers - it lands in
# results/rader_reranker_diag, so no board number can be touched.
set -e

# On a region whose home storage has NEVER held this repo (msp3 for this
# account), $HOME/saberstable doesn't exist at all, so a bare `cd` died here
# before the rsync self-heal below ever got a chance to populate it
# (confirmed: job 716300 on msp3-1 failed exactly this way). Create the
# directory first - a no-op everywhere the repo already exists - and let
# the self-heal pull fill it.
mkdir -p "${SABERMATH_REPO:-$HOME/saberstable}"
cd "${SABERMATH_REPO:-$HOME/saberstable}"

# Self-healing: some cluster regions have separate, unsynced home-directory
# storage from the canonical EU copy (see scripts/rerankers/_common.sh).
# Always pull a fresh copy over SSH unless we're literally on the canonical
# host - a presence-only check isn't enough, since a node can have a
# stale-but-present copy from an earlier sync. rsync is cheap/idempotent
# when nothing has changed, so this is safe to run unconditionally on every
# non-canonical node. -u (update mode) never regresses results/checkpoints
# that are further along locally.
export SABERMATH_CANONICAL_HOST="${SABERMATH_CANONICAL_HOST:-hala}"
export SABERMATH_CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/dimitadi/saberstable}"
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
# fragmentation OOM'd the 14B sibling model partway through a run.
# expandable_segments reduces that fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Template/backend diagnostic for rader-reranker-7b's low SaberMath score -
# see experiments/rader-reranker-diag/sweep.py for the full motivation.
# Two sequential processes, NOT one: a bf16 7B HF model and a vLLM engine
# cannot share one GPU at any usable gpu_memory_utilization, and a separate
# process is a cleaner guarantee of that than trying to free the first.
#
# The vllm arm is the backend that produced the published number, so it runs
# first and carries all 7 template arms; the hf arm re-scores the reference
# template only, as an independent-stack cross-check on the same 60 queries.
mkenv -n reranker-vllm-rader-reranker -f scripts/rerankers/envs/env_vllm_feas.yml -- \
  python experiments/rader-reranker-diag/sweep.py --backend vllm "$@"

mkenv -n reranker-vllm-rader-reranker -f scripts/rerankers/envs/env_vllm_feas.yml -- \
  python experiments/rader-reranker-diag/sweep.py --backend hf --arms T0 T2 "$@"

# --arms lets a follow-up rerun only the arms that are new, e.g. the App.G
# spacing 2x2 added 2026-08-28 (T0 is included as an in-job control so a
# rerun is comparable to itself, not just to the earlier job):
#   sbatch scripts/rerankers/run_rader_reranker_diag.slurm --arms T0 T7 T8 T9

echo "Done."
