#!/bin/bash
#SBATCH --job-name=reranker_diver_grouprank_32b
#SBATCH --nodes=1
#SBATCH --partition=non-reserved
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=3-0
#SBATCH --output=/scratch/%u/logs/reranker_diver_grouprank_32b.log
#SBATCH --error=/scratch/%u/logs/reranker_diver_grouprank_32b.error
# --chdir: slurmd otherwise tries to chdir into the *submit-time* working
# directory before the script runs - on a region whose home storage lacks
# this repo (e.g. msp3) that path doesn't exist, so it logs a noisy
# "couldn't chdir ... going to /tmp instead" error. Start in /tmp
# explicitly; the script cd's into the repo itself below.
#SBATCH --chdir=/tmp

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_diver_grouprank_32b.slurm
#   sbatch scripts/rerankers/run_diver_grouprank_32b.slurm --n 20   # smoke test first
#   sbatch scripts/rerankers/run_diver_grouprank_32b.slurm --task statement-full
#
# Diver's groupwise generative reranker (AQ-MedAI/Diver-GroupRank-32B),
# served via vLLM - see GroupRankProcessor. Per query it scores 150
# candidates as 8 groups of 20 x repeat_num=2 reshuffled epochs = 16
# generations (with up to 8000 reasoning tokens each), so this is the
# slowest new model - hence the rank1-32b-style 3-day limit. Resubmitting
# the same command resumes from the per-query checkpoint. 32B in bf16 fits a
# single H200; pass --tensor-parallel-size 2 (and bump --gpus) only if you
# have first re-validated scores on a --n 20 subset, given rank1-32b's
# history of tensor parallelism silently corrupting scores in this vLLM
# version.
set -e

# On a region whose home storage has NEVER held this repo (msp3 for this
# account), $HOME/sabermath doesn't exist at all, so a bare `cd` died here
# before the rsync self-heal below ever got a chance to populate it
# (confirmed: job 716300 on msp3-1 failed exactly this way). Create the
# directory first - a no-op everywhere the repo already exists - and let
# the self-heal pull fill it.
mkdir -p "${SABERMATH_REPO:-$HOME/sabermath}"
cd "${SABERMATH_REPO:-$HOME/sabermath}"

# Self-healing: some cluster regions have separate, unsynced home-directory
# storage from the canonical EU copy (see scripts/rerankers/_common.sh).
# Always pull a fresh copy over SSH unless we're literally on the canonical
# host - a presence-only check isn't enough, since a node can have a
# stale-but-present copy from an earlier sync. rsync is cheap/idempotent
# when nothing has changed, so this is safe to run unconditionally on every
# non-canonical node. -u (update mode) never regresses results/checkpoints
# that are further along locally.
export SABERMATH_CANONICAL_HOST="${SABERMATH_CANONICAL_HOST:-hala}"
export SABERMATH_CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/ivo_petrov/sabermath}"
if [[ "$(hostname)" != "$SABERMATH_CANONICAL_HOST" ]]; then
  echo "[~] Syncing latest code + any existing progress from $SABERMATH_CANONICAL_HOST (this node's home storage may not be the canonical copy)..."
  rsync -auz --exclude=".git" \
    "$SABERMATH_CANONICAL_HOST:$SABERMATH_CANONICAL_PATH/" "./" || {
    echo "[!!] Pulling the repo from $SABERMATH_CANONICAL_HOST:$SABERMATH_CANONICAL_PATH failed."
    echo "[!!] This node's home storage has no usable copy, so the job cannot continue."
    echo "[!!] Check SSH reachability/keys from $(hostname) to $SABERMATH_CANONICAL_HOST,"
    echo "[!!] and that the canonical path exists there (override via SABERMATH_CANONICAL_HOST/_PATH)."
    exit 1
  }
  # Belt-and-braces: even a "successful" rsync against a wrong/empty source
  # leaves an unusable tree - fail loudly now, not at some later source/cd.
  if [[ ! -f scripts/rerankers/_common.sh ]]; then
    echo "[!!] Repo pull completed but scripts/rerankers/_common.sh is still missing in $(pwd) - aborting."
    exit 1
  fi
  export NEEDS_REGION_SYNC=1
fi

source scripts/rerankers/_common.sh

mkenv -n reranker-grouprank -f scripts/rerankers/envs/env_vllm.yml -- \
  python scripts/run_rerankers.py --models diver-grouprank-32b "$@"

echo "Done."
