#!/bin/bash
#SBATCH --job-name=reranker_reason_moderncolbert
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:2
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=1-0
#SBATCH --output=/scratch/%u/logs/reranker_reason_moderncolbert.log
#SBATCH --error=/scratch/%u/logs/reranker_reason_moderncolbert.error
# --chdir: slurmd otherwise tries to chdir into the *submit-time* working
# directory before the script runs - on a region whose home storage lacks
# this repo (e.g. msp3) that path doesn't exist, so it logs a noisy
# "couldn't chdir ... going to /tmp instead" error. Start in /tmp
# explicitly; the script cd's into the repo itself below.
#SBATCH --chdir=/tmp

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_reason_moderncolbert.slurm
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
# host - a presence-only check (e.g. "does _common.sh exist") isn't enough,
# since a node can have a stale-but-present copy from an earlier sync
# (confirmed the hard way: msp3 nodes kept a working-but-outdated copy
# across an actual code fix, silently skipping the self-heal that would
# have picked it up). rsync is cheap/idempotent when nothing has changed,
# so this is safe to run unconditionally on every non-canonical node.
export SABERMATH_CANONICAL_HOST="${SABERMATH_CANONICAL_HOST:-hala}"
export SABERMATH_CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/ivo_petrov/sabermath}"
if [[ "$(hostname)" != "$SABERMATH_CANONICAL_HOST" ]]; then
  echo "[~] Syncing latest code + any existing progress from $SABERMATH_CANONICAL_HOST (this node's home storage may not be the canonical copy)..."
  # -u (update): never overwrite a file that's newer on this side than on
  # the canonical side. Safe to pull results/.checkpoints unconditionally
  # with this flag - it only fills in gaps (e.g. progress from a run that
  # died elsewhere and got synced back) or takes the canonical version when
  # this node has nothing yet, but never regresses a checkpoint that's
  # already further along locally (e.g. resubmitting on the same node).
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

# Shares env_colbert.yml with gte-moderncolbert (same package list) but its
# own env name, so the two jobs never race building the same conda env.
mkenv -n reranker-colbert-reason -f scripts/rerankers/envs/env_colbert.yml -- \
  python scripts/run_rerankers.py --models reason-moderncolbert "$@"

echo "Done."
