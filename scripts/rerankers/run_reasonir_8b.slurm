#!/bin/bash
#SBATCH --job-name=reranker_reasonir_8b
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:2
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=1-0
#SBATCH --output=/scratch/%u/logs/reranker_reasonir_8b.log
#SBATCH --error=/scratch/%u/logs/reranker_reasonir_8b.error

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_reasonir_8b.slurm
set -e
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
export SABERMATH_CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/maria_drencheva/sabermath}"
if [[ "$(hostname)" != "$SABERMATH_CANONICAL_HOST" ]]; then
  echo "[~] Syncing latest code + any existing progress from $SABERMATH_CANONICAL_HOST (this node's home storage may not be the canonical copy)..."
  # -u (update): never overwrite a file that's newer on this side than on
  # the canonical side. Safe to pull results/.checkpoints unconditionally
  # with this flag - it only fills in gaps (e.g. progress from a run that
  # died elsewhere and got synced back) or takes the canonical version when
  # this node has nothing yet, but never regresses a checkpoint that's
  # already further along locally (e.g. resubmitting on the same node).
  rsync -auz --exclude=".git" \
    "$SABERMATH_CANONICAL_HOST:$SABERMATH_CANONICAL_PATH/" "./"
  export NEEDS_REGION_SYNC=1
fi

source scripts/rerankers/_common.sh

mkenv -n reranker-reasonir -f scripts/rerankers/envs/env_reasonir.yml -- \
  python scripts/run_rerankers.py --models reasonir-8b "$@"

echo "Done."
