#!/bin/bash
#SBATCH --job-name=reranker_rader_versions
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/%u/logs/reranker_rader_versions.log
#SBATCH --error=/scratch/%u/logs/reranker_rader_versions.error
# --chdir: slurmd otherwise tries to chdir into the *submit-time* working
# directory before the script runs - on a region whose home storage lacks
# this repo (e.g. msp3) that path doesn't exist, so it logs a noisy
# "couldn't chdir ... going to /tmp instead" error. Start in /tmp
# explicitly; the script cd's into the repo itself below.
#SBATCH --chdir=/tmp

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_rader_reranker_versions.slurm
#
# LIBRARY-VERSION arm of the rader-reranker-7b diagnostic. The production
# number is already reproduced by two independent stacks (HF+peft and
# vllm==0.26.0 agree to <=5.5e-4 nDCG on the full benchmark), so this is a
# falsification attempt, not the leading hypothesis. Both arms score the
# SAME 60-query subset and the SAME reference template (T0) as
# run_rader_reranker_diag.slurm, so all three jobs are directly comparable.
#
#   arm 1  vllm==0.27.0     - different serving stack version
#   arm 2  transformers 4.51/peft 0.14 on the pure HF path (no vllm at all)
#          - the one library failure that would explain the score: the
#          trained SEQ_CLS head ships under a non-standard adapter key, so a
#          peft that mapped it differently would leave a random head.
#
# set -e is deliberately NOT relied on to couple the arms: arm 1 failing to
# resolve its pins must not stop arm 2 from running, so each is guarded.
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


# Each arm's status and output are written INTO its results dir, not just to
# the SLURM log: that log lives on the compute node's own /scratch, which the
# canonical host cannot read, and the guarded `if` below means the job exits 0
# whether an arm worked or not (confirmed: a scancel'd run of this script was
# recorded by sacct as "COMPLETED 0:0"). Without ARM_STATUS/arm.log riding the
# results rsync back, a failed arm would be indistinguishable from a slow one.
run_arm() {
  local name="$1"; local outdir="$2"; shift 2
  mkdir -p "$outdir"
  echo "[~] ===== arm: $name ====="
  "$@" 2>&1 | tee "$outdir/arm.log"
  local rc=${PIPESTATUS[0]}
  if [[ $rc -eq 0 ]]; then
    echo "ok" > "$outdir/ARM_STATUS"
    echo "[+] arm $name finished"
  else
    echo "FAILED rc=$rc" > "$outdir/ARM_STATUS"
    echo "[!!] arm $name FAILED (rc=$rc) - continuing to the next arm"
  fi
}

set +e
run_arm "vllm-0.27.0" results/rader_reranker_diag_vllm027 \
  mkenv -n reranker-rader-vllm-alt -f scripts/rerankers/envs/env_vllm_feas_alt.yml -- \
  python experiments/rader-reranker-diag/sweep.py --backend vllm --arms T0 \
    --out results/rader_reranker_diag_vllm027 "$@"

run_arm "hf-transformers4.51-peft0.15.2" results/rader_reranker_diag_hfold \
  mkenv -n reranker-rader-hf-old -f scripts/rerankers/envs/env_rader_hf_old.yml -- \
  python experiments/rader-reranker-diag/sweep.py --backend hf --arms T0 \
    --out results/rader_reranker_diag_hfold "$@"

echo "Done."
