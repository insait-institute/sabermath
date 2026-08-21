#!/bin/bash
# Rerun the RaDeR bi-encoder nDCG benchmarks after the 2026-08-20 fixes:
# rader-3b/7b's committed results were produced through sentence-transformers'
# chat-template-wrapped preprocessing (invalid - see run_rerankers.py's legacy
# _build_rader_biencoder_processor) and rader-14b never had a full run at all.
# The launchers now run the vLLM backend (validated in
# scripts/test_vllm_feasibility.py, Spearman 0.9999-1.0 vs the fixed legacy
# reference).
#
# Run from the sabermath repo root:
#   bash scripts/rerankers/rerun_rader.sh
#
# WHY THE CEREMONY - the rsync resurrection trap: the region sync-back is
# `rsync -auz` (update mode, no --delete), so (a) a file deleted on the
# canonical host is RE-CREATED by any later job's exit-trap push from a node
# still holding a stale copy, and (b) a rerun job would silently RESUME from
# a stale node-local checkpoint the canonical pull can never remove. This
# script closes (a) by deleting on canonical and submitting the reruns
# IMMEDIATELY (the first fresh checkpoint write is newer than every stale
# copy, and -u protects it from then on); the run_rader_*.slurm launchers
# close (b) with their own node-side mtime-guarded purge blocks.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."  # sabermath repo root

CANONICAL_HOST="${SABERMATH_CANONICAL_HOST:-hala}"
CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/ivo_petrov/sabermath}"

# 1. Quiesce: an in-flight job's whole-results/ exit-trap push during the
# deletion window is the main resurrection vector - refuse to run while any
# reranker/timing job is active or queued.
if squeue -u "$USER" -h 2>/dev/null | grep -q .; then
  echo "[!!] You still have jobs in the queue:"
  squeue -u "$USER"
  echo "[!!] Their exit-time results sync could resurrect the files this"
  echo "[!!] script deletes. Wait for (or cancel) them, then rerun."
  exit 1
fi

# 2. Delete the poisoned artifacts ON THE CANONICAL TREE, explicitly over
# SSH - never assume the local host is canonical (multi-region homes).
# rader-14b entries are included defensively; no full run should exist.
echo "[~] Deleting pre-fix rader results/checkpoints on $CANONICAL_HOST:$CANONICAL_PATH ..."
ssh "$CANONICAL_HOST" "cd '$CANONICAL_PATH' && \
  rm -rf results/rerankers/.checkpoints/rader-3b \
         results/rerankers/.checkpoints/rader-7b \
         results/rerankers/.checkpoints/rader-14b && \
  rm -f  results/rerankers/rader-3b*.json \
         results/rerankers/rader-7b*.json \
         results/rerankers/rader-14b*.json && \
  echo '[+] Canonical tree cleaned.'"

# If this checkout is a separate copy of the canonical tree (same host,
# different mount), clean it too so a local sync can't push the files back.
rm -rf results/rerankers/.checkpoints/rader-3b \
       results/rerankers/.checkpoints/rader-7b \
       results/rerankers/.checkpoints/rader-14b
rm -f  results/rerankers/rader-3b*.json \
       results/rerankers/rader-7b*.json \
       results/rerankers/rader-14b*.json

# 3. Submit immediately - the resurrection window stays open until the first
# fresh checkpoint lands on canonical.
echo "[~] Submitting the three rader runs..."
sbatch scripts/rerankers/run_rader_3b.slurm "$@"
sbatch scripts/rerankers/run_rader_7b.slurm "$@"
sbatch scripts/rerankers/run_rader_14b.slurm "$@"

echo "[+] Done. After completion, regenerate the CIs:"
echo "      python scripts/rerankers/compute_confidence_intervals.py (see its --help)"
