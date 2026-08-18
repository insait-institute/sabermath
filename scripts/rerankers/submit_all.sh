#!/bin/bash
# Submits all 7 reranker sweep jobs as independent SLURM jobs (one per
# model), so they run in parallel on separate GPU allocations and can be
# resubmitted/monitored/cancelled individually. Run from the sabermath repo
# root: bash scripts/rerankers/submit_all.sh
#
# Any extra args are forwarded to every job, e.g.:
#   bash scripts/rerankers/submit_all.sh --n 20          # smoke test all 7
#   bash scripts/rerankers/submit_all.sh --task statement-full
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"  # scripts/rerankers/

for script in \
  run_rank1_32b.slurm \
  run_qwen3_embedding_8b.slurm \
  run_qwen3_reranker_8b.slurm \
  run_gte_moderncolbert.slurm \
  run_reason_moderncolbert.slurm \
  run_reasonir_8b.slurm \
  run_splade_code_8b.slurm
do
  echo "Submitting $script ..."
  sbatch "$script" "$@"
done
