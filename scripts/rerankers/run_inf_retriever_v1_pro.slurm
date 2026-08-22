#!/bin/bash
#SBATCH --job-name=reranker_inf_retriever_v1_pro
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --mail-type=ALL
#SBATCH --time=1-0
#SBATCH --output=/scratch/%u/logs/reranker_inf_retriever_v1_pro.log
#SBATCH --error=/scratch/%u/logs/reranker_inf_retriever_v1_pro.error
# --chdir: slurmd otherwise tries to chdir into the *submit-time* working
# directory before the script runs - on a region whose home storage lacks
# this repo (e.g. msp3) that path doesn't exist, so it logs a noisy
# "couldn't chdir ... going to /tmp instead" error. Start in /tmp
# explicitly; the script cd's into the repo itself below.
#SBATCH --chdir=/tmp

# Run from the sabermath repo root:
#   sbatch scripts/rerankers/run_inf_retriever_v1_pro.slurm
#   sbatch scripts/rerankers/run_inf_retriever_v1_pro.slurm --n 20   # smoke test first
#
# INF-Retriever-v1-Pro (infly/inf-retriever-v1-pro): 7B gte-Qwen2-lineage
# bi-encoder, lasttoken pooling over an appended EOS. Runs through the
# SentenceTransformers path ON PURPOSE - its bundled modeling_qwen.py embeds
# with BIDIRECTIONAL attention, which vLLM's stock (causal) Qwen2 impl does
# not reproduce without a still-unvalidated is_causal override; see
# _build_inf_retriever_processor in run_rerankers.py and its candidate entry
# in scripts/test_vllm_feasibility.py. Hence its OWN env
# (env_inf_retriever.yml: transformers exact-pinned to 4.51.0 because the
# remote modeling code crashes on transformers 5.x's Qwen2Config - see that
# yml's header - plus the sentence-transformers==5.7.0 pin, load-bearing
# because this tokenizer ships a chat template, the exact ST>=5.7
# silent-corruption vector documented for the RaDeR family) rather than
# env_vllm_feas.yml. Single GPU, no tensor parallelism.
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

# fp16 7B on the ST path with full-length (up to 32k-token) documents:
# variable-length batches fragment the allocator exactly the way that OOM'd
# rader-14b's ST runs partway through - same precaution as the other
# ST-path launchers.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Env name carries the transformers pin (tf4513) because mkenv reuses an
# existing env name without re-checking the yml - bump it whenever
# env_inf_retriever.yml's pins change (stale names listed in that header).
mkenv -n reranker-inf-retriever-v1-pro-tf4513 -f scripts/rerankers/envs/env_inf_retriever.yml -- \
  python scripts/run_rerankers.py --models inf-retriever-v1-pro "$@"

echo "Done."
