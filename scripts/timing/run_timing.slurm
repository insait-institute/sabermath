#!/bin/bash
#SBATCH --job-name=timing
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/timing_%j.log
#SBATCH --error=/scratch/%u/logs/timing_%j.error

# IMPORTANT - run this once FIRST, before submitting any --model job below
# (cheap enough to just run directly, no sbatch needed - it only reads the
# queries dataset, no model loading):
#   python scripts/measure_query_time.py --generate-sample --n-queries 50 --seed 42
# This samples the shared query indices ONCE, from every query index
# EXCEPT the 29 known pya0/approach0 segfault triggers (excluded for every
# model, not just approach0, so the same 50 queries are valid input for
# all of them), and writes results/timing/query_sample.json. Every --model
# job below then reads that same file rather than sampling independently -
# critical, since two independent rng.sample() calls with the same seed
# but different population sizes/contents do NOT produce the same actual
# indices. --n-queries/--seed passed to a --model job below are ignored;
# they only apply to --generate-sample.
#
# Run from the sabermath repo root:
#   sbatch scripts/timing/run_timing.slurm --model rank1-32b
#   sbatch --job-name=timing_bm25 scripts/timing/run_timing.slurm --model bm25
#
# One GPU node per model (point 4 of the methodology - every model runs on
# a single H200, including the classical/closed-API ones that don't
# actually need the GPU, for environment consistency). --model is required
# and everything after it is forwarded straight to measure_query_time.py.
#
# Env selection: each model reuses whichever conda env file its production
# reranker run already uses (scripts/rerankers/envs/*.yml) wherever one
# exists, or one of the two new envs under scripts/timing/envs/ for the
# classical/closed-API categories that have no existing env. Every model
# gets its OWN env *name* even when several share the same env *file*
# (e.g. every scripts/rerankers/envs/env_st_biencoders.yml-based model
# below) - confirmed the hard way, on the production reranker rollout,
# that two jobs racing to build/refresh an identically-named env on the
# same node corrupts one of the builds (a stdlib _struct ModuleNotFoundError
# from concurrent linking). This is 2 hours per job - not the 30+ hour
# rank1-32b/rader-14b production runs - so collisions are just as possible
# here if several of these are submitted back-to-back.
set -e
cd "${SABERMATH_REPO:-$HOME/sabermath}"

export SABERMATH_CANONICAL_HOST="${SABERMATH_CANONICAL_HOST:-hala}"
export SABERMATH_CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/maria_drencheva/sabermath}"
if [[ "$(hostname)" != "$SABERMATH_CANONICAL_HOST" ]]; then
  echo "[~] Syncing latest code + any existing progress from $SABERMATH_CANONICAL_HOST (this node's home storage may not be the canonical copy)..."
  rsync -auz --exclude=".git" \
    "$SABERMATH_CANONICAL_HOST:$SABERMATH_CANONICAL_PATH/" "./"
  export NEEDS_REGION_SYNC=1
fi

source scripts/rerankers/_common.sh

# Parse --model out of "$@" without consuming the rest - everything
# (including --model itself) still gets forwarded to the python script
# below, since argparse there also requires --model.
MODEL=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--model" ]]; then
    MODEL="${args[$((i + 1))]}"
    break
  fi
done

if [[ -z "$MODEL" ]]; then
  echo "[!!] --model <key> is required, e.g.: sbatch scripts/timing/run_timing.slurm --model rank1-32b"
  exit 1
fi

declare -A MODEL_ENV_YML=(
  [rank1-32b]="scripts/rerankers/envs/env_vllm.yml"
  [qwen3-reranker-8b]="scripts/rerankers/envs/env_qwen3_reranker.yml"
  [splade-code-8b]="scripts/rerankers/envs/env_splade.yml"
  [gte-moderncolbert]="scripts/rerankers/envs/env_colbert.yml"
  [reason-moderncolbert]="scripts/rerankers/envs/env_colbert.yml"
  [reasonir-8b]="scripts/rerankers/envs/env_reasonir.yml"
  [reason-embed-qwen3-8b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [diver-retriever-4b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [rader-14b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [qwen3-embedding-8b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [qwen3-embedding-4b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [qwen3-embedding-0.6b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [harrier-oss-v1-270m]="scripts/rerankers/envs/env_st_biencoders.yml"
  [harrier-oss-v1-0.6b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [harrier-oss-v1-27b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [bge-m3]="scripts/rerankers/envs/env_st_biencoders.yml"
  [llama-embed-nemotron-8b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [kalm-embedding-gemma3-12b-2511]="scripts/rerankers/envs/env_st_biencoders.yml"
  [roberta-base]="scripts/rerankers/envs/env_st_biencoders.yml"
  [bert-base-uncased]="scripts/rerankers/envs/env_st_biencoders.yml"
  [embeddinggemma-300m]="scripts/rerankers/envs/env_st_biencoders.yml"
  [multilingual-e5-large]="scripts/rerankers/envs/env_st_biencoders.yml"
  [jina-embeddings-v5-text-nano]="scripts/rerankers/envs/env_st_biencoders.yml"
  [jina-embeddings-v5-text-small]="scripts/rerankers/envs/env_st_biencoders.yml"
  [octen-embedding-4b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [octen-embedding-8b]="scripts/rerankers/envs/env_st_biencoders.yml"
  [gemini-embedding-001]="scripts/timing/envs/env_closed_api.yml"
  [gemini-embedding-2]="scripts/timing/envs/env_closed_api.yml"
  [text-embedding-3-small]="scripts/timing/envs/env_closed_api.yml"
  [text-embedding-3-large]="scripts/timing/envs/env_closed_api.yml"
  [bm25]="scripts/timing/envs/env_classical.yml"
  [tf-idf]="scripts/timing/envs/env_classical.yml"
  [jaccard]="scripts/timing/envs/env_classical.yml"
  [approach0]="scripts/timing/envs/env_classical.yml"
)

ENV_YML="${MODEL_ENV_YML[$MODEL]:-}"
if [[ -z "$ENV_YML" ]]; then
  echo "[!!] Unknown model key '$MODEL' - not in MODEL_ENV_YML. Known keys: ${!MODEL_ENV_YML[*]}"
  exit 1
fi

# Sanitize (micromamba env names shouldn't contain dots, e.g.
# qwen3-embedding-0.6b) and always prefix so this never collides with a
# production reranker env of the same underlying model name.
ENV_NAME="timing-${MODEL//./-}"

mkenv -n "$ENV_NAME" -f "$ENV_YML" -- \
  python scripts/measure_query_time.py "$@"

echo "Done."
