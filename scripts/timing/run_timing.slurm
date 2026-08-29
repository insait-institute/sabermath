#!/bin/bash
#SBATCH --job-name=timing
#SBATCH --nodes=1
#SBATCH --partition=non-reserved
#SBATCH --constraint=type-gpu
#SBATCH --gpus=h200:1
#SBATCH --mem=120G
#SBATCH --export=ALL
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/timing_%j.log
#SBATCH --error=/scratch/%u/logs/timing_%j.error
# --chdir: slurmd otherwise tries to chdir into the *submit-time* working
# directory before the script runs - on a region whose home storage lacks
# this repo (e.g. msp3, which the non-reserved partition includes) that
# path doesn't exist, so it logs a noisy "couldn't chdir ... going to /tmp
# instead" error. Start in /tmp explicitly; the script cd's into the repo
# itself below.
#SBATCH --chdir=/tmp

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
# Env selection: each model reuses its PRODUCTION env file. Since the
# 2026-08-20 vLLM-default rollout that is env_vllm_feas.yml for most
# neural models (jina-v5-text-nano stays on env_st_biencoders.yml -
# vLLM-infeasible; splade/colbert keep their own envs), or one of the two
# envs under scripts/timing/envs/ for the classical/closed-API categories. Every model
# gets its OWN env *name* even when several share the same env *file*
# (e.g. every scripts/rerankers/envs/env_st_biencoders.yml-based model
# below) - confirmed the hard way, on the production reranker rollout,
# that two jobs racing to build/refresh an identically-named env on the
# same node corrupts one of the builds (a stdlib _struct ModuleNotFoundError
# from concurrent linking). This is 2 hours per job - not the 30+ hour
# rank1-32b/rader-14b production runs - so collisions are just as possible
# here if several of these are submitted back-to-back.
set -e

# On a region whose home storage has NEVER held this repo (msp3 for this
# account), $HOME/sabermath doesn't exist at all, so a bare `cd` would die
# here before the rsync self-heal below ever got a chance to populate it
# (confirmed on the production reranker launchers: job 716300 on msp3-1
# failed exactly this way). Create the directory first - a no-op everywhere
# the repo already exists - and let the self-heal pull fill it.
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

# Uniform CPU-parallelism envelope (methodology: 16 parallel lanes for every
# model, GPU batch or CPU threads alike). This is the REAL parallelism bound
# for the classical baselines - their score_batch_size=16 slicing is protocol
# uniformity, but BLAS threading is where bm25/tf-idf actually parallelize.
# Applied to every timing job, GPU models included, so CPU-side preprocessing
# is bounded identically everywhere.
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16

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
  [rank1-7b]="scripts/rerankers/envs/env_vllm.yml"
  [rank1-0.5b]="scripts/rerankers/envs/env_vllm.yml"
  [diver-grouprank-32b]="scripts/rerankers/envs/env_vllm.yml"
  [qwen3-reranker-8b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [qwen3-reranker-4b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [qwen3-reranker-0.6b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [splade-code-8b]="scripts/rerankers/envs/env_splade.yml"
  [splade-code-0.6b]="scripts/rerankers/envs/env_splade.yml"
  [gte-moderncolbert]="scripts/rerankers/envs/env_colbert.yml"
  [reason-moderncolbert]="scripts/rerankers/envs/env_colbert.yml"
  [reasonir-8b]="scripts/rerankers/envs/env_reasonir.yml"
  [reason-embed-qwen3-8b]="scripts/rerankers/envs/env_vllm_feas.yml"
  # Added 2026-08-28 alongside their measure_query_time.py ALL_BUILDERS
  # entries. Both take the env their own production launcher in
  # scripts/rerankers/ uses (run_reason_embed_llama_3.1_8b.slurm and
  # run_retro_star_32b.slurm both mkenv against env_vllm_feas.yml), which is
  # the rule the rest of this table follows: time the model in the same
  # environment the benchmark runs it in.
  [reason-embed-llama-3.1-8b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [retro-star-32b]="scripts/rerankers/envs/env_vllm_feas.yml"
  # Composed rewrite systems - same env as their production launcher
  # (run_reason_rewriter_reason_embed_8b.slurm uses env_vllm_feas.yml).
  [reason-rewriter-reason-embed-8b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [reason-rewriter-reason-embed-llama-3.1-8b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [retro-star-32b-rewritten]="scripts/rerankers/envs/env_vllm_feas.yml"
  [retro-star-32b-rewritten-uncached]="scripts/rerankers/envs/env_vllm_feas.yml"
  [diver-retriever-4b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [diver-retriever-0.6b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [rader-3b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [rader-7b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [rader-14b]="scripts/rerankers/envs/env_vllm_feas.yml"
  # SentenceTransformers production path (bidirectional remote code, no
  # validated vLLM recipe yet) - same env as its reranker launcher; its
  # remote code needs the exact transformers==4.51.x pin in there.
  [inf-retriever-v1-pro]="scripts/rerankers/envs/env_inf_retriever.yml"
  # Composed system (aligner is a standard Qwen2.5 chat model, same env).
  [inf-x-retriever]="scripts/rerankers/envs/env_inf_retriever.yml"
  # env_vllm_feas installs peft, which the one-off LoRA merge needs.
  [rader-reranker-7b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [qwen3-embedding-8b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [qwen3-embedding-4b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [qwen3-embedding-0.6b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [harrier-oss-v1-270m]="scripts/rerankers/envs/env_vllm_feas.yml"
  [harrier-oss-v1-0.6b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [harrier-oss-v1-27b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [bge-m3]="scripts/rerankers/envs/env_vllm_feas.yml"
  [llama-embed-nemotron-8b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [kalm-embedding-gemma3-12b-2511]="scripts/rerankers/envs/env_vllm_feas.yml"
  [roberta-base]="scripts/rerankers/envs/env_vllm_feas.yml"
  [bert-base-uncased]="scripts/rerankers/envs/env_vllm_feas.yml"
  [embeddinggemma-300m]="scripts/rerankers/envs/env_vllm_feas.yml"
  [multilingual-e5-large]="scripts/rerankers/envs/env_vllm_feas.yml"
  [jina-embeddings-v5-text-nano]="scripts/rerankers/envs/env_st_biencoders.yml"
  [jina-embeddings-v5-text-small]="scripts/rerankers/envs/env_vllm_feas.yml"
  [octen-embedding-4b]="scripts/rerankers/envs/env_vllm_feas.yml"
  [octen-embedding-8b]="scripts/rerankers/envs/env_vllm_feas.yml"
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
