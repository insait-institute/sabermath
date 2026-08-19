#!/bin/bash
# Sourced by scripts/rerankers/run_*.slurm - not meant to be run directly.
# Assumes the caller has already `cd`-ed into the sabermath repo root. Every
# run_*.slurm here does that with an absolute path (`cd
# "${SABERMATH_REPO:-$HOME/sabermath}"`), NOT `cd "$SLURM_SUBMIT_DIR"` - that
# variable isn't reliably populated in this cluster's batch job environment
# (confirmed: it silently left the shell in the wrong directory, so the
# later `source scripts/rerankers/_common.sh` failed with "No such file or
# directory" - and since that happened *before* this file's own `set -e`
# below ever got a chance to run, the failure didn't even abort the job).
# Every run_*.slurm now also has its own `set -e` immediately before that
# `cd`, so a failure at that step aborts loudly instead of silently falling
# through to the final `echo "Done."`.
set -e

# The env_*.yml postinstall scripts need an absolute path to this repo
# (mkenv runs postinstall with cwd set to the *conda env's own directory*,
# not the caller's cwd - confirmed the hard way: `pip install -e .` there
# tried to install the env directory itself as a package and failed with
# "does not appear to be a Python project"). Exporting it here, from the
# cwd we already `cd`-ed into, makes it correct regardless of region/host.
export SABERMATH_REPO="$(pwd)"

# This cluster spans multiple regions (zone-sof1/zone-gcp-eu1 in the EU,
# zone-msp3 in the US) with SEPARATE, non-synced home-directory storage per
# region - a job landing on a region whose copy of this repo is stale would
# otherwise fail outright (confirmed: msp3's copy was a months-old stub).
# The calling run_*.slurm already self-heals this by pulling a fresh copy
# (code AND any existing results/checkpoints, via -u/update mode - see that
# block) from the canonical EU host when it detects it isn't already on it,
# and sets NEEDS_REGION_SYNC=1 when it does.
#
# Two things push progress back to that canonical copy so it's never
# stranded on a region-local filesystem that compute_confidence_intervals.py
# (run from the canonical host) can't see:
#   1. scripts/run_rerankers.py's on_progress callback, every
#      --progress-every queries (default 10) - the main defense, since it
#      doesn't depend on the process getting a chance to clean up at all.
#   2. sync_back_to_canonical() here, armed via the EXIT trap below, as a
#      last-call safety net for whatever happened since that last periodic
#      push (only fires on a normal exit or a caught signal like SIGTERM,
#      same limitation as any exit-handler - a bare SIGKILL skips it, which
#      is exactly why (1) exists too).
SABERMATH_CANONICAL_HOST="${SABERMATH_CANONICAL_HOST:-hala}"
SABERMATH_CANONICAL_PATH="${SABERMATH_CANONICAL_PATH:-/home/maria_drencheva/sabermath}"

sync_back_to_canonical() {
  [[ "${NEEDS_REGION_SYNC:-0}" == "1" ]] || return 0
  echo "[~] Syncing results back to $SABERMATH_CANONICAL_HOST (this node's home storage isn't the canonical copy)..."
  if rsync -auz results/ "$SABERMATH_CANONICAL_HOST:$SABERMATH_CANONICAL_PATH/results/"; then
    echo "[+] Synced."
  else
    echo "[!!] Sync-back to $SABERMATH_CANONICAL_HOST failed - results remain only on $(hostname):$(pwd)/results until manually rsynced."
  fi
}

if [[ "${NEEDS_REGION_SYNC:-0}" == "1" ]]; then
  trap sync_back_to_canonical EXIT
fi

mkdir -p "/scratch/$USER/logs"

# Model weights/datasets cache on scratch (fast local disk, keeps ~60GB+
# reranker checkpoints off the /home quota) - results/checkpoints still
# write to the repo under $HOME/sabermath (durable, not scratch).
export HF_HOME="/scratch/$USER/hf"

if [[ -f .hftok ]]; then
  export HF_TOKEN="$(cat .hftok)"
elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
  # Falls back to the token from a prior `huggingface-cli login` on this
  # machine, so nothing needs to be pasted into any file at all.
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
else
  echo "WARNING: no HF token found (.hftok or ~/.cache/huggingface/token) - gated HF models may fail to download. See scripts/rerankers/README.md."
fi

# Same .<name>tok-file pattern as .hftok above, for the timing harness's
# closed-API models (scripts/measure_query_time.py). Neither is needed by
# anything else in this repo, so a missing file here is silent (no
# warning) - only measure_query_time.py's own gemini-embedding-*/
# text-embedding-3-* jobs actually need these.
if [[ -f .openroutertok ]]; then
  export OPENROUTER_API_KEY="$(cat .openroutertok)"
fi
if [[ -f .geminitok ]]; then
  export GEMINI_API_KEY="$(cat .geminitok)"
fi

num_gpus=$(echo "$SLURM_GPUS_ON_NODE" | grep -o '[0-9]*' | head -1)
num_gpus="${num_gpus:-1}"
export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((num_gpus - 1)))"
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Job: ${SLURM_JOB_NAME:-(none)} (${SLURM_JOB_ID:-local}) | GPUs: $num_gpus | CWD: $(pwd)"
