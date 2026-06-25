#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$SCRIPT_DIR"
VENV="$SCRIPT_DIR/venv"
ARGS_FILE="$SCRIPT_DIR/models.txt"

EXTRA_ARGS=("$@")

echo "$SCRIPT_DIR"

if [ "${DEV_MODE:-0}" = "1" ]; then
    echo "DEV_MODE enabled: not redirecting logs and skipping venv check"
    PYTHON="python"
else
    LOG_DIR="$SCRIPT_DIR/logs"
    mkdir -p "$LOG_DIR"

    LOCAL_LOG="$LOG_DIR/eval_models.out"

    echo "Redirecting all output to: $LOCAL_LOG"
    exec > >(tee -a "$LOCAL_LOG") 2>&1

    if [ ! -x "$VENV/bin/python" ]; then
        echo "Virtual environment missing or broken: $VENV"
        echo ""
        echo "Create it with:"
        echo "   cd $PROJECT"
        echo "   python3 -m venv venv"
        echo "   source venv/bin/activate"
        echo "   pip install --upgrade pip"
        echo "   pip install -e '.[vllm]'"
        exit 1
    fi

    PYTHON="$VENV/bin/python"
fi

if [ ! -f "$ARGS_FILE" ]; then
    echo "Missing models file: $ARGS_FILE"
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-DEV_MODE/local}"
echo "Project: $PROJECT"
echo "Models file: $ARGS_FILE"
echo "Extra Python args: ${EXTRA_ARGS[*]:-}"
echo "Python: $PYTHON"

cd "$PROJECT"

export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

total=0
successes=0
failures=0

while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines
    [[ -z "${line//[[:space:]]/}" ]] && continue

    # Skip comments
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    total=$((total + 1))

    # Split the line like a shell command, preserving quoted strings.
    parsed_args="$(
        "$PYTHON" -c '
import shlex
import sys

for arg in shlex.split(sys.argv[1]):
    print(arg)
' "$line"
    )"

    mapfile -t line_args <<< "$parsed_args"

    echo ""
    echo "============================================================"
    echo "Run $total"
    echo "============================================================"
    echo "Running:"
    printf '  %q' "$PYTHON" "$SCRIPT_DIR/run_model.py" "${line_args[@]}" "${EXTRA_ARGS[@]}"
    echo
    echo ""

    if "$PYTHON" "$SCRIPT_DIR/run_model.py" \
        "${line_args[@]}" \
        "${EXTRA_ARGS[@]}"
    then
        echo ""
        echo "Run $total succeeded"
        successes=$((successes + 1))
    else
        status=$?
        echo ""
        echo "Run $total failed with exit code $status"
        echo "Continuing with next line..."
        failures=$((failures + 1))
    fi

done < "$ARGS_FILE"

echo ""
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Total runs: $total"
echo "Succeeded:  $successes"
echo "Failed:     $failures"

if [ "$failures" -gt 0 ]; then
    echo "Done, but some runs failed."
    exit 1
fi

echo "Done"