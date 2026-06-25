#!/bin/bash

if [[ $# -lt 3 ]]; then
  echo "Usage: make_tags.sh <parallel_count> <vllm_port> <huggingface_path>"
  exit 1
fi

VLLM_PORT="$2"
VLLM_LOG="vllm_server.log"

vllm serve openai/gpt-oss-120b --port $VLLM_PORT --tensor-parallel-size $1 > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "[INFO] Started vLLM server with PID $VLLM_PID"

cleanup() {
    echo "[INFO] Cleaning up..."
    if ps -p "$VLLM_PID" > /dev/null 2>&1; then
        echo "[INFO] Stopping vLLM server (PID $VLLM_PID)"
        kill "$VLLM_PID" || true
        sleep 5 || true
        if ps -p "$VLLM_PID" > /dev/null 2>&1; then
            echo "[WARN] vLLM still alive, sending SIGKILL"
            kill -9 "$VLLM_PID" || true
        fi
    fi
}
trap cleanup EXIT

echo "[INFO] Waiting for vLLM API to startup..."

MAX_WAIT=2400
SLEEP_INTERVAL=3
ELAPSED=0

while ! curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; do
    if ! ps -p "$VLLM_PID" > /dev/null 2>&1; then
        echo "[ERROR] vLLM process died while starting. Check $VLLM_LOG."
        exit 1
    fi

    if (( ELAPSED >= MAX_WAIT )); then
        echo "[ERROR] Timed out waiting for vLLM API to start."
        exit 1
    fi

    sleep "$SLEEP_INTERVAL"
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
    echo "[INFO] Still waiting... (${ELAPSED}s)"
done

echo "[INFO] vLLM API is up."

echo "[INFO] Starting Tag Maker on dataset $3..."
python ./tags/make_tags.py $3 --api-url http://localhost:$VLLM_PORT/v1/ --reasoning high --tree ../data/tree.json
echo "[INFO] Tag agent finished"
