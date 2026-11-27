#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="./venv/bin/python"

get_env() {
  "$PYTHON" -c "from dotenv import dotenv_values; import os,sys; print(os.environ.get(sys.argv[1]) or dotenv_values('.env').get(sys.argv[1]) or sys.argv[2])" "$1" "$2"
}

RETRIEVER_TYPE="${1:-base}"
case "$RETRIEVER_TYPE" in
  base) MODULE="agents.base_retriever"; NAME="base_retriever" ;;
  baseline_reranker) MODULE="agents.baseline_reranker"; NAME="baseline_reranker" ;;
  *) echo "Usage: $0 [base|baseline_reranker]"; exit 2 ;;
esac

export PYTHONPATH="./src"
export RETRIEVER_PORT="${RETRIEVER_PORT:-$(get_env RETRIEVER_PORT 8102)}"
export RETRIEVER_URL="${RETRIEVER_URL:-$(get_env RETRIEVER_URL "http://localhost:$RETRIEVER_PORT")}"
export SUMMARIZER_URL="${SUMMARIZER_URL:-$(get_env SUMMARIZER_URL "http://localhost:$(get_env SUMMARIZER_PORT 8104)")}"
export SUMMARIZER_PORT="${SUMMARIZER_PORT:-$(get_env SUMMARIZER_PORT 8104)}"
export ORCHESTRATOR_PORT="${ORCHESTRATOR_PORT:-$(get_env ORCHESTRATOR_PORT 8101)}"
export SUB_AGENT_URLS="$RETRIEVER_URL,$SUMMARIZER_URL"
LOG_DIR="./var/logs"
mkdir -p "$LOG_DIR"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES='' "$PYTHON" -m "$MODULE" > /dev/null 2>&1 &
PIDS+=("$!")
"$PYTHON" -m agents.summarizer > /dev/null 2>&1 &
PIDS+=("$!")
"$PYTHON" -m main > /dev/null 2>&1 &
PIDS+=("$!")

echo "Started $NAME with orchestrator and summarizer"
for port in "$RETRIEVER_PORT" "${SUMMARIZER_PORT:-8104}" "${ORCHESTRATOR_PORT:-8101}"; do
  ready=0
  for _ in {1..12}; do
    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
      echo "  localhost:$port OK"
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" == 0 ]]; then
    echo "  localhost:$port NOT READY"
  fi
done

echo "Press Ctrl+C to stop all agents."
wait -n "${PIDS[@]}"
