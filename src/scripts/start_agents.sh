#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON="./venv/bin/python"
export PYTHONPATH="./src"
LOG_DIR="./var/logs"
mkdir -p "$LOG_DIR"

echo "Starting A2A agents..."

# Start sub-agents first
$PYTHON -m agents.retriever_embedding > /dev/null 2>&1 &
PID_EMB=$!

$PYTHON -m agents.retriever_bm25 > /dev/null 2>&1 &
PID_BM25=$!

$PYTHON -m agents.summarizer > /dev/null 2>&1 &
PID_SUM=$!

# Start orchestrator
$PYTHON -m main > /dev/null 2>&1 &
PID_ORCH=$!

echo "Agents started:"
echo "  Retriever-Embedding PID=$PID_EMB  port=8102  log=$LOG_DIR/retriever_embedding.log"
echo "  Retriever-BM25      PID=$PID_BM25  port=8103  log=$LOG_DIR/retriever_bm25.log"
echo "  Summarizer          PID=$PID_SUM   port=8104  log=$LOG_DIR/summarizer.log"
echo "  Orchestrator        PID=$PID_ORCH  port=8101  log=$LOG_DIR/orchestrator.log"
echo ""
echo "Health checks:"
sleep 5
for port in 8101 8102 8103 8104; do
    if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "  localhost:$port OK"
    else
        echo "  localhost:$port NOT READY"
    fi
done

echo ""
echo "Press Ctrl+C to stop all agents."
wait
