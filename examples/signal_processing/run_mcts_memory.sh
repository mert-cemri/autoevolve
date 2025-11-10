#!/bin/bash
# Run OpenEvolve with MCTS strategy + Memory for Signal Processing

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (MCTS + Memory)"
echo "=========================================="
echo "Strategy: MCTS (expansion_width=4, UCT=√2) with Memory"
echo "Iterations: 100"
echo "Memory: Enabled (semantic search with embeddings)"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_mcts_memory.yaml \
  --mcts

echo ""
echo "=========================================="
echo "MCTS + Memory run complete!"
echo "Results saved in: openevolve_output/"
echo "=========================================="
