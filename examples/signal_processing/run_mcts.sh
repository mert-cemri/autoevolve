#!/bin/bash
# Run OpenEvolve with MCTS strategy for Signal Processing
# Uses Monte Carlo Tree Search with UCT selection

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (MCTS)"
echo "=========================================="
echo "Strategy: MCTS (expansion_width=4, UCT=√2)"
echo "Iterations: 25"
echo "LLM Calls: ~100"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_mcts.yaml \
  --mcts

echo ""
echo "=========================================="
echo "MCTS run complete!"
echo "Results saved in: openevolve_output/"
echo "=========================================="
