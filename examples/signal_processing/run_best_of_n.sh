#!/bin/bash
# Run OpenEvolve with Best-of-N strategy for Signal Processing
# Evolves N independent lineages in parallel (round-robin)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (Best-of-N)"
echo "=========================================="
echo "Strategy: Best-of-N (4 independent lineages)"
echo "Iterations: 100"
echo "LLM Calls: ~100"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_best_of_n.yaml \
  --best-of-n

echo ""
echo "=========================================="
echo "Best-of-N run complete!"
echo "Results saved in: openevolve_output/"
echo "=========================================="
