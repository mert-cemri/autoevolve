#!/bin/bash
# Run OpenEvolve with MAP-Elites strategy for Signal Processing
# This is the default strategy using island-based evolution

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (MAP-Elites)"
echo "=========================================="
echo "Strategy: MAP-Elites (island-based evolution)"
echo "Iterations: 100"
echo "LLM Calls: ~100"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config.yaml \
  --iterations 100

echo ""
echo "=========================================="
echo "MAP-Elites run complete!"
echo "Results saved in: openevolve_output/"
echo "=========================================="
