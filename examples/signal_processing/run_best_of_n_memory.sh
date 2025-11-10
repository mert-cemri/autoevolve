#!/bin/bash
# Run OpenEvolve with Best-of-N strategy + Memory for Signal Processing

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (Best-of-N + Memory)"
echo "=========================================="
echo "Strategy: Best-of-N (4 independent lineages) with Memory"
echo "Iterations: 100"
echo "Memory: Enabled (semantic search with embeddings)"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_best_of_n_memory.yaml \
  --best-of-n

echo ""
echo "=========================================="
echo "Best-of-N + Memory run complete!"
echo "Results saved in: openevolve_output/"
echo "=========================================="
