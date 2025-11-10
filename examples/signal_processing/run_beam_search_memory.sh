#!/bin/bash
# Run OpenEvolve with Beam Search strategy + Memory for Signal Processing

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (Beam Search + Memory)"
echo "=========================================="
echo "Strategy: Beam Search (beam_width=4, branch_factor=1) with Memory"
echo "Iterations: 100"
echo "Memory: Enabled (semantic search with embeddings)"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_beam_search_memory.yaml \
  --beam-search

echo ""
echo "=========================================="
echo "Beam Search + Memory run complete!"
echo "Results saved in: openevolve_output/"
echo "=========================================="
