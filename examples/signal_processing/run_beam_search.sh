#!/bin/bash
# Run OpenEvolve with Beam Search strategy for Signal Processing
# Keeps top M programs and branches N candidates per iteration

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (Beam Search)"
echo "=========================================="
echo "Strategy: Beam Search (beam_width=4, branch_factor=1)"
echo "Iterations: 25"
echo "LLM Calls: ~100"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_beam_search.yaml \
  --beam-search

echo ""
echo "=========================================="
echo "Beam Search run complete!"
echo "Results saved in: openevolve_output/"
echo "=========================================="
