#!/bin/bash
# Run OpenEvolve with Adaptive Exploration/Exploitation for Signal Processing
# This uses MAP-Elites with dynamic exploration/exploitation ratios

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (Adaptive)"
echo "=========================================="
echo "Strategy: MAP-Elites with Adaptive Search"
echo "Adaptive: 10-70% exploration based on improvements"
echo "Iterations: 100"
echo "Models: gpt-5-mini (80%) + gpt-5-nano (20%)"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_adaptive.yaml \
  --output openevolve_output_adaptive \
  --iterations 100

echo ""
echo "=========================================="
echo "Adaptive run complete!"
echo "Results saved in: openevolve_output_adaptive/"
echo "=========================================="
