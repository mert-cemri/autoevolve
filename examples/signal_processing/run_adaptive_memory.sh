#!/bin/bash
# Run OpenEvolve with Adaptive Exploration/Exploitation + Memory for Signal Processing
# This combines MAP-Elites with dynamic exploration/exploitation AND memory-augmented evolution

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing"
echo "Adaptive + Memory"
echo "=========================================="
echo "Strategy: MAP-Elites with Adaptive Search + Memory"
echo "Adaptive: 10-70% exploration based on improvements"
echo "Memory: Semantic search (text-embedding-3-large)"
echo "Iterations: 100"
echo "Models: gpt-5-mini (80%) + gpt-5-nano (20%)"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_adaptive_memory.yaml \
  --output openevolve_output_adaptive_memory \
  --iterations 100

echo ""
echo "=========================================="
echo "Adaptive + Memory run complete!"
echo "Results saved in: openevolve_output_adaptive_memory/"
echo "=========================================="
