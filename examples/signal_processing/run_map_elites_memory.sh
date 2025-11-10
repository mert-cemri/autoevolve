#!/bin/bash
# Run OpenEvolve with MAP-Elites strategy + Memory for Signal Processing

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================="
echo "OpenEvolve - Signal Processing (MAP-Elites + Memory)"
echo "=========================================="
echo "Strategy: MAP-Elites (island-based evolution) with Memory"
echo "Iterations: 100"
echo "Memory: Enabled (semantic search with embeddings)"
echo "=========================================="

cd "$PROJECT_ROOT"

python openevolve-run.py \
  examples/signal_processing/initial_program.py \
  examples/signal_processing/evaluator.py \
  --config examples/signal_processing/config_map_elites_memory.yaml \
  --iterations 100

echo ""
echo "=========================================="
echo "MAP-Elites + Memory run complete!"
echo "Results saved in: openevolve_output/"
echo "=========================================="
