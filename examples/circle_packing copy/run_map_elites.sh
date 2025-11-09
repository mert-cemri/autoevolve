#!/bin/bash
# Run MAP-Elites evolution for circle packing (n=26)

# Activate conda environment if needed
# eval "$(conda shell.bash hook)" 2>/dev/null || true
# conda activate autoevolve 2>/dev/null || echo "Warning: Could not activate conda environment"

# Configuration
ITERATIONS=${1:-100}

echo "========================================"
echo "Running MAP-Elites Evolution"
echo "Circle Packing (n=26 circles)"
echo "========================================"
echo "Iterations: $ITERATIONS"
echo "Target: 2.635 (sum of radii)"
echo "========================================"

python ../../openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --config config_map_elites.yaml \
  --iterations $ITERATIONS

echo ""
echo "Evolution complete! Results saved to:"
echo "  openevolve_output/best/best_program.py"
echo "  openevolve_output/checkpoints/"
echo ""
echo "To visualize the best solution:"
echo "  python -c 'from openevolve_output.best.best_program import run_packing, visualize; c, r, s = run_packing(); print(f\"Sum: {s}\"); visualize(c, r)'"
