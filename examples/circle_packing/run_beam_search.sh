#!/bin/bash
# Run Beam Search evolution for circle packing (n=26)

# Activate conda environment if needed
# eval "$(conda shell.bash hook)" 2>/dev/null || true
# conda activate autoevolve 2>/dev/null || echo "Warning: Could not activate conda environment"

# Configuration
ITERATIONS=${1:-100}  # Default 100 iterations

echo "========================================"
echo "Running Beam Search Evolution"
echo "Circle Packing (n=26 circles)"
echo "========================================"
echo "Iterations: $ITERATIONS"
echo "Beam width: 4"
echo "Branch factor: 1 (per iteration)"
echo "Target: 2.635 (sum of radii)"
echo "========================================"

python ../../openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --beam-search \
  --config config_beam_search.yaml \
  --iterations $ITERATIONS

echo ""
echo "Evolution complete! Results saved to:"
echo "  openevolve_output/beam_search/best/best_program.py"
echo "  openevolve_output/beam_search/checkpoints/"
echo ""
echo "To visualize the best solution:"
echo "  python -c 'from openevolve_output.beam_search.best.best_program import run_packing, visualize; c, r, s = run_packing(); print(f\"Sum: {s}\"); visualize(c, r)'"
