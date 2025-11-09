#!/bin/bash
# Run MAP-Elites evolution for 100 iterations with 20 questions

# Activate conda environment
# Note: If this fails, run manually: conda activate autoevolve
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate autoevolve 2>/dev/null || echo "Warning: Could not activate conda environment. Please run: conda activate autoevolve"

# Configuration
ITERATIONS=${1:-100}
NUM_PROBLEMS=${2:-20}

# Set environment
export MATH_EVAL_PROBLEMS=$NUM_PROBLEMS

echo "========================================"
echo "Running MAP-Elites Evolution"
echo "========================================"
echo "Iterations: $ITERATIONS"
echo "Problems per evaluation: $NUM_PROBLEMS"
echo "========================================"

python ../../openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --config config.yaml \
  --iterations $ITERATIONS

echo ""
echo "Evolution complete! Results saved to:"
echo "  openevolve_output/best/best_program.py"
echo "  openevolve_output/checkpoints/"
