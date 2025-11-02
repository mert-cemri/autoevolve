#!/bin/bash
# Run Best-of-N evolution for 100 iterations with 100 questions

# Configuration
ITERATIONS=${1:-100}
NUM_PROBLEMS=${2:-100}

# Set environment
export MATH_EVAL_PROBLEMS=$NUM_PROBLEMS

echo "========================================"
echo "Running Best-of-N Evolution"
echo "========================================"
echo "Iterations: $ITERATIONS"
echo "Problems per evaluation: $NUM_PROBLEMS"
echo "========================================"

python ../../openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --best-of-n \
  --config config_best_of_n.yaml \
  --iterations $ITERATIONS

echo ""
echo "Evolution complete! Results saved to:"
echo "  openevolve_output/best_of_n/best/best_program.py"
echo "  openevolve_output/best_of_n/checkpoints/"
