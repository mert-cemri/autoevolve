#!/bin/bash
# Run Gradient-Based Evolution for circle packing (n=26)
# Uses gradient = improvement / code_distance for automatic exploration/exploitation
# Run this script from the circle_packing directory

set -e  # Exit on error

# Configuration
ITERATIONS=${1:-100}  # Default 100 iterations (1 eval/iter × 100 = 100 total evaluations)

echo "========================================"
echo "Running Gradient-Based Evolution"
echo "Circle Packing (n=26 circles)"
echo "========================================"
echo "Iterations: $ITERATIONS"
echo "Target: 2.635 (sum of radii)"
echo "Strategy: Gradient = improvement / code_distance"
echo "- Automatic exploration (unvisited programs)"
echo "- Automatic exploitation (high gradient)"
echo "- Zero manual thresholds"
echo "========================================"
echo ""
echo "Requirements:"
echo "  export OPENAI_API_KEY=your_key_here"
echo ""

python ../../openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --config config_gradient.yaml \
  --output openevolve_output_gradient \
  --iterations $ITERATIONS

echo ""
echo "Evolution complete! Results saved to:"
echo "  openevolve_output/best/best_program.py"
echo "  openevolve_output/checkpoints/"
echo "  openevolve_output/memory/memory_log.jsonl  (gradient data)"
echo ""
echo "To visualize the best solution:"
echo "  python -c 'from openevolve_output.best.best_program import run_packing, visualize; c, r, s = run_packing(); print(f\"Sum: {s}\"); visualize(c, r)'"
echo ""
echo "To analyze gradient evolution:"
echo "  grep 'Gradient selection' openevolve_output/evolution.log | tail -20"
echo "  grep 'info_score' openevolve_output/memory/memory_log.jsonl | tail -10"
