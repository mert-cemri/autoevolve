#!/bin/bash
# Run Gradient-Based Evolution for circle packing (n=26)
# Uses gradient = improvement / code_distance for automatic exploration/exploitation
# Run this script from the circle_packing directory

set -e  # Exit on error

# Configuration
ITERATIONS=${1:-100}  # Default 100 iterations (1 eval/iter × 100 = 100 total evaluations)

# Generate timestamp for unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="openevolve_output_gradient_${TIMESTAMP}"

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
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""
echo "Requirements:"
echo "  export OPENAI_API_KEY=your_key_here"
echo ""

python ../../openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --config config_gradient.yaml \
  --output "$OUTPUT_DIR" \
  --iterations $ITERATIONS

echo ""
echo "Evolution complete! Results saved to:"
echo "  $OUTPUT_DIR/best/best_program.py"
echo "  $OUTPUT_DIR/checkpoints/"
echo "  $OUTPUT_DIR/memory_add_log.jsonl  (gradient data)"
echo ""
echo "To visualize the best solution:"
echo "  python -c 'from $OUTPUT_DIR.best.best_program import run_packing, visualize; c, r, s = run_packing(); print(f\"Sum: {s}\"); visualize(c, r)'"
echo ""
echo "To analyze gradient evolution:"
echo "  grep 'Gradient selection' $OUTPUT_DIR/logs/*.log | tail -20"
echo "  grep 'info_score' $OUTPUT_DIR/memory_add_log.jsonl | tail -10"
