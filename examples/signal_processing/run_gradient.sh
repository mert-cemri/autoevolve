#!/bin/bash
# Run OpenEvolve with Gradient-Based Evolution for Signal Processing
# Uses gradient = improvement / code_distance for automatic exploration/exploitation
# Run this script from the signal_processing directory

set -e  # Exit on error

# Generate timestamp for unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="openevolve_output_gradient_${TIMESTAMP}"

echo "=========================================="
echo "OpenEvolve - Signal Processing"
echo "Gradient-Based Evolution"
echo "=========================================="
echo "Strategy: Gradient = improvement / code_distance"
echo "- Automatic exploration (unvisited programs)"
echo "- Automatic exploitation (high gradient)"
echo "- Zero manual thresholds"
echo "Memory: Semantic search with gradient scoring"
echo "Retrieval: info_score = similarity × |gradient|"
echo "Iterations: 100"
echo "Models: gpt-5-mini (80%) + gpt-5-nano (20%)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Requirements:"
echo "  export OPENAI_API_KEY=your_key_here"
echo ""

python ../../openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --config config_gradient.yaml \
  --output "$OUTPUT_DIR" \
  --iterations 100

echo ""
echo "=========================================="
echo "Gradient-based run complete!"
echo "Results saved in: $OUTPUT_DIR/"
echo "=========================================="
echo ""
echo "To analyze gradient evolution:"
echo "  # View parent selection decisions"
echo "  grep 'Gradient selection' $OUTPUT_DIR/logs/*.log | tail -20"
echo ""
echo "  # View memory retrieval with gradients"
echo "  grep 'info_score' $OUTPUT_DIR/memory_add_log.jsonl | tail -10"
echo ""
echo "  # View best program"
echo "  cat $OUTPUT_DIR/best/best_program.py"
echo ""
