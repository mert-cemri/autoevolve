#!/bin/bash
# Run OpenEvolve with Gradient-Based Evolution for Symbolic Regression
# Uses gradient = improvement / code_distance for automatic exploration/exploitation
# Run this script from the symbolic_regression directory

set -e  # Exit on error

# Default to first problem in chem_react if not specified
PROBLEM_DIR=${1:-"problems/chem_react/CRK0"}
ITERATIONS=${2:-200}

if [[ ! -d "$PROBLEM_DIR" ]]; then
    echo "ERROR: Problem directory '$PROBLEM_DIR' not found"
    echo ""
    echo "Usage: $0 [problem_dir] [iterations]"
    echo ""
    echo "Available problems:"
    echo "  problems/bio_pop_growth/BPG0 through BPG23"
    echo "  problems/chem_react/CRK0 through CRK35"
    echo "  problems/matsci/MatSci0 through MatSci24"
    echo "  problems/phys_osc/PO0 through PO43"
    echo ""
    echo "Example:"
    echo "  $0 problems/chem_react/CRK0 200"
    exit 1
fi

if [[ ! -f "$PROBLEM_DIR/initial_program.py" ]]; then
    echo "ERROR: initial_program.py not found in $PROBLEM_DIR"
    exit 1
fi

if [[ ! -f "$PROBLEM_DIR/evaluator.py" ]]; then
    echo "ERROR: evaluator.py not found in $PROBLEM_DIR"
    exit 1
fi

echo "=========================================="
echo "OpenEvolve - Symbolic Regression"
echo "Gradient-Based Evolution"
echo "=========================================="
echo "Problem: $PROBLEM_DIR"
echo "Strategy: Gradient = improvement / code_distance"
echo "- Automatic exploration (unvisited programs)"
echo "- Automatic exploitation (high gradient)"
echo "- Zero manual thresholds"
echo "Memory: Semantic search with gradient scoring"
echo "Retrieval: info_score = similarity × |gradient|"
echo "Iterations: $ITERATIONS"
echo "=========================================="
echo ""
echo "Requirements:"
echo "  export OPENAI_API_KEY=your_key_here"
echo ""

# Use the gradient config from the root symbolic_regression directory
python ../../openevolve-run.py \
  "$PROBLEM_DIR/initial_program.py" \
  "$PROBLEM_DIR/evaluator.py" \
  --config config_gradient.yaml \
  --output "$PROBLEM_DIR/openevolve_output_gradient" \
  --iterations $ITERATIONS

echo ""
echo "=========================================="
echo "Gradient-based run complete!"
echo "Results saved in: $PROBLEM_DIR/openevolve_output_gradient/"
echo "=========================================="
echo ""
echo "To analyze gradient evolution:"
echo "  # View parent selection decisions"
echo "  grep 'Gradient selection' $PROBLEM_DIR/openevolve_output_gradient/evolution.log | tail -20"
echo ""
echo "  # View memory retrieval with gradients"
echo "  grep 'info_score' $PROBLEM_DIR/openevolve_output_gradient/memory/memory_log.jsonl | tail -10"
echo ""
echo "  # View best program"
echo "  cat $PROBLEM_DIR/openevolve_output_gradient/best/best_program.py"
echo ""
