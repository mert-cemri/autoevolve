#!/bin/bash
# Master script to run gradient-based evolution experiments
# Usage: ./run_gradient_experiments.sh [all|signal|symbolic|circle]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY=your_key_here"
    exit 1
fi

echo "========================================"
echo "Gradient-Based Evolution Experiments"
echo "========================================"
echo "Strategy: gradient = improvement / code_distance"
echo "- Automatic exploration/exploitation"
echo "- Zero manual thresholds"
echo "- Memory-augmented with gradient scoring"
echo "========================================"
echo ""

EXPERIMENT=${1:-all}

run_signal_processing() {
    echo "=========================================="
    echo "1/3: Signal Processing"
    echo "=========================================="
    cd "$SCRIPT_DIR/examples/signal_processing"
    bash run_gradient.sh 100
    cd "$SCRIPT_DIR"
    echo ""
}

run_symbolic_regression() {
    echo "=========================================="
    echo "2/3: Symbolic Regression"
    echo "=========================================="
    cd "$SCRIPT_DIR/examples/symbolic_regression"
    bash run_gradient.sh 200
    cd "$SCRIPT_DIR"
    echo ""
}

run_circle_packing() {
    echo "=========================================="
    echo "3/3: Circle Packing"
    echo "=========================================="
    cd "$SCRIPT_DIR/examples/circle_packing"
    bash run_gradient.sh 100
    cd "$SCRIPT_DIR"
    echo ""
}

case $EXPERIMENT in
    signal)
        run_signal_processing
        ;;
    symbolic)
        run_symbolic_regression
        ;;
    circle)
        run_circle_packing
        ;;
    all)
        run_signal_processing
        run_symbolic_regression
        run_circle_packing
        ;;
    *)
        echo "Usage: $0 [all|signal|symbolic|circle]"
        echo ""
        echo "Options:"
        echo "  all       - Run all three experiments (default)"
        echo "  signal    - Run signal processing only"
        echo "  symbolic  - Run symbolic regression only"
        echo "  circle    - Run circle packing only"
        exit 1
        ;;
esac

echo "========================================"
echo "All experiments complete!"
echo "========================================"
echo ""
echo "Results directories:"
echo "  examples/signal_processing/openevolve_output_gradient/"
echo "  examples/symbolic_regression/openevolve_output_gradient/"
echo "  examples/circle_packing/openevolve_output/"
echo ""
echo "To analyze gradient statistics:"
echo "  grep 'Gradient selection' examples/*/openevolve_output*/evolution.log | wc -l"
echo ""
