#!/bin/bash
# Run LLM-SQL optimization evolution with full adaptive exploration features
# Features enabled:
# - Softmax-weighted exploitation sampling
# - Inverse visit-count exploration sampling
# - Stagnation detection with multi-child generation
# - Sibling context for prompts
# - Multiple seed parents per island

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
OUTPUT_DIR="${SCRIPT_DIR}/output_adaptive_exploration_full_v2"
ITERATIONS=100
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -o, --output DIR      Output directory (default: ${OUTPUT_DIR})"
            echo "  -i, --iterations N    Number of iterations (default: ${ITERATIONS})"
            echo "  -c, --checkpoint DIR  Resume from checkpoint directory"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build checkpoint argument if provided
CHECKPOINT_ARG=""
if [ -n "${CHECKPOINT}" ]; then
    CHECKPOINT_ARG="--checkpoint ${CHECKPOINT}"
fi

echo "=============================================="
echo "LLM-SQL Optimization - Adaptive Exploration"
echo "=============================================="
echo "Config: ${SCRIPT_DIR}/config_adaptive_exploration.yaml"
echo "Output: ${OUTPUT_DIR}"
echo "Iterations: ${ITERATIONS}"
if [ -n "${CHECKPOINT}" ]; then
    echo "Resuming from: ${CHECKPOINT}"
fi
echo "=============================================="

# Activate conda environment if needed
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate autoevolve 2>/dev/null || true
fi

# Run OpenEvolve
cd "${PROJECT_ROOT}"
python openevolve-run.py \
    "${SCRIPT_DIR}/initial_program.py" \
    "${SCRIPT_DIR}/evaluator.py" \
    --config "${SCRIPT_DIR}/config_adaptive_exploration.yaml" \
    --output "${OUTPUT_DIR}" \
    --iterations "${ITERATIONS}" \
    ${CHECKPOINT_ARG}

echo ""
echo "Evolution complete! Results saved to: ${OUTPUT_DIR}"
