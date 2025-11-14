#!/bin/bash

# Signal Processing Evolution with Custom Config
# Uses 4 islands with gpt-5-mini and gpt-5-nano models
# Modified algorithm: Best-of-K (k=3), exploration from archive, exploitation picks best

set -e  # Exit on error

# Get script directory and repo root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/../.." && pwd )"

# Set custom output directory to avoid overwriting
OUTPUT_DIR="${SCRIPT_DIR}/openevolve_output_best_of_k"

# Run OpenEvolve with custom configuration
cd "${REPO_ROOT}"
python openevolve-run.py \
    "${SCRIPT_DIR}/initial_program.py" \
    "${SCRIPT_DIR}/evaluator.py" \
    --config "${SCRIPT_DIR}/config_custom.yaml" \
    --output "${OUTPUT_DIR}" \
    --iterations 100

echo "Evolution complete! Results saved to: ${OUTPUT_DIR}"
echo "Config: Best-of-3 children, exploration from archive, exploitation picks highest score"
