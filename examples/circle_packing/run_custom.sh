#!/bin/bash

# Circle Packing Evolution with Custom Config
# Uses 4 islands with gpt-5 and gpt-5-mini models

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run OpenEvolve with custom configuration
python openevolve-run.py \
    "${SCRIPT_DIR}/initial_program.py" \
    "${SCRIPT_DIR}/evaluator.py" \
    --config "${SCRIPT_DIR}/config_custom.yaml" \
    --iterations 100

echo "Evolution complete! Check openevolve_output/ for results."
