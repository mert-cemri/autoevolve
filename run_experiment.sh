#!/bin/bash

# Ensure OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# === EDIT THESE ===
# TASK="circle_packing"
# TASK="algotune/eigenvectors_complex"
# TASK="algotune/fft_convolution"
# TASK="signal_processing"  
CONFIG="config.yaml"
OUTPUT="output"  # Evolution output folder

# === RUN ===
BASE="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE="${BASE}/examples/${TASK}"

python ${BASE}/openevolve-run.py \
    ${EXAMPLE}/initial_program.py \
    ${EXAMPLE}/evaluator.py \
    --config ${EXAMPLE}/${CONFIG} \
    --output ${EXAMPLE}/${OUTPUT}