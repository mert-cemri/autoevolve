#!/bin/bash

# === SETUP ===
BASE="$(cd "$(dirname "$0")" && pwd)"

# === EDIT THESE ===
TASK="cloudcast"
OUTPUT="run_$(date +%Y%m%d_%H%M%S)"  # Timestamped output folder

# Ensure OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# === RUN ===
EXAMPLE="${BASE}/${TASK}"
CONFIG="${EXAMPLE}/config.yaml"

python ${BASE}/../openevolve-run.py \
    ${EXAMPLE}/initial_program.py \
    ${EXAMPLE}/evaluator.py \
    --config ${CONFIG} \
    --output ${EXAMPLE}/${OUTPUT}

