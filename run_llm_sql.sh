#!/bin/bash

# === SETUP ===
BASE="$(cd "$(dirname "$0")" && pwd)"

# === EDIT THESE ===
TASK="llm_sql"
OUTPUT="output"  # Evolution output folder

# === RUN ===
EXAMPLE="${BASE}/examples/${TASK}"
CONFIG="${EXAMPLE}/config.yaml"

# Ensure OPENAI_API_KEY is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

python ${BASE}/openevolve-run.py \
    ${EXAMPLE}/initial_program.py \
    ${EXAMPLE}/evaluator.py \
    --config ${CONFIG} \
    --output ${EXAMPLE}/${OUTPUT}

