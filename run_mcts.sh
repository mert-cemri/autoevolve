#!/bin/bash
# Run MCTS strategy

ITERATIONS="${1:-50}"
MATH_PROBLEMS="${MATH_EVAL_PROBLEMS:-100}"
AGENT_MODEL="${OPENEVOLVE_MODEL:-gpt-5-nano}"

MATH_EVAL_PROBLEMS="$MATH_PROBLEMS" OPENEVOLVE_MODEL="$AGENT_MODEL" \
python openevolve-run.py \
  examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --mcts \
  --iterations "$ITERATIONS"
