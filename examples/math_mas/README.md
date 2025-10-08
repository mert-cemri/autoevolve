# Multi-Agent Math Solver Evolution

This example uses OpenEvolve to evolve a multi-agent system for solving mathematical problems from the Math500 dataset.

## Overview

The system evolves a collaborative multi-agent architecture with up to 4 agents:
- **Solver**: Initial problem-solving
- **Verifier**: Solution verification
- **Reviser**: Error correction based on feedback
- **Refiner**: Final answer polishing

OpenEvolve optimizes:
- Agent system prompts (roles and expertise)
- Communication protocols (interaction patterns)
- Workflow structure (agent coordination)

## Setup

### 1. Install Dependencies

```bash
# Install OpenEvolve in development mode
cd /Users/mertcemri/Desktop/research/autoevolve
pip install -e ".[dev]"

# Install additional dependencies for this example
pip install langchain langchain-openai datasets word2number sympy latex2sympy2
```

### 2. Set Environment Variables

```bash
# OpenAI API key (used for both evolution and multi-agent system)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Configure the model used by agents (inside the multi-agent system)
export OPENEVOLVE_MODEL="gpt-4o-mini"  # Default model for agents in the multi-agent system

# Optional: Number of test problems per evaluation
export MATH_EVAL_PROBLEMS="10"  # Default: 10 problems
```

### 3. Test the Initial System

```bash
cd examples/math_mas

# Test the initial multi-agent system
python initial_program.py

# Test the evaluator
python evaluator.py
```

## Running Evolution

### Basic Evolution Run

```bash
# From the repository root
python openevolve-run.py \
  examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --config examples/math_mas/config.yaml \
  --iterations 50
```

### Resume from Checkpoint

```bash
python openevolve-run.py \
  examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --config examples/math_mas/config.yaml \
  --checkpoint examples/math_mas/openevolve_output/checkpoints/checkpoint_40 \
  --iterations 20
```

## Configuration

Key configuration options in `config.yaml`:

### Evolution Settings
- `max_iterations: 50` - Number of evolution iterations
- `diff_based_evolution: false` - Use full rewrites instead of diffs
- `early_stopping_patience: 20` - Stop if no improvement for 20 iterations

### Database (MAP-Elites)
- `population_size: 100` - Maximum programs in population
- `num_islands: 4` - Isolated populations for diversity
- `feature_dimensions: [accuracy, completion_rate]` - Quality-diversity space

### Evaluator
- `cascade_evaluation: true` - Fast-fail for bad programs
- `parallel_evaluations: 4` - Run 4 evaluations concurrently
- `timeout: 600` - 10 minute timeout per evaluation




### Adjust Problem Difficulty
```bash
# Use fewer problems for faster iterations
export MATH_EVAL_PROBLEMS="5"

# Use more problems for better evaluation
export MATH_EVAL_PROBLEMS="20"
```

### Customize Agent Model
```bash
# Use a more powerful model for agents
export OPENEVOLVE_MODEL="gpt-4o"

# Or use GPT-5 for agents too (expensive but powerful)
export OPENEVOLVE_MODEL="gpt-5"
```

### Visualize Evolution
```bash
# After evolution completes
python scripts/visualizer.py \
  --path examples/math_mas/openevolve_output/checkpoints/checkpoint_50/
```

## Troubleshooting

### "Import langchain_openai could not be resolved"
```bash
pip install langchain langchain-openai
```

### "No problems loaded"
```bash
# Install datasets library
pip install datasets

# Or problems will fall back to synthetic test problems
```

### API Rate Limits
- Reduce `parallel_evaluations` in config.yaml
- Increase `timeout` if models are slow
- Use faster models (gpt-4o-mini instead of gpt-5)

### Out of Memory
- Reduce `population_size` in config.yaml
- Reduce `MATH_EVAL_PROBLEMS` environment variable
- Enable cascade evaluation to fail fast on bad programs

