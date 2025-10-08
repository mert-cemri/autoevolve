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

### LLM Models (NEW: GPT-5!)
- **Primary**: GPT-5 (80% weight) - For main evolution mutations
- **Secondary**: GPT-5-mini (20% weight) - For diversity
- Both evolution and agents use OpenAI API directly (just need OPENAI_API_KEY)

## What Gets Evolved

The code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` markers is evolved:

1. **Agent System Prompts**:
   - `SOLVER_PROMPT` - How the solver approaches problems
   - `VERIFIER_PROMPT` - Verification criteria and feedback format
   - `REVISER_PROMPT` - Revision strategies
   - `REFINER_PROMPT` - Final polishing approach

2. **Communication Protocol**:
   - `MAX_REVISION_ROUNDS` - Number of verification loops
   - `USE_REFINER` - Whether to use final refinement
   - `REFINER_THRESHOLD` - Confidence threshold for refinement

3. **Agent Functions**:
   - `solver_agent()`, `verifier_agent()`, `reviser_agent()`, `refiner_agent()`
   - `multi_agent_solve()` - Main coordination logic
   - `extract_decision()` - Parsing verifier responses
   - `extract_boxed_answer()` - Answer extraction

## Evaluation Metrics

The system is evaluated on:
- **Accuracy**: Correctness on math problems (using Math500 dataset)
- **Completion Rate**: Percentage of problems attempted
- **Time Efficiency**: Average time per problem
- **Combined Score**: Weighted combination (70% accuracy, 20% completion, 10% speed)

## Output

Results are saved to `examples/math_mas/openevolve_output/`:
- `best/best_program.py` - Best evolved multi-agent system
- `best/best_program_info.json` - Metrics and metadata
- `checkpoints/checkpoint_N/` - Periodic snapshots
- `logs/` - Detailed evolution logs

## Expected Results

The evolution should improve:
- Mathematical reasoning in agent prompts
- Verification criteria and feedback quality
- Error correction strategies
- Answer extraction accuracy

Initial baseline: ~30-40% accuracy on Math500
Expected after evolution with GPT-5: ~60-80% accuracy

## Advanced Usage

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

## Summary

You now have a complete multi-agent system optimization setup:

✅ **initial_program.py** - 4-agent system with evolvable prompts and protocols
✅ **evaluator.py** - Cascade evaluation using Math500 dataset with `get_success()`
✅ **config.yaml** - GPT-5 (80%) + GPT-5-mini (20%) for evolution
✅ **eval_utils.py** - Math evaluation utilities (already present)

Run `python openevolve-run.py examples/math_mas/initial_program.py examples/math_mas/evaluator.py --config examples/math_mas/config.yaml --iterations 50` to start evolution!
