# Custom Search Strategies for Code Evolution

This directory contains alternative search strategies for evolving code as ablations to compare against OpenEvolve.

**📖 For a detailed explanation of how evolution works, see [EVOLUTION_EXPLAINED.md](EVOLUTION_EXPLAINED.md)**

## How Evolution Works

The search strategies evolve the **code** of the multi-agent system, not the solutions to math problems.

### Evolution Process

1. **Program Structure**: The `initial_program.py` contains marked sections between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` that can be evolved:
   - Agent prompts (SOLVER_PROMPT, VERIFIER_PROMPT, etc.)
   - Communication protocols (MAX_REVISION_ROUNDS, USE_REFINER, etc.)
   - Decision logic and control flow

2. **Mutation**: The evolution model (e.g., GPT-5) receives:
   - Current program code
   - Performance metrics (accuracy, efficiency, speed)
   - Evaluation feedback from previous iterations
   - Specific guidance on what to improve

3. **Validation**: Generated programs are validated for:
   - Syntactic correctness (AST parsing)
   - Required functions (`run_evaluation_sample`)
   - EVOLVE-BLOCK markers preserved
   - If validation fails, the parent code is used instead

4. **Evaluation**: Valid programs are evaluated by:
   - Running them on 10 random math problems (seed=42)
   - Measuring accuracy (% correct), efficiency (LLM calls), and speed
   - Computing combined score: 70% accuracy + 25% efficiency + 5% speed

5. **Selection**: Better-performing programs are kept and evolved further

### Two-Model Architecture

- **Evolution Model** (`model`): Evolves the multi-agent system code (e.g., GPT-5)
- **Agent Model** (`agent_model`): Powers the agents solving math problems (e.g., GPT-5-mini)

This separation allows expensive models for code evolution and cheaper models for problem-solving.

### Example Evolution Flow

```
Initial Program (accuracy=50%, efficiency=0.7, score=0.525)
    ↓
[Evolution Model] Generate improved code
    ↓
[Validation] Check syntax & required functions
    ↓ (if valid)
Mutated Program
    ↓
[Evaluation] Run on 10 math problems using agent_model
    ↓
Metrics: accuracy=60%, efficiency=0.8, speed=0.7
    ↓
Combined Score = 0.7×0.6 + 0.25×0.8 + 0.05×0.7 = 0.655
    ↓
[Selection] Keep if score > parent.score
    ↓
Use as parent for next iteration...
```

**Key Insight**: The evolution model (GPT-5) improves the multi-agent system's **code**, while the agent model (GPT-5-mini) is used by that code to **solve problems**. Better agent prompts and protocols → better problem-solving → higher scores.

### Potential Issues & Solutions

| Issue | How We Address It |
|-------|-------------------|
| **Generated code has syntax errors** | AST parsing validation before evaluation |
| **Missing required functions** | Check for `run_evaluation_sample` function |
| **LLM removes EVOLVE markers** | Validation ensures markers are present |
| **Non-executable programs** | Fallback to parent code if validation fails |
| **Evolution doesn't know what to improve** | Detailed prompt with metrics, feedback, and specific guidance |
| **No feedback on what works** | Performance metrics passed to mutation prompt |
| **Code changes break functionality** | Prompt emphasizes keeping code outside EVOLVE blocks unchanged |
| **Expensive evaluation** | Use cheaper agent_model for problem-solving |

## Search Strategies

### 1. Best of N
Generates N independent evolutionary lineages and evolves each linearly.

**Algorithm:**
1. Create N variants from initial program
2. Evolve each lineage independently for T iterations
3. Return the best program across all lineages

**Pros:**
- Simple and parallelizable
- Good diversity through independent lineages
- No premature convergence

**Cons:**
- No information sharing between lineages
- May be inefficient compared to focused search

### 2. Beam Search
Maintains a beam of M best programs and branches N times per iteration.

**Algorithm:**
1. Start with initial program
2. For T iterations:
   - Branch each beam member to create N total candidates
   - Evaluate all candidates
   - Keep top M as new beam
3. Return best program from final beam

**Pros:**
- Balances exploration and exploitation
- Maintains diversity in beam
- Can recover from local optima

**Cons:**
- Greedy selection may miss good programs
- Requires many evaluations per iteration

### 3. Monte Carlo Tree Search (MCTS)
Uses tree search with UCT (Upper Confidence bounds for Trees) selection.

**Algorithm:**
1. **Selection:** Use UCT to traverse tree to a leaf
2. **Expansion:** Generate children from the leaf
3. **Simulation:** Evaluate new programs
4. **Backpropagation:** Update statistics up to root
5. Repeat for budget iterations

**Pros:**
- Principled exploration-exploitation tradeoff
- Adapts based on feedback
- Can build deep search trees

**Cons:**
- More complex implementation
- Needs tuning of exploration constant
- Tree can grow large

## Installation

```bash
cd custom_search
```

The `custom_search/` directory is self-contained with:
- `initial_program.py` - Multi-agent math solving system
- `evaluator.py` - Math500 evaluation logic
- `eval_utils.py` - Evaluation utilities
- Search strategy implementations
- Configuration files

Requires:
- `openai` - For LLM API (evolution model)
- `langchain-openai` - For multi-agent system (agent model)
- `pyyaml` - For configuration
- `datasets` - For Math500 dataset

## Usage

### Quick Start

```bash
# Run Best of N (output auto-generated with timestamp)
python run_search.py best_of_n

# Run Beam Search (output auto-generated with timestamp)
python run_search.py beam_search

# Run MCTS (output auto-generated with timestamp)
python run_search.py mcts
```

Each run automatically creates a timestamped output directory:
- `custom_search/results/best_of_n/run_2025-10-08_14-30-45/`
- `custom_search/results/beam_search/run_2025-10-08_14-35-22/`
- `custom_search/results/mcts/run_2025-10-08_15-00-10/`

### Custom Configuration

```bash
# Use custom config file
python run_search.py best_of_n --config my_config.yaml

# Override specific parameters (including custom output dir)
python run_search.py best_of_n \
  --initial-program path/to/program.py \
  --evaluator path/to/evaluator.py \
  --output-dir results/my_custom_run
```

You can also override parameters via environment variables:
```bash
# Evaluate on 20 problems instead of default 10
MATH_EVAL_PROBLEMS=20 python run_search.py best_of_n

# Use different agent model (GPT-4o for agents instead of GPT-5-mini)
OPENEVOLVE_MODEL=gpt-4o python run_search.py best_of_n

# Combine multiple overrides
MATH_EVAL_PROBLEMS=50 OPENEVOLVE_MODEL=gpt-4o python run_search.py mcts
```

## Configuration

Each strategy has a default config in `config/`:
- `best_of_n_config.yaml`
- `beam_search_config.yaml`
- `mcts_config.yaml`

### Best of N Configuration

```yaml
# Search parameters
n: 4                    # Number of parallel lineages
iterations: 10          # Iterations per lineage

# Evaluation parameters
num_eval_problems: 10   # Number of problems to evaluate on (randomly sampled with seed=42)

# LLM configuration
model: "gpt-5"            # Model for search/evolution (mutating programs)
agent_model: "gpt-5-mini" # Model for multi-agent system (solving problems)
temperature: 0.8
max_tokens: 16000
```

### Beam Search Configuration

```yaml
# Search parameters
beam_width: 4           # Beam size (M)
branch_factor: 8        # Total branches (N)
iterations: 10          # Number of iterations (T)

# Evaluation parameters
num_eval_problems: 10   # Number of problems to evaluate on (randomly sampled with seed=42)

# LLM configuration
model: "gpt-5"            # Model for search/evolution (mutating programs)
agent_model: "gpt-5-mini" # Model for multi-agent system (solving problems)
temperature: 0.8
max_tokens: 16000
```

### MCTS Configuration

```yaml
# Search parameters
iterations: 50          # MCTS iterations
expansion_width: 3      # Children per expansion
exploration_constant: 1.414  # UCT exploration (√2)

# Evaluation parameters
num_eval_problems: 10   # Number of problems to evaluate on (randomly sampled with seed=42)

# LLM configuration
model: "gpt-5"            # Model for search/evolution (mutating programs)
agent_model: "gpt-5-mini" # Model for multi-agent system (solving problems)
temperature: 0.8
max_tokens: 16000
```

**Model Configuration**:
- **`model`**: The LLM used by the search algorithm to mutate/evolve programs (e.g., GPT-5 for generating improved code)
- **`agent_model`**: The LLM used by the multi-agent system to solve math problems (e.g., GPT-5-mini for cost efficiency)
- This separation allows you to use a powerful model for evolution and a cheaper model for problem-solving

**Evaluation Settings**:
- Problems are randomly sampled from HuggingFaceH4/MATH-500 dataset with `seed=42` for reproducibility
- All runs with the same `num_eval_problems` will evaluate on the exact same problems
- This ensures fair comparison across different search strategies

**Important GPT-5 API Notes**:
- GPT-5 uses `max_completion_tokens` instead of the older `max_tokens` parameter
- GPT-5 only supports `temperature=1` (default). Custom temperature values are automatically ignored for GPT-5
- The code automatically handles these API differences - no changes needed to the config files

## Output

Each search saves results to an auto-generated timestamped directory:

### Search Results
```
custom_search/results/{strategy_name}/run_{YYYY-MM-DD_HH-MM-SS}/
  # Iteration tracking
  iteration_0000_best.py     # Initial program
  iteration_0000_best.json   # Metadata
  iteration_0001_best.py     # Best after iteration 1
  iteration_0001_best.json
  ...
  iteration_summary.json     # Complete score history across iterations

  # Final results
  best_program.py            # Best program found
  best_program.json          # Metadata
  history.json               # Detailed search history
  search.log                 # Detailed logs

  # Strategy-specific:
  lineage_*.py               # Best of N final lineages
  beam_*.py                  # Beam Search final beam
  mcts_tree_stats.json       # MCTS tree statistics
```

### Conversation Logs
Multi-agent conversations are logged hierarchically by strategy:

```
logs/
├── best_of_n/
│   └── run_{YYYY-MM-DD_HH-MM-SS}/
│       ├── iteration_0000.json    # Initial evaluation conversations
│       ├── iteration_0001.json    # Iteration 1 conversations
│       └── ...
├── beam_search/
│   └── run_{YYYY-MM-DD_HH-MM-SS}/
│       └── ...
└── mcts/
    └── run_{YYYY-MM-DD_HH-MM-SS}/
        └── ...
```

Each `iteration_XXXX.json` contains:
- Full conversation history for all problems in that iteration
- Agent roles, messages, and LLM calls
- Problem text, gold answers, and predictions
- Timestamps and metadata
```

## Running Ablation Studies

To compare all strategies (each gets auto-timestamped directory):

```bash
# Run all three strategies (output dirs auto-generated with timestamps)
python run_search.py best_of_n
python run_search.py beam_search
python run_search.py mcts

# Compare results across all runs
python compare_results.py custom_search/results/*/run_*/iteration_summary.json

# Or run with custom base directory for easier comparison
python run_search.py best_of_n --output-dir results/ablation_2025/best_of_n
python run_search.py beam_search --output-dir results/ablation_2025/beam_search
python run_search.py mcts --output-dir results/ablation_2025/mcts
```

## Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-api-key"

# Optional (for evaluator)
export MATH_EVAL_PROBLEMS="10"
export OPENEVOLVE_MODEL="gpt-4o-mini"
```

## Comparing to OpenEvolve

Run OpenEvolve for comparison:

```bash
# OpenEvolve with MAP-Elites + Islands
python openevolve-run.py \
  examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --config examples/math_mas/config.yaml \
  --iterations 50
```

Then compare:
- Final best program scores
- Number of LLM calls
- Diversity of solutions
- Convergence speed

## Expected Results

Rough performance expectations (problem-dependent):

| Strategy | Best Score | Diversity | LLM Calls | Speed |
|----------|-----------|-----------|-----------|-------|
| Best of N | Medium | High | N×T | Fast |
| Beam Search | High | Medium | N×T | Medium |
| MCTS | High | Medium-High | E×I | Slow |
| **OpenEvolve** | **Highest** | **Highest** | **Varies** | **Medium** |

Where:
- N = number of lineages/branches
- T = iterations
- E = expansion width
- I = MCTS iterations

OpenEvolve typically finds better solutions due to:
- MAP-Elites quality-diversity tradeoff
- Island-based evolution for diversity
- Adaptive population management

## Directory Structure

```
custom_search/
├── initial_program.py          # Multi-agent math system to evolve
├── evaluator.py                # Math500 evaluation
├── eval_utils.py               # Evaluation utilities
├── base_search.py              # Base search class
├── best_of_n.py                # Best of N search
├── beam_search.py              # Beam search
├── mcts_search.py              # MCTS search
├── run_search.py               # CLI entry point
└── config/
    ├── best_of_n_config.yaml
    ├── beam_search_config.yaml
    └── mcts_config.yaml
```

## Troubleshooting

### "Module not found" errors
```bash
# Run from the custom_search directory
cd custom_search
python run_search.py best_of_n
```

### API rate limits
- Reduce iteration counts
- Use slower models (gpt-4o-mini)
- Add delays between LLM calls

### Evaluation failures
- Check MATH_EVAL_PROBLEMS is not too high
- Verify evaluator dependencies installed
- Check logs in output_dir/search.log

## Citation

If you use these search strategies in your research:

```bibtex
@software{custom_search_2025,
  title = {Custom Search Strategies for Code Evolution},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/autoevolve}
}
```
