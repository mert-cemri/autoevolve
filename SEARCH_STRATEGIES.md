python openevolve-run.py examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --config examples/math_mas/config.yaml \
  --iterations 50

# Best-of-N
python openevolve-run.py examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --best-of-n \
  --iterations 50

# Beam Search
python openevolve-run.py examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --beam-search \
  --iterations 50

# MCTS
python openevolve-run.py examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --mcts \
  --iterations 50


# Search Strategies in OpenEvolve

OpenEvolve now supports multiple search strategies that all share the same infrastructure (prompt building, parallel execution, LLM ensembles) but use different parent selection and program organization approaches.

## Available Strategies

### 1. MAP-Elites (Default)
**Command**: `python openevolve-run.py` (no strategy flag)

The default strategy using MAP-Elites with island-based evolution.

**Key Features**:
- Multi-dimensional feature grid (e.g., accuracy × efficiency)
- Island-based populations with migration
- Maintains diversity through quality-diversity trade-off
- Parent selection: 50% exploit (top 20%), 50% explore (random)

**When to use**:
- When you want to explore diverse solutions
- When feature dimensions are well-defined
- For long evolution runs (100+ iterations)

**Config**: `examples/math_mas/config.yaml`

### 2. Best-of-N
**Command**: `python openevolve-run.py --best-of-n`

Maintains N independent lineages that evolve in parallel.

**Key Features**:
- N independent lineages (default: 4)
- Each lineage evolves linearly (child replaces parent if better)
- No information sharing between lineages
- Fully parallelized execution

**When to use**:
- When you want simple, interpretable evolution
- For quick experiments
- When you don't have well-defined feature dimensions

**Config**: `examples/math_mas/config_best_of_n.yaml`

**Parameters**:
```yaml
search_strategy: "best_of_n"
n_lineages: 4  # Number of independent lineages
```

### 3. Beam Search
**Command**: `python openevolve-run.py --beam-search`

Maintains a beam of M best programs and branches N times per iteration.

**Key Features**:
- Beam width M (default: 4) - keeps top M programs
- Branch factor N (default: 8) - generates N candidates per iteration
- Greedy selection (keeps best)
- Balances exploration and exploitation

**When to use**:
- When you want focused search around best solutions
- For medium-length runs (20-50 iterations)
- When quality matters more than diversity

**Config**: `examples/math_mas/config_beam_search.yaml`

**Parameters**:
```yaml
search_strategy: "beam_search"
beam_width: 4       # Keep top M programs
branch_factor: 8    # Generate N candidates per iteration
```

### 4. MCTS (Monte Carlo Tree Search)
**Command**: `python openevolve-run.py --mcts`

Uses UCT (Upper Confidence bounds for Trees) for selection.

**Key Features**:
- Tree-based search structure
- UCT balances exploration vs exploitation
- Tracks visit counts and rewards
- Principled exploration strategy

**When to use**:
- When you want adaptive exploration
- For problems with clear reward signals
- When you need balance between diversity and quality

**Config**: `examples/math_mas/config_mcts.yaml`

**Parameters**:
```yaml
search_strategy: "mcts"
expansion_width: 3          # Children per expansion
exploration_constant: 1.414  # UCT exploration (√2)
```

## Usage Examples

### Basic Usage

```bash
# Use default MAP-Elites
python openevolve-run.py \
  examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --config examples/math_mas/config.yaml \
  --iterations 50

# Use Best-of-N
python openevolve-run.py \
  examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --best-of-n \
  --iterations 50

# Use Beam Search
python openevolve-run.py \
  examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --beam-search \
  --iterations 50

# Use MCTS
python openevolve-run.py \
  examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --mcts \
  --iterations 50
```

### Using Shell Scripts

Convenient wrapper scripts are provided:

```bash
# Best-of-N (default: 50 iterations, 100 problems, gpt-5-nano agent)
./run_best_of_n.sh

# Custom iterations
./run_best_of_n.sh 100

# Override environment variables
MATH_EVAL_PROBLEMS=200 OPENEVOLVE_MODEL=gpt-4o ./run_best_of_n.sh 50

# Beam Search
./run_beam_search.sh 50

# MCTS
./run_mcts.sh 50
```

### Auto-Config Selection

If no config is specified, the appropriate config is automatically selected:

```bash
# Automatically uses config_best_of_n.yaml
python openevolve-run.py examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py --best-of-n

# Automatically uses config_beam_search.yaml
python openevolve-run.py examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py --beam-search
```

## Shared Infrastructure

All strategies share the following OpenEvolve components:

### 1. Prompt Building
- Same prompt templates
- Context programs (top performers + diverse examples)
- Evolution history
- Metrics and improvement areas

### 2. Parallel Execution
- Process-based parallelism (4 workers by default)
- Worker pool with database snapshots
- True parallelism (bypasses Python GIL)

### 3. LLM Ensembles
- Evolution models (gpt-5 + gpt-5-mini for generating code)
- Agent models (gpt-5-nano for multi-agent system)
- Weighted ensemble with configurable ratios

### 4. Evaluation
- Cascade evaluation (Stage 1 → Stage 2 → Stage 3)
- Timeout protection
- Metrics calculation (accuracy, efficiency, combined score)

### 5. Checkpointing
- Save/resume support
- Iteration tracking
- Best program preservation

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│ OpenEvolve / OpenEvolveWithStrategy                        │
│  - Config loading                                          │
│  - LLM ensembles (evolution + agent)                       │
│  - Evaluator                                               │
│  - Prompt sampler                                          │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│ Search Strategy (pluggable)                                │
│  ├─ MAPElitesStrategy (default)                            │
│  ├─ BestOfNStrategy                                        │
│  ├─ BeamSearchStrategy                                     │
│  └─ MCTSStrategy                                           │
│                                                            │
│ Common interface:                                          │
│  - add_program(program, iteration)                         │
│  - sample_parent(iteration) → Program                      │
│  - get_context_programs(parent) → (best, inspirations)    │
│  - get_best_program() → Program                            │
│  - get_snapshot() → Dict (for workers)                     │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│ StrategyParallelController                                 │
│  - Manages worker pool (4 processes)                       │
│  - Submits iterations to workers                           │
│  - Collects results and updates strategy                   │
└────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────┐
│ Worker Processes (in parallel)                             │
│  1. Receive: parent + context programs                     │
│  2. Build prompt (shared prompt builder)                   │
│  3. Generate mutation (LLM ensemble)                       │
│  4. Evaluate child (shared evaluator)                      │
│  5. Return: child program + metrics                        │
└────────────────────────────────────────────────────────────┘
```

## Key Differences

| Aspect | MAP-Elites | Best-of-N | Beam Search | MCTS |
|--------|-----------|-----------|-------------|------|
| **Parent Selection** | Island + exploit/explore | Round-robin lineages | UCT from beam | UCT from tree |
| **Program Storage** | Feature grid (per island) | Lineage heads | Beam list | Tree nodes |
| **Context Programs** | Island best + diverse | Lineage heads | Beam members | Tree best + leaves |
| **Diversity** | High (feature grid) | Medium (N lineages) | Low (greedy) | Medium-High (UCT) |
| **Complexity** | High | Low | Medium | High |
| **Best For** | Long runs, diversity | Quick experiments | Focused search | Adaptive search |

## Comparison Study

To compare all strategies:

```bash
# Run all strategies with same parameters
for strategy in "" "--best-of-n" "--beam-search" "--mcts"; do
  python openevolve-run.py \
    examples/math_mas/initial_program.py \
    examples/math_mas/evaluator.py \
    $strategy \
    --iterations 50
done

# Compare results
# Each run saves to: openevolve_output/best/best_program_info.json
```

## Configuration

### Common Configuration (all strategies)

```yaml
# Evolution settings
max_iterations: 50
checkpoint_interval: 5
log_level: "INFO"
random_seed: 42

# LLM configuration
llm:
  primary_model: "gpt-5"         # Evolution model (80%)
  primary_model_weight: 0.8
  secondary_model: "gpt-5-mini"  # Evolution model (20%)
  secondary_model_weight: 0.2
  temperature: 0.8
  max_tokens: 16000

# Prompt configuration
prompt:
  num_top_programs: 3      # Top programs shown to LLM
  num_diverse_programs: 2  # Diverse programs for inspiration
  use_template_stochasticity: true

# Evaluator configuration
evaluator:
  timeout: 600
  cascade_evaluation: true
  cascade_thresholds: [0.3, 0.6]
  parallel_evaluations: 4  # 4 worker processes

# Evolution settings
diff_based_evolution: false  # Use full rewrites
max_code_length: 50000

# Early stopping
early_stopping_patience: 20
early_stopping_metric: "combined_score"
convergence_threshold: 0.01
```

### Strategy-Specific Configuration

**Best-of-N**:
```yaml
search_strategy: "best_of_n"
n_lineages: 4
```

**Beam Search**:
```yaml
search_strategy: "beam_search"
beam_width: 4
branch_factor: 8
```

**MCTS**:
```yaml
search_strategy: "mcts"
expansion_width: 3
exploration_constant: 1.414
```

**MAP-Elites** (additional):
```yaml
database:
  num_islands: 4
  feature_dimensions:
    - "accuracy"
    - "avg_llm_calls"
  feature_bins:
    accuracy: 10
    avg_llm_calls: 8
  migration_interval: 10
  migration_rate: 0.1
```

## Output

All strategies produce the same output structure:

```
openevolve_output/
├── best/
│   ├── best_program.py          # Best program code
│   └── best_program_info.json   # Metrics and metadata
├── checkpoints/
│   ├── checkpoint_5/
│   │   ├── strategy.json        # Strategy state
│   │   ├── best_program.py
│   │   └── best_program_info.json
│   └── checkpoint_10/
│       └── ...
└── logs/
    └── openevolve_YYYYMMDD_HHMMSS.log
```

## Extending with New Strategies

To add a new search strategy:

1. **Create strategy class** in `openevolve/search_strategies/`:
   ```python
   from openevolve.search_strategies.base_strategy import SearchStrategy

   class MyStrategy(SearchStrategy):
       def add_program(self, program, iteration):
           # Store program
           pass

       def sample_parent(self, iteration):
           # Select parent
           pass

       def get_context_programs(self, parent, iteration):
           # Return (best_programs, inspirations)
           pass

       def get_best_program(self):
           # Return best program
           pass

       def get_snapshot(self):
           # Return serializable state
           pass
   ```

2. **Add to registry** in `__init__.py`:
   ```python
   from openevolve.search_strategies.my_strategy import MyStrategy
   __all__ = [..., "MyStrategy"]
   ```

3. **Update controller** in `strategy_controller.py`:
   ```python
   def _create_strategy(self, strategy_name: str):
       if strategy_name == "my_strategy":
           return MyStrategy(self.config)
       # ...
   ```

4. **Add CLI flag** in `cli.py`:
   ```python
   strategy_group.add_argument(
       "--my-strategy",
       action="store_true",
       help="Use My Strategy"
   )
   ```

5. **Create config** file: `config_my_strategy.yaml`

The new strategy automatically inherits:
- Prompt building
- Parallel execution
- LLM ensembles
- Evaluation
- Checkpointing

## Troubleshooting

### Strategy not found
```bash
# Make sure strategy is in __init__.py
python -c "from openevolve.search_strategies import BestOfNStrategy"
```

### Config not auto-selected
```bash
# Explicitly specify config
python openevolve-run.py ... --best-of-n --config config_best_of_n.yaml
```

### Old OpenEvolve not working
```bash
# Default (no strategy flag) should work as before
python openevolve-run.py examples/math_mas/initial_program.py \
  examples/math_mas/evaluator.py \
  --config examples/math_mas/config.yaml
```

If issues occur, it's likely due to missing strategy selection logic in the controller.
