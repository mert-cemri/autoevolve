# Function Minimization - Memory Testing Instructions

## Quick Start

```bash
cd examples/function_minimization
python ../../openevolve-run.py initial_program.py evaluator.py --config config_with_strategies.yaml
```

## Strategy Selection

Edit `config_with_strategies.yaml` and **uncomment ONE** strategy section:

### Option 1: Default OpenEvolve (MAP-Elites)
```yaml
# Leave all strategies commented - this is the default
```

### Option 2: Best-of-N
```yaml
search_strategy: "best_of_n"
n_lineages: 4
```

### Option 3: Beam Search
```yaml
search_strategy: "beam_search"
beam_width: 2
branch_factor: 4
```

### Option 4: MCTS
```yaml
search_strategy: "mcts"
expansion_width: 3
exploration_constant: 1.414
```

## Memory Toggle

**Line 113** in config:
```yaml
memory:
  enabled: true   # Change to false to disable memory
```

## Running All Combinations

### 1. Default OpenEvolve
- Comment out all strategies
- Memory: `enabled: true` → Run → Rename output to `openevolve_output_open_evolve_with_mem`
- Memory: `enabled: false` → Run → Rename output to `openevolve_output_open_evolve_no_mem`

### 2. Best-of-N
- Uncomment `search_strategy: "best_of_n"` and `n_lineages: 4`
- Memory: `enabled: true` → Run → Rename output to `openevolve_output_bon_with_mem`
- Memory: `enabled: false` → Run → Rename output to `openevolve_output_bon_no_mem`

### 3. Beam Search
- Uncomment `search_strategy: "beam_search"`, `beam_width: 2`, `branch_factor: 4`
- Memory: `enabled: true` → Run → Rename output to `openevolve_output_beam_with_mem`
- Memory: `enabled: false` → Run → Rename output to `openevolve_output_beam_no_mem`

### 4. MCTS
- Uncomment `search_strategy: "mcts"`, `expansion_width: 3`, `exploration_constant: 1.414`
- Memory: `enabled: true` → Run → Rename output to `openevolve_output_mcts_with_mem`
- Memory: `enabled: false` → Run → Rename output to `openevolve_output_mcts_no_mem`

## Expected Results

- **Target**: Global minimum at (-1.704, 0.678) with value -1.519
- **Baseline**: Random search gets ~0.0 combined_score
- **Good result**: Evolved algorithms should achieve >0.9 combined_score
- **Best result**: Simulated annealing achieves 0.922 combined_score (see README)

## Configuration Details

All settings are based on circle_packing config:
- **Models**: gpt-5-mini (80%) + gpt-5-nano (20%)
- **Temperature**: 0.7
- **Max tokens**: 8192
- **Iterations**: 30 (for testing, increase to 50+ for full runs)
- **Memory**: Semantic search with topk=3, text-embedding-3-large

