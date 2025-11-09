# Circle Packing Experiments

This directory contains experiment scripts for comparing different search strategies on the circle packing problem (n=26 circles in a unit square).

## Problem Overview

**Objective**: Pack 26 non-overlapping circles into a unit square to maximize the sum of their radii.

**Target**: 2.635 (from AlphaEvolve paper)

**Constraints**:
- All circles must fit entirely within the unit square (0 ≤ x, y ≤ 1)
- No circles may overlap
- Exactly 26 circles must be placed

## Search Strategies

We compare 4 different search strategies:

1. **MAP-Elites**: Quality-diversity algorithm with island-based evolution
2. **Best-of-N**: Maintains N independent lineages, keeps the best
3. **Beam Search**: Keeps top M programs, generates N candidates per iteration
4. **MCTS**: Monte Carlo Tree Search with UCT exploration

## Quick Start

### Run Individual Strategies

```bash
# MAP-Elites (100 iterations)
./run_map_elites.sh 100

# Best-of-N (4 lineages)
./run_best_of_n.sh 100

# Beam Search (beam_width=4, branch_factor=8)
./run_beam_search.sh 100

# MCTS (expansion_width=3)
./run_mcts.sh 100
```

### Test a Program

```bash
# Test the initial program
python test_program.py initial_program.py

# Test an evolved program
python test_program.py openevolve_output/best/best_program.py

# Test without visualization
python test_program.py initial_program.py --no-visualize

# Compare multiple programs
python test_program.py initial_program.py best_program.py
```

## Configuration Files

Each search strategy has a dedicated config file:

- `config_map_elites.yaml` - MAP-Elites with 2D feature grid (sum_radii × eval_time)
- `config_best_of_n.yaml` - Best-of-N with 4 lineages
- `config_beam_search.yaml` - Beam Search with beam_width=4, branch_factor=8
- `config_mcts.yaml` - MCTS with expansion_width=3, exploration_constant=√2

### Key Configuration Parameters

All configs share:
```yaml
max_iterations: 100
checkpoint_interval: 10
random_seed: 42
llm:
  primary_model: "gpt-5"
  temperature: 0.7
  max_tokens: 8192
evaluator:
  timeout: 600  # 10 minutes per evaluation
  cascade_evaluation: true
  cascade_thresholds: [0.5, 0.75]
```

### MAP-Elites Specific

```yaml
database:
  population_size: 60
  num_islands: 4
  feature_dimensions:
    - "sum_radii"    # Maximize this (0.0 to 2.7)
    - "eval_time"    # Minimize this (faster is better)
  feature_bins:
    sum_radii: 20
    eval_time: 10
```

## Expected Evolution Patterns

### Early Iterations (0-20)
- Simple geometric patterns (concentric rings, grids)
- Sum of radii: 0.5 - 1.5
- Quick evaluations (<1s)

### Mid Iterations (20-60)
- Hexagonal arrangements emerge
- Variable-sized circles
- Sum of radii: 1.5 - 2.2
- Evaluation times increase (1-5s)

### Late Iterations (60-100)
- Mathematical optimization (scipy.optimize)
- Hybrid approaches
- Sum of radii: 2.2 - 2.6+
- Longer evaluations (5-60s)

## Metrics Tracked

All evaluations return:

```python
{
    "sum_radii": 2.634,          # Primary objective
    "target_ratio": 0.9996,      # sum_radii / 2.635
    "validity": 1.0,             # 1.0 if valid, 0.0 if invalid
    "eval_time": 12.5,           # Seconds to evaluate
    "combined_score": 0.9996     # target_ratio * validity
}
```

## Output Structure

Each strategy creates its own output directory:

```
openevolve_output/
├── best/                          # MAP-Elites results
│   ├── best_program.py
│   └── best_program_info.json
├── checkpoints/
│   ├── checkpoint_10/
│   ├── checkpoint_20/
│   └── ...
├── best_of_n/                     # Best-of-N results
│   ├── best/
│   └── checkpoints/
├── beam_search/                   # Beam Search results
│   ├── best/
│   └── checkpoints/
└── mcts/                          # MCTS results
    ├── best/
    └── checkpoints/
```

## Analyzing Results

### View Best Program

```bash
# Display best program code
cat openevolve_output/best/best_program.py

# View metrics
cat openevolve_output/best/best_program_info.json
```

### Visualize Best Solution

```python
from openevolve_output.best.best_program import run_packing, visualize

centers, radii, sum_radii = run_packing()
print(f"Sum of radii: {sum_radii}")
print(f"Target ratio: {sum_radii / 2.635:.2%}")
visualize(centers, radii)
```

### Compare Strategies

```bash
# Test all best programs
python test_program.py \
  openevolve_output/best/best_program.py \
  openevolve_output/best_of_n/best/best_program.py \
  openevolve_output/beam_search/best/best_program.py \
  openevolve_output/mcts/best/best_program.py
```

## Resuming from Checkpoint

You can resume evolution from any checkpoint:

```bash
# Resume MAP-Elites from iteration 50
python ../../openevolve-run.py \
  initial_program.py \
  evaluator.py \
  --config config_map_elites.yaml \
  --checkpoint openevolve_output/checkpoints/checkpoint_50 \
  --iterations 100  # Run 100 MORE iterations
```

## Two-Phase Evolution (Advanced)

For breaking through plateaus, run a two-phase evolution:

### Phase 1: Exploration (100 iterations)
```bash
./run_map_elites.sh 100
```

### Phase 2: Exploitation (100 more iterations)

Modify the config to encourage more aggressive optimization:

```yaml
# Phase 2 adjustments
database:
  population_size: 70      # More diversity
  num_islands: 5           # More parallel exploration
  exploitation_ratio: 0.6  # More exploration

prompt:
  system_message: |
    Focus on breaking through the plateau by trying fundamentally
    different approaches. Consider:
    - scipy.optimize for mathematical optimization
    - Hybrid geometric + numerical approaches
    - Variable circle sizes with strategic placement
```

Then resume:
```bash
python ../../openevolve-run.py \
  openevolve_output/checkpoints/checkpoint_100/best_program.py \
  evaluator.py \
  --config config_phase_2.yaml \
  --iterations 100
```

## Troubleshooting

### Programs timeout during evaluation

Increase the timeout:
```yaml
evaluator:
  timeout: 1200  # 20 minutes instead of 10
```

### Evolution converges too quickly

Increase diversity:
```yaml
database:
  population_size: 80
  num_islands: 6
  exploration_ratio: 0.4  # More exploration
```

### Low quality solutions

The LLM might need better guidance:
```yaml
prompt:
  system_message: |
    # Add specific strategies and examples
    # Emphasize use of scipy.optimize
    # Provide geometric insights
```

## Expected Runtime

**Per Strategy** (100 iterations):
- Total time: ~12-24 hours
- Per iteration: ~5-15 minutes
- Parallel evaluations: 4 concurrent
- Checkpoint every 10 iterations

**All 4 Strategies** (run in parallel):
- Total time: ~12-24 hours
- Resource usage: 16 concurrent evaluations

## Success Criteria

A successful evolution should achieve:

- ✅ **Valid packing**: No overlaps, all circles inside square
- ✅ **Sum of radii ≥ 2.5**: Getting close to target
- ✅ **Target ratio ≥ 95%**: Within 5% of AlphaEvolve result
- 🎯 **Sum of radii ≥ 2.63**: Matching or beating AlphaEvolve

## Research Questions

These experiments are designed to answer:

1. **Which search strategy performs best?**
   - Compare final sum_radii across strategies
   - Analyze convergence speed

2. **What algorithmic discoveries emerge?**
   - Geometric constructions vs numerical optimization
   - Hybrid approaches

3. **How does diversity help?**
   - MAP-Elites with feature grid vs single-objective
   - Island-based evolution benefits

4. **What is the role of LLM temperature?**
   - Higher temp = more exploration
   - Lower temp = more exploitation

## Next Steps

After running experiments:

1. **Analyze logs**: Check `openevolve_output/*/logs/`
2. **Compare strategies**: Use `test_program.py` for side-by-side comparison
3. **Visualize evolution**: Plot sum_radii over iterations
4. **Extract insights**: What patterns led to breakthroughs?
5. **Iterate**: Run phase 2 with best strategy

## Citation

If you use these experiments, please cite the OpenEvolve paper and the original AlphaEvolve work:

```
@article{alphaevolve2024,
  title={AlphaEvolve: Evolutionary Code Generation},
  author={DeepMind Team},
  journal={Nature},
  year={2024}
}
```
