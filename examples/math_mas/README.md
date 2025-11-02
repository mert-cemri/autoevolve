# Multi-Agent Math Solving System Evolution

This directory contains scripts for evolving and testing multi-agent systems that solve mathematical problems from the OlympiadBench dataset.

## Quick Start

### Run Evolution (Single Strategy)

```bash
# Run MAP-Elites for 100 iterations with 100 problems
./run_map_elites.sh 100 100

# Run Best-of-N
./run_best_of_n.sh 100 100

# Run Beam Search
./run_beam_search.sh 100 100

# Run MCTS
./run_mcts.sh 100 100
```

### Run All Strategies in Parallel

```bash
# Run all 4 strategies simultaneously
./run_all_strategies.sh 100 100
```

### Test a Program

```bash
# Test initial program with 100 problems (seed=42)
python test_program.py initial_program.py

# Test evolved program with different seed for test set
python test_program.py openevolve_output/best/best_program.py --seed 99

# Test with more problems
python test_program.py path/to/program.py --num-problems 200 --seed 1234
```

---

## Testing Script: test_program.py

Standalone script to evaluate any program on math problems with configurable random seed.

**Usage:**
```bash
python test_program.py <program_path> [options]

Options:
  -n, --num-problems N  Number of problems (default: 100, use -1 for all 675)
  -s, --seed N          Random seed for sampling (default: 42)
  -o, --output FILE     Output JSON file
```

**Examples:**
```bash
# Test with default settings (100 problems, seed=42)
python test_program.py initial_program.py

# Test on DIFFERENT problems (seed=99 instead of 42)
python test_program.py openevolve_output/best/best_program.py --seed 99

# Test on full dataset
python test_program.py path/to/program.py --num-problems -1
```

## Train/Test Split with Seeds

Use different random seeds to create train/test splits:

```bash
# Evolution uses seed=42 (from config.yaml)
./run_map_elites.sh 100 100

# Test on different problems (seed=99)
python test_program.py openevolve_output/best/best_program.py --seed 99
```

See full README for more details.
