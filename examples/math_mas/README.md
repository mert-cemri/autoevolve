# Multi-Agent Math Solving System Evolution

This directory contains scripts for evolving and testing multi-agent systems that solve mathematical problems from the OlympiadBench dataset.

## Prerequisites

**IMPORTANT**: All scripts must be run in the `autoevolve` conda environment:

```bash
conda activate autoevolve
```

The shell scripts (`.sh` files) automatically activate this environment, but if you run Python scripts directly, make sure you're in the correct environment first.

## Quick Start

### Run Evolution (Single Strategy)

```bash
# Run MAP-Elites for 100 iterations with 20 problems
./run_map_elites.sh 100 20

# Run Best-of-N
./run_best_of_n.sh 100 20

# Run Beam Search
./run_beam_search.sh 100 20

# Run MCTS
./run_mcts.sh 100 20
```

### Run All Strategies in Parallel

```bash
# Run all 4 strategies simultaneously (if script exists)
# ./run_all_strategies.sh 100 20
```

### Test a Program

```bash
# Test initial program with 20 problems (seed=42)
python test_program.py initial_program.py -n 20

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
  -n, --num-problems N  Number of problems (default: 20, use -1 for all 675)
  -s, --seed N          Random seed for sampling (default: 42)
  -o, --output FILE     Output JSON file
```

**Examples:**
```bash
# Test with default settings (20 problems, seed=42)
python test_program.py initial_program.py

# Test on DIFFERENT problems (seed=99 instead of 42)
python test_program.py openevolve_output/best/best_program.py --seed 99 -n 20

# Test on full dataset
python test_program.py path/to/program.py --num-problems -1
```

## Train/Test Split with Seeds

Use different random seeds to create train/test splits:

```bash
# Evolution uses seed=42 (from config.yaml)
./run_map_elites.sh 100 20

# Test on different problems (seed=99)
python test_program.py openevolve_output/best/best_program.py --seed 99
```

See full README for more details.
