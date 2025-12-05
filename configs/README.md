# OpenEvolve Configuration Files

This directory contains configuration files for OpenEvolve with examples for different use cases.

## Configuration Files

### `default_config.yaml`
The main configuration file containing all available options with sensible defaults. This file includes:
- Complete documentation for all configuration parameters
- Default values for all settings
- **Island-based evolution parameters** for proper evolutionary diversity

Use this file as a template for your own configurations.

### `island_config_example.yaml`
A practical example configuration demonstrating proper island-based evolution setup. Shows:
- Recommended island settings for most use cases
- Balanced migration parameters
- Complete working configuration

### `island_examples.yaml`
Multiple example configurations for different scenarios:
- **Maximum Diversity**: Many islands, frequent migration
- **Focused Exploration**: Few islands, rare migration  
- **Balanced Approach**: Default recommended settings
- **Quick Exploration**: Small-scale rapid testing
- **Large-Scale Evolution**: Complex optimization runs

Includes guidelines for choosing parameters based on your problem characteristics.

## 🚀 AutoEvolve Advanced Features

> **AutoEvolve** is a [Sky Computing Lab, UC Berkeley](https://sky.cs.berkeley.edu/) research project that extends OpenEvolve with advanced adaptive mechanisms. The core insight: LLM-driven evolution can achieve breakthrough performance when guided by **stagnation-aware paradigm generation** and **failure-informed search adaptation**.

All advanced features are **disabled by default**. Add these to your config to enable:

### Quick Start: Enable All Features

```yaml
database:
  # === PARADIGM BREAKTHROUGH ===
  # Auto-generates breakthrough ideas when evolution stagnates
  enable_paradigm_breakthrough: true
  stagnation_window: 5
  stagnation_improvement_threshold: 0.01
  stagnation_paradigm_samples: 3
  paradigm_model: "gpt-5-mini"

  # === ERROR RETRY ===
  # Retries failed generations/evaluations with error context
  enable_error_retry: true
  max_error_retries: 2

  # === ADAPTIVE SEARCH ===
  # Dynamically adjusts exploration/exploitation ratio
  use_adaptive_search: true
  adaptive_window_size: 20
  adaptive_min_exploration: 0.1
  adaptive_max_exploration: 0.7

  # === SOFTMAX EXPLOITATION ===
  exploitation_temperature: 1.0

  # === STAGNATION MULTI-CHILD ===
  stagnation_threshold: 10
  stagnation_multi_child_count: 3
  sibling_context_limit: 5
```

### Feature Summary

| Feature | Enable With | What It Does |
|---------|-------------|--------------|
| Paradigm Breakthrough | `enable_paradigm_breakthrough: true` | LLM generates breakthrough ideas on stagnation |
| Error Retry | `enable_error_retry: true` | Auto-retry failed generations (2 retries = 3 attempts) |
| Adaptive Search | `use_adaptive_search: true` | Dynamic exploration/exploitation ratio |

---

## Island-Based Evolution Parameters

The key new parameters for proper evolutionary diversity are:

```yaml
database:
  num_islands: 5                      # Number of separate populations
  migration_interval: 50              # Migrate every N generations  
  migration_rate: 0.1                 # Fraction of top programs to migrate
```

### Parameter Guidelines

- **num_islands**: 3-10 for most problems (more = more diversity)
- **migration_interval**: 25-100 generations (higher = more independence)
- **migration_rate**: 0.05-0.2 (5%-20%, higher = faster knowledge sharing)

### When to Use What

- **Complex problems** → More islands, less frequent migration
- **Simple problems** → Fewer islands, more frequent migration
- **Long runs** → More islands to maintain diversity
- **Short runs** → Fewer islands for faster convergence

## Usage

Copy any of these files as a starting point for your configuration:

```bash
cp configs/default_config.yaml my_config.yaml
# Edit my_config.yaml for your specific needs
```

Then use with OpenEvolve:

```python
from openevolve import OpenEvolve
evolve = OpenEvolve(
    initial_program_path="program.py",
    evaluation_file="evaluator.py", 
    config_path="my_config.yaml"
)
```
