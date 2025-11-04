# Memory Branch - Complete Changes Summary

This branch contains all memory integration and search strategy improvements made after cloning from main. This is a **separate version** - no merge with main is intended.

## Branch Created From
- **Base**: Main branch (commit: 5edde31 - "upload olympiadbench")
- **Branch Name**: `memory`
- **Purpose**: Clean snapshot of all memory integration and search strategy work

---

## Modified Files (12 files)

### 1. **openevolve/config.py**
- Added `beam_width`, `branch_factor`, `n_lineages`, `search_strategy` fields to `Config` dataclass
- Added `expansion_width` and `exploration_constant` for MCTS support
- Updated `from_dict()` to parse strategy-specific parameters from YAML

### 2. **openevolve/controller.py**
- Memory initialization: supports `snapshot_path` for loading and separate output path for writing
- Fixed: When loading from snapshot, source snapshot is NOT overwritten (writes to output_dir)
- Memory store wiring to parallel controller
- Support for `load_from_snapshot` and `snapshot_path` configuration

### 3. **openevolve/strategy_controller.py**
- Same memory snapshot loading/writing logic as controller.py
- Ensures source snapshots are not overwritten

### 4. **openevolve/strategy_parallel.py**
- Memory integration for all search strategies (Best-of-N, Beam Search, MCTS)
- Semantic parent search before each evolution step
- Memory logging for all results (success/failure/timeout)
- Strategy snapshot support with memory snapshots

### 5. **openevolve/process_parallel.py**
- Memory store integration for default OpenEvolve
- Semantic parent search and logging
- Memory entry creation for all evolution steps

### 6. **openevolve/prompt/sampler.py**
- Similar parent changes formatting from memory
- Integration with memory search results
- Prompt enrichment with past evolution experiences

### 7. **openevolve/search_strategies/best_of_n_strategy.py**
- Complete Best-of-N implementation with N independent lineages
- Round-robin parent sampling
- Memory integration (semantic search, logging)
- Fixed: Lineage initialization and parent tracking bugs

### 8. **openevolve/search_strategies/beam_search_strategy.py**
- Complete Beam Search implementation
- Beam width and branch factor configuration
- Fixed: Deadlock when not all branches succeed
- Fixed: Premature beam selection bug
- Memory integration
- Comprehensive logging for beam cycles

### 9. **openevolve/search_strategies/mcts_strategy.py**
- Complete MCTS (Monte Carlo Tree Search) implementation
- UCT (Upper Confidence Bound for Trees) selection
- Expansion width configuration
- Fixed: `uct_value()` for root node (parent is None)
- Memory integration
- Comprehensive initialization logging

### 10. **openevolve/cli.py**
- Support for strategy-based runs
- Configuration parsing for all strategies

### 11. **memory/in_memory.py**
- `load_from_snapshot()` method for loading pre-existing memory snapshots
- Embedding loading from `memory_embeddings.json`
- Semantic search improvements
- Snapshot path handling (load vs write separation)

### 12. **memory/ui_app.py**
- UI improvements for memory visualization

---

## New Files (3 files)

### 1. **examples/circle_packing/MEMORY_SNAPSHOT_RUNS.md**
- Instructions for running all 4 strategies with pre-loaded memory snapshot
- Step-by-step guide for warm-start experiments

### 2. **examples/function_minimization/config_with_strategies.yaml**
- Complete configuration for testing all strategies with/without memory
- Commented blocks for easy switching between strategies
- Memory toggle (true/false)

### 3. **examples/function_minimization/RUN_INSTRUCTIONS.md**
- Instructions for running function minimization with all strategies
- Memory configuration guide

---

## Modified Configuration

### **examples/circle_packing/config_phase_1.yaml**
- Added commented strategy blocks (Best-of-N, Beam Search, MCTS)
- Memory configuration with snapshot loading
- `load_from_snapshot: true` (boolean, not "yes")
- `snapshot_path` pointing to Best-of-N memory snapshot

---

## Key Features Added

### 1. **Memory Integration**
- Semantic memory store for all evolution steps
- Embedding-based parent code similarity search
- Memory snapshot loading/saving
- Failure tracking (errors, timeouts)
- Non-blocking memory operations

### 2. **Search Strategies**
- **Best-of-N**: N independent lineages, round-robin evolution
- **Beam Search**: Beam-based search with branch factor
- **MCTS**: Monte Carlo Tree Search with UCT selection
- **Default OpenEvolve**: MAP-Elites (unchanged, but with memory support)

### 3. **Memory Snapshot System**
- Load pre-existing memory snapshots for warm-start
- Save new snapshots to output directory
- Source snapshots are never overwritten
- Embeddings stored separately in `memory_embeddings.json`

### 4. **Bug Fixes**
- Beam Search deadlock when branches fail
- Beam Search premature beam selection
- MCTS root node UCT calculation
- Config parameter loading for all strategies
- Memory snapshot path handling

---

## Testing & Validation

All strategies tested with:
- Circle packing example (30 iterations)
- Memory enabled/disabled comparisons
- Warm-start experiments (loading Best-of-N memory)
- Fair comparison across all 4 strategies

---

## Notes

- **No merge intended**: This branch is a standalone version
- **All strategies support memory**: Best-of-N, Beam Search, MCTS, OpenEvolve
- **Memory is optional**: Can be enabled/disabled via config
- **Backward compatible**: Default OpenEvolve behavior unchanged

---

## Commit Message

```
Memory integration and search strategies

- Added memory integration for all search strategies (Best-of-N, Beam Search, MCTS, OpenEvolve)
- Implemented complete Best-of-N, Beam Search, and MCTS strategies
- Fixed critical bugs: beam search deadlock, premature beam selection, MCTS root node
- Added memory snapshot loading/saving with source snapshot protection
- Enhanced configuration parsing for strategy-specific parameters
- Added warm-start experiments documentation
- All strategies tested with circle_packing and function_minimization examples
```

