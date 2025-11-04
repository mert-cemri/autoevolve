# Circle Packing: Testing All 4 Strategies with Pre-loaded Best-of-N Memory

## Experiment Setup

**Goal**: Test if memory learned from Best-of-N (30 iterations) helps other strategies when pre-loaded.

**Source Memory**: `openevolve_output_bon_30_with_mem/memory_snapshot.json`
- Contains ~30 iterations of Best-of-N evolution knowledge
- Includes parent-child code pairs, scores, deltas, and embeddings
- All strategies will query this pre-learned knowledge via semantic search

## Configuration

**Config File**: `config_phase_1.yaml`

**Memory Settings** (already configured):
```yaml
memory:
  enabled: true
  snapshot_path: "examples/circle_packing/openevolve_output_bon_30_with_mem/memory_snapshot.json"
  load_from_snapshot: true
```

**Path Note**: If running from `circle_packing/` directory, change to:
```yaml
snapshot_path: "openevolve_output_bon_30_with_mem/memory_snapshot.json"
```

## Run Sequence

### 1. Default OpenEvolve (MAP-Elites) with Best-of-N Memory

**Config**: Leave all strategies commented
```yaml
# All strategies commented = Default OpenEvolve
```

**Command**:
```bash
cd examples/circle_packing
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config_phase_1.yaml \
  --output-dir openevolve_output_open_evolve_with_bon_mem
```

**Expected**: Loads Best-of-N memory, writes to its own output directory

---

### 2. Best-of-N with Best-of-N Memory (Continuation)

**Config**: Uncomment Best-of-N
```yaml
search_strategy: "best_of_n"
n_lineages: 4
```

**Command**:
```bash
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config_phase_1.yaml \
  --output-dir openevolve_output_bon_continuation_with_mem
```

**Expected**: Continues from its own memory + adds new entries

---

### 3. Beam Search with Best-of-N Memory

**Config**: Uncomment Beam Search
```yaml
search_strategy: "beam_search"
beam_width: 2
branch_factor: 4
```

**Command**:
```bash
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config_phase_1.yaml \
  --output-dir openevolve_output_beam_with_bon_mem
```

**Expected**: Loads Best-of-N memory, benefits from cross-strategy knowledge transfer

---

### 4. MCTS with Best-of-N Memory

**Config**: Uncomment MCTS
```yaml
search_strategy: "mcts"
expansion_width: 3
exploration_constant: 1.414
```

**Command**:
```bash
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config_phase_1.yaml \
  --output-dir openevolve_output_mcts_with_bon_mem
```

**Expected**: Loads Best-of-N memory, benefits from cross-strategy knowledge

---

## What Changed in Code

**Files Modified**:
- `openevolve/controller.py`: Support loading from source snapshot, writing to output_dir
- `openevolve/strategy_controller.py`: Same fix for strategy-based runs

**Key Change**: If `snapshot_path` is set and `load_from_snapshot: true`:
- **Loads from**: `snapshot_path` (Best-of-N memory)
- **Writes to**: `output_dir/memory_snapshot.json` (each strategy's own memory)
- **Result**: Best-of-N snapshot is preserved, each strategy accumulates its own memory

## Expected Outcomes

1. **Knowledge Transfer**: Other strategies benefit from Best-of-N's 30 iterations of learning
2. **Faster Convergence**: Strategies start with useful examples instead of empty memory
3. **Better Results**: Memory examples guide evolution from iteration 1

## Verification

After each run, check logs for:
```
Loaded existing memory snapshot: X entries, Y embeddings. Continuing evolution with accumulated knowledge.
Memory store initialized successfully (snapshot: ...)
Memory: Found N similar parent(s) for iteration 1
```

---

## Quick Reference: Config Toggle

**Strategy Selection** (uncomment ONE):
- Line 2-3: Best-of-N
- Line 6-8: Beam Search  
- Line 11-13: MCTS
- All commented: Default OpenEvolve

**Memory Toggle**:
- Line 85: `load_from_snapshot: true` (already set)
- Line 84: `snapshot_path` (already points to Best-of-N)

