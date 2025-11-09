# Multi-Agent Architecture Evolution Guide

## Problem Solved

**Original Issue:** Evolution wasn't working properly because:
1. Timeouts were marked as success (OpenEvolve bug) - FIXED ✅
2. EVOLVE-BLOCK included framework code, causing evolved programs to break (0 LLM calls) - FIXED ✅
3. Only prompts could evolve, not the multi-agent architecture itself - FIXED ✅

## New Design: `initial_program_v2.py`

### Structure

```python
# FIXED FRAMEWORK (outside EVOLVE-BLOCK)
├── call_llm()              # LLM calling function
├── run_evaluation_sample() # Evaluation interface
└── Logging functions       # Save conversation logs

# EVOLVABLE (inside EVOLVE-BLOCK)
├── Agent Prompts           # SOLVER_PROMPT, VERIFIER_PROMPT, etc.
├── AgentConfig             # MAX_REVISION_ROUNDS, USE_REFINER, etc.
├── Agent Functions         # solver_agent(), verifier_agent(), etc.
├── solve_problem()         # Main orchestration logic ⭐
├── extract_decision()      # Decision parsing
└── extract_boxed_answer()  # Answer extraction
```

### What Can OpenEvolve Evolve Now?

#### 1. Agent Prompts
Modify instructions for each agent:
```python
SOLVER_PROMPT = """You are a mathematical problem solver...
Step 1: Understand the problem
Step 2: Identify key concepts
..."""
```

#### 2. Agent Architecture
Add/remove agents, change roles:
```python
def planner_agent(problem: str) -> str:
    """New agent: Plan solution strategy"""
    return call_llm(PLANNER_PROMPT, problem, agent_role="planner")
```

#### 3. Communication Protocols
Modify AgentConfig parameters:
```python
class AgentConfig:
    MAX_REVISION_ROUNDS = 3  # Increase for harder problems
    USE_VERIFIER = True       # Enable/disable verification
    USE_REFINER = False       # Speed up by skipping refinement
```

#### 4. Orchestration Logic
The most powerful: modify `solve_problem()` to change architecture:

**Example 1: Sequential (current)**
```python
def solve_problem(problem: str) -> Dict[str, str]:
    solution = solver_agent(problem)
    verification = verifier_agent(problem, solution)
    if needs_revision:
        solution = reviser_agent(problem, solution, verification)
    if USE_REFINER:
        solution = refiner_agent(problem, solution)
    return format_result(solution)
```

**Example 2: Iterative Loop**
```python
def solve_problem(problem: str) -> Dict[str, str]:
    solution = solver_agent(problem)
    for _ in range(MAX_REVISION_ROUNDS):
        verification = verifier_agent(problem, solution)
        if "APPROVED" in verification:
            break
        solution = reviser_agent(problem, solution, verification)
    return format_result(solution)
```

**Example 3: Hierarchical**
```python
def solve_problem(problem: str) -> Dict[str, str]:
    plan = planner_agent(problem)  # Plan solution
    solution = solver_agent(f"{problem}\n\nPlan: {plan}")  # Solve with plan
    verification = verifier_agent(problem, solution)
    if needs_revision:
        solution = reviser_agent(problem, solution, verification)
    return format_result(solution)
```

**Example 4: Parallel Voting**
```python
def solve_problem(problem: str) -> Dict[str, str]:
    # Get 3 independent solutions
    solutions = [solver_agent(problem) for _ in range(3)]
    
    # Voter selects best
    vote_prompt = f"Problem: {problem}\nSolutions:\n" + "\n\n".join(solutions)
    best_solution = voter_agent(vote_prompt)
    
    # Refine
    if USE_REFINER:
        best_solution = refiner_agent(problem, best_solution)
    
    return format_result(best_solution)
```

## Key Differences from v1

| Aspect | v1 (initial_program.py) | v2 (initial_program_v2.py) |
|--------|------------------------|----------------------------|
| **EVOLVE-BLOCK Size** | Too large (lines 1-233) | Minimal framework, large evolvable zone |
| **Framework Safety** | Framework could be deleted | Framework is protected (outside EVOLVE-BLOCK) |
| **Architecture Evolution** | ❌ Only prompts | ✅ Full architecture |
| **Agent Communication** | ❌ Fixed | ✅ Fully evolvable |
| **Number of Agents** | ❌ Fixed (4) | ✅ Can add/remove |
| **Orchestration Logic** | ❌ Fixed | ✅ Fully evolvable |
| **Evolved Programs** | Often broken (0 LLM calls) | Always functional |

## Migration to v2

### ✅ Migration Complete!

The `initial_program.py` has been updated to v2 with:
- Protected framework (lines 1-192) - cannot be deleted by evolution
- Evolvable architecture (lines 193-394) - full multi-agent system

The old version is backed up as `initial_program_v1_backup.py` if needed.

### Running Evolution

Simply use the shell scripts:

```bash
# Run evolution (100 iterations, 20 questions per evaluation)
./run_map_elites.sh 100 20

# Or use different search strategies
./run_best_of_n.sh 100 20
./run_beam_search.sh 100 20
./run_mcts.sh 100 20
```

## Updated Config

The `config.yaml` system prompt now includes:
- Detailed guidance on what can be evolved
- Architecture patterns to explore (Sequential, Iterative, Hierarchical, Parallel)
- Optimization strategies based on metrics
- Clear constraints (what NOT to modify)

## Expected Evolution Patterns

### Early Iterations (1-20)
- Prompt improvements
- Parameter tuning (MAX_REVISION_ROUNDS, thresholds)
- Minor orchestration changes

### Mid Iterations (20-60)
- Agent role specialization
- Communication protocol improvements
- Early stopping logic
- Better decision extraction

### Late Iterations (60-100)
- Novel architectures (parallel voting, hierarchical planning)
- Adaptive strategies (change behavior based on problem type)
- Efficiency optimizations (smart early stopping)
- Hybrid approaches (combine multiple strategies)

## Verification

### Check v2 Works

```bash
conda activate autoevolve
cd examples/math_mas

# Test v2 directly
python initial_program_v2.py

# Should output:
# Testing Multi-Agent Math Solver
# Problem: What is the value of $3! + 4! + 5!$?
# ...
# Extracted Answer: 150
# LLM Calls: 4-7
```

### Monitor Evolution

```bash
# Watch logs
tail -f openevolve_output/logs/openevolve_*.log

# Should see:
# - LLM calls > 0 (not 0!)
# - Real accuracy scores (not always 1.0 or 0.0)
# - Variety of architectures being tried
```

### Check Evolved Programs

```bash
# Inspect best program
cat openevolve_output/best/best_program.py | grep -A 20 "def solve_problem"

# Should see:
# - Modified orchestration logic
# - Different agent calling patterns
# - New parameters or strategies
```

## Troubleshooting

### If evolved programs still have 0 LLM calls

**Cause:** OpenEvolve removed too much code

**Fix:** Check EVOLVE-BLOCK boundaries:
```bash
grep -n "EVOLVE-BLOCK" initial_program_v2.py
# Should see:
# Line ~75: EVOLVE-BLOCK-START
# Line ~290: EVOLVE-BLOCK-END
```

Make sure framework code is OUTSIDE these boundaries.

### If all programs timeout

**Cause:** Too many LLM calls or slow responses

**Fix 1:** Reduce complexity in initial program:
```python
class AgentConfig:
    MAX_REVISION_ROUNDS = 1  # Instead of 2
    USE_REFINER = False      # Disable for testing
```

**Fix 2:** Increase timeout further:
```yaml
evaluator:
  timeout: 3600  # 1 hour
```

### If accuracy stays low

**Cause:** Architecture not suitable for hard problems

**Solutions:**
1. Improve solver prompt (add problem decomposition)
2. Increase MAX_REVISION_ROUNDS
3. Add planning agent
4. Use multiple verification passes

## Summary

✅ **Framework Protected:** Core functions can't be deleted by evolution
✅ **Full Architecture Evolution:** Can modify agents, protocols, orchestration
✅ **Functional Guarantees:** Evolved programs always call LLMs properly
✅ **Guided Evolution:** Config prompt explains what to evolve and how
✅ **Multiple Strategies:** Can evolve toward different architectures

Evolution can now discover novel multi-agent architectures for solving hard math problems!
