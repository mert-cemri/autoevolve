# How Code Evolution Works in Custom Search

## The Big Picture

**Goal**: Evolve a multi-agent system to solve math problems better, not solve the problems directly.

```
┌─────────────────────────────────────────────────────────────┐
│  Search Algorithm (Best-of-N / Beam / MCTS)                │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ 1. Evolution Model (GPT-5)                        │    │
│  │    - Reads: Multi-agent system CODE               │    │
│  │    - Generates: IMPROVED CODE                     │    │
│  │    - Modifies: Agent prompts, protocols, logic    │    │
│  └───────────────────────────────────────────────────┘    │
│                         ↓                                  │
│  ┌───────────────────────────────────────────────────┐    │
│  │ 2. Validation                                      │    │
│  │    - Syntax check (AST parsing)                   │    │
│  │    - Required functions present                   │    │
│  │    - EVOLVE blocks preserved                      │    │
│  └───────────────────────────────────────────────────┘    │
│                         ↓                                  │
│  ┌───────────────────────────────────────────────────┐    │
│  │ 3. Evaluation                                      │    │
│  │    ┌─────────────────────────────────────────┐    │    │
│  │    │ Agent Model (GPT-5-mini)                │    │    │
│  │    │ - Runs the evolved CODE                 │    │    │
│  │    │ - Solves 10 math problems               │    │    │
│  │    │ - Measures: accuracy, efficiency, speed │    │    │
│  │    └─────────────────────────────────────────┘    │    │
│  └───────────────────────────────────────────────────┘    │
│                         ↓                                  │
│  ┌───────────────────────────────────────────────────┐    │
│  │ 4. Selection                                       │    │
│  │    - Keep better code                             │    │
│  │    - Use as parent for next iteration             │    │
│  └───────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## What Gets Evolved?

The code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`:

### 1. Agent Prompts
```python
SOLVER_PROMPT = """You are a mathematical problem solver..."""
VERIFIER_PROMPT = """You are a solution verifier..."""
REVISER_PROMPT = """You are a solution reviser..."""
REFINER_PROMPT = """You are a mathematical refiner..."""
```

**Evolution Examples**:
- Add chain-of-thought reasoning
- Include self-consistency checks
- Add error detection strategies
- Make prompts more specific for math domains

### 2. Communication Protocols
```python
MAX_REVISION_ROUNDS = 2
USE_REFINER = True
REFINER_THRESHOLD = 0.8
```

**Evolution Examples**:
- Reduce revision rounds to save LLM calls
- Adjust when to use refiner vs reviser
- Add early stopping conditions

### 3. Decision Logic
```python
# Control flow between agents
# Error handling strategies
# Information passing between agents
```

## How Evolution Model Knows What to Improve

The evolution prompt includes:

1. **Current Performance**:
   ```
   Score: 0.5250
   Metrics: {
     'accuracy': 0.50,
     'avg_llm_calls': 4.2,
     'efficiency_score': 0.829,
     'time_per_problem': 46.8
   }
   ```

2. **What Matters** (Evaluation Metrics):
   - Accuracy: 70% of score (primary goal)
   - Efficiency: 25% of score (fewer LLM calls = better)
   - Speed: 5% of score (faster = better)

3. **Specific Guidance**:
   - Target: 3 LLM calls per problem (ideal)
   - Penalty: 10+ calls per problem (full penalty)
   - Focus areas: prompts, protocols, logic

4. **Context** (iteration info):
   ```
   Iteration 5/10
   Parent score: 0.5250
   This is branch 2/4 from beam member 1
   ```

## Validation Ensures Executability

Before evaluating any generated code, we check:

```python
def _validate_program(new_code, fallback_code):
    # 1. Parse with AST (catches syntax errors)
    ast.parse(new_code)

    # 2. Check required function exists
    if "def run_evaluation_sample" not in new_code:
        return False

    # 3. Check EVOLVE blocks preserved
    if "# EVOLVE-BLOCK-START" not in new_code:
        return False

    # 4. If any check fails, use parent code
    return True
```

**Benefits**:
- No runtime errors from syntax mistakes
- Ensures program structure is intact
- Falls back to parent if validation fails
- Logs validation failures for debugging

## Evaluation Process

Each generated program is evaluated by:

1. **Write to temp file**: Save the code to `temp_{id}.py`

2. **Load and execute**:
   ```python
   spec = importlib.util.spec_from_file_location("program", temp_file)
   program = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(program)
   ```

3. **Load problems**:
   ```python
   # Math500 dataset has 500 problems
   # We randomly sample N problems with seed=42
   random.seed(42)
   problems = random.sample(all_problems, num_eval_problems)

   # Example with num_eval_problems=10:
   # Problems [42, 157, 89, 301, ...] (always the same 10 with seed=42)
   ```

4. **Run on problems**:
   ```python
   for problem in problems:  # Same 10 problems every iteration
       result = program.run_evaluation_sample(problem)
       # Uses agent_model (GPT-5-mini) internally
   ```

5. **Compute metrics**:
   - Accuracy: % correct answers
   - Efficiency: Based on avg LLM calls (target: 3, penalty: 10+)
   - Speed: Time per problem

6. **Combined score**:
   ```python
   score = 0.70 * accuracy + 0.25 * efficiency_score + 0.05 * time_score
   ```

### Problem Sampling Strategy

```
Math500 Dataset (500 problems)
         ↓ random.seed(42)
         ↓ random.sample(problems, 10)
         ↓
[Problem 42, 157, 89, 301, 8, 445, 223, 91, 367, 129]
         ↓
Same 10 problems used in:
  - Every iteration (0, 1, 2, ..., 10)
  - Every strategy (Best-of-N, Beam, MCTS)
  - Every run (if same seed & num_eval_problems)
         ↓
Benefits:
  ✅ Fair comparison across strategies
  ✅ Reproducible results
  ✅ Fast evaluation (10 problems)

Considerations:
  ⚠️  Risk of overfitting to these 10 problems
  ⚠️  Limited coverage (2% of dataset)
  ✅ Can increase to 50, 100, or all 500 problems
```

## Why Two Models?

### Evolution Model (e.g., GPT-5)
- **Used for**: Generating improved code
- **Calls**: ~100-200 times during search
- **Why powerful**: Needs to understand code, design patterns, optimization
- **Cost**: Expensive but used sparingly

### Agent Model (e.g., GPT-5-mini)
- **Used for**: Solving math problems
- **Calls**: ~3-10 times per problem × 10 problems × 100+ iterations = thousands
- **Why cheaper**: Many calls needed for evaluation
- **Cost**: Much cheaper, can afford many calls

**Total Cost Breakdown**:
```
Evolution Model: 100 calls × $X = $100X
Agent Model:     10,000 calls × $Y = $10,000Y

If Y = X/10 (GPT-5-mini vs GPT-5):
Total = $100X + $1,000X = $1,100X

Without separation (using GPT-5 for both):
Total = $100X + $10,000X = $10,100X  (9× more expensive!)
```

## Common Issues & Solutions

### Issue 1: "Generated code has syntax errors"
**Solution**: AST validation catches this before evaluation
```python
try:
    ast.parse(new_code)
except SyntaxError:
    logger.warning("Syntax error, using parent code")
    return parent_code
```

### Issue 2: "Evolution model removes critical code"
**Solution**: Prompt emphasizes keeping non-EVOLVE code unchanged
```
CRITICAL RULES:
1. Keep ALL code outside EVOLVE blocks EXACTLY the same
2. Maintain the same function signatures
```

### Issue 3: "No clear improvement direction"
**Solution**: Detailed prompt with performance metrics and specific guidance
```
Current program performance:
- Score: 0.5250
- Accuracy: 50% (need to improve)
- Avg LLM calls: 4.2 (target: 3)

Focus on:
- Enhancing prompts for better reasoning
- Reducing unnecessary LLM calls
```

### Issue 4: "Programs fail during evaluation"
**Solution**:
- Validation checks for required functions
- Try-catch in evaluator
- Return 0 score if program crashes

### Issue 5: "Evolution gets stuck"
**Solution**: Different search strategies explore differently
- Best-of-N: Multiple independent lineages (diversity)
- Beam Search: Maintains beam of top programs (exploration)
- MCTS: UCT selection balances exploration/exploitation

## Verification Steps

To verify evolution is working:

1. **Check logs for validation**:
   ```
   [Iteration 1, Lineage 1/4] Generating variant...
   LLM response received (12450 chars)
   ✓ Generated code validated successfully
   ```

2. **Check iteration files**:
   ```bash
   ls custom_search/results/best_of_n/run_*/
   # Should see: iteration_0000_best.py, iteration_0001_best.py, etc.
   ```

3. **Compare programs**:
   ```bash
   diff iteration_0000_best.py iteration_0005_best.py
   # Should show evolved prompts/protocols in EVOLVE blocks
   ```

4. **Check metrics improve**:
   ```bash
   cat iteration_summary.json
   # Should see increasing scores over iterations
   ```

## Example: What Good Evolution Looks Like

**Iteration 0** (Initial):
```python
SOLVER_PROMPT = """You are a mathematical problem solver.
Solve the problem step-by-step."""
```
Score: 0.50, Avg calls: 5.2

**Iteration 3** (Evolved):
```python
SOLVER_PROMPT = """You are a mathematical problem solver.
1. First, identify the problem type (algebra, geometry, etc.)
2. Break down into sub-problems
3. Solve each step with clear reasoning
4. Verify your answer makes sense
Provide final answer in \\boxed{} format."""
```
Score: 0.62, Avg calls: 4.1

**Iteration 7** (Further evolved):
```python
SOLVER_PROMPT = """You are an expert mathematical problem solver.

STRATEGY:
1. Classify problem (algebra/geometry/calculus/combinatorics)
2. Recall relevant formulas and theorems
3. Plan solution approach before computing
4. Execute with explicit intermediate steps
5. Sanity-check: Does answer make sense given problem context?

FORMAT: Provide final answer as \\boxed{numerical_value}"""

MAX_REVISION_ROUNDS = 1  # Reduced from 2 to save LLM calls
```
Score: 0.71, Avg calls: 3.2

## Summary

✅ **Evolution model** generates improved code
✅ **Validation** ensures code is executable
✅ **Agent model** tests code on math problems
✅ **Metrics** guide next evolution
✅ **Search strategies** explore solution space
✅ **Result**: Better multi-agent systems over time
