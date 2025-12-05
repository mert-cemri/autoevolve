#!/usr/bin/env python3
"""
Breakthrough Paradigm Generator for OpenEvolve
==============================================

Generates breakthrough optimization ideas by analyzing:
  - Problem objective (from config.yaml)
  - Evaluation logic (from evaluator.py)  
  - Current solution (from best_program.py or initial_program.py)

Uses a structured 6-step analysis framework to identify improvement opportunities
and produces actionable, library-specific implementation guidance.

Output: JSONL file with paradigm ideas for evolutionary code improvement.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import yaml


# =============================================================================
# CONFIGURATION LOADING
# =============================================================================

def load_paradigms_config() -> Optional[Dict[str, Any]]:
    """
    Load modular paradigms configuration from YAML.
    
    Config categories:
      - problem_type_techniques: Condition-based guidance for problem types
      - library_recommendations: Which libraries/functions to use
      - anti_patterns: Critical rules about what NOT to do
      - diversity_requirements: Rules for generating diverse ideas
      - output_format: Structure for generated paradigms
    
    Returns:
        Config dict, or None if not found
    """
    config_paths = [
        Path(__file__).parent / "openevolve" / "prompts" / "paradigms_config.yaml",
        Path(__file__).parent.parent / "openevolve" / "prompts" / "paradigms_config.yaml",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Failed to load paradigms config: {e}")
                return None
    return None


def format_techniques_from_config(config: Dict[str, Any]) -> str:
    """Format problem type techniques from config for prompt injection."""
    techniques_config = config.get("problem_type_techniques", {})
    if not techniques_config.get("enabled", True):
        return ""
    
    lines = []
    for tech in techniques_config.get("techniques", []):
        condition = tech.get("condition", "")
        guidance = " ".join(tech.get("guidance", "").split())
        if condition and guidance:
            lines.append(f"- **{condition}:** {guidance}")
    return "\n".join(lines)


def format_libraries_from_config(config: Dict[str, Any]) -> str:
    """Format library recommendations from config for prompt injection."""
    lib_config = config.get("library_recommendations", {})
    if not lib_config.get("enabled", True):
        return ""
    
    lines = ["**Effective techniques to consider:**"]
    for rec in lib_config.get("recommended", []):
        name, use_for = rec.get("name", ""), rec.get("use_for", "")
        if name and use_for:
            lines.append(f"  * {name} for {use_for}")
    return "\n".join(lines)


# =============================================================================
# PROMPT BUILDING - CONTEXT SECTIONS
# =============================================================================

def build_problem_context(system_message: str, evaluator_code: str, 
                          program_code: str, program_label: str) -> str:
    """Build the problem context section of the prompt."""
    return f"""Analyze this problem and suggest breakthrough ideas that can improve the score.

## Problem Objective:
{system_message}

## Evaluator Code (shows how solution is scored):
```python
{evaluator_code}
```

## {program_label} (needs improvement):
```python
{program_code if program_code else "N/A"}
```"""


def build_current_program_analysis(best_score: float) -> str:
    """Build the current program analysis directive."""
    return f"""
**CRITICAL: ANALYZE THE CURRENT PROGRAM FIRST**
Before suggesting new ideas, carefully analyze the Current Program above:
- What algorithm/approach does it use? (This is what's WORKING - score {best_score:.6f})
- What are its strengths? (Why does it achieve this score?)
- What are its weaknesses? (What limits further improvement?)
- How can you improve it? (How to beat it?)

**IMPORTANT:** The program above is the CURRENT program that needs to be improved. Start by understanding what works, then suggest breakthrough ideas that build on or improve it."""


def build_previously_tried_section(previously_tried_ideas: Optional[List[str]]) -> str:
    """Build the previously tried ideas section."""
    if not previously_tried_ideas:
        return ""
    
    # Check if using new format (approach type grouping)
    if any("## CRITICAL" in idea or "Successful Approach Types" in idea 
           for idea in previously_tried_ideas):
        return "\n".join(previously_tried_ideas) + "\n"
    
    # Parse old format (SUCCESS:/FAILED:/UNCLEAR: prefixes)
    successful = [i for i in previously_tried_ideas if i.startswith("SUCCESS:")]
    failed = [i for i in previously_tried_ideas if i.startswith("FAILED:")]
    unclear = [i for i in previously_tried_ideas if i.startswith("UNCLEAR:")]
    
    def format_list(items):
        return chr(10).join(f"- {i}" for i in items) if items else "- None"
    
    return f"""
## CRITICAL: Previously Tried Ideas - CHECK THIS FIRST

**IMPORTANT:** Before generating new ideas, carefully review what was already tried. Do NOT suggest ideas that use the same libraries, functions, or approaches as failed attempts.

### Successful Ideas (worked - note what made them effective):
{format_list(successful)}

### Failed Ideas (avoid similar approaches - these did not help):
{format_list(failed)}

### Unclear Results (no clear improvement):
{format_list(unclear)}

**CRITICAL REQUIREMENT:** Your new ideas must use DIFFERENT libraries, functions, or techniques than the failed attempts listed above.

**STRICT PROHIBITION:** Do NOT keep suggesting approaches that have already failed. If a library or approach like numpy.argsort has failed many (like 3+) times, do NOT suggest numpy.argsort, numpy.argpartition, or any argsort variant. If scipy.optimize.linprog failed 2+ times, do NOT suggest any linear programming approach. Learn from failures - they indicate fundamental mismatches.

**Learning from Failures - Understand Root Causes:**
When a technique fails badly (score decreased significantly), understand WHY before suggesting alternatives:
- **Fundamental mismatch:** Wrong problem type (e.g., continuous optimizer on discrete problem) -> avoid that entire class of approaches
- **Structural mismatch:** Wrong approach for problem structure (e.g., linear proxy for non-linear objective) -> use approaches that match the actual structure
- **Implementation issues:** If the same library failed multiple times or very badly (>10% decrease), it likely indicates a fundamental mismatch - suggest a different class of approaches
- If failures were minor or unclear, the approach might work with better implementation - but prioritize proven successful patterns first
"""


# =============================================================================
# PROMPT BUILDING - ANALYSIS FRAMEWORK
# =============================================================================

def build_analysis_framework(best_score: float) -> str:
    """Build the 6-step analysis framework section."""
    return f"""
## CRITICAL: BEFORE GENERATING IDEAS, YOU MUST COMPLETE THIS ANALYSIS

**WHERE TO FIND INFORMATION:**
- **Problem Objective section**: Contains the task description (from config.yaml) - THIS IS THE TASK ITSELF
- **Evaluator Code section**: Shows how solutions are scored (from evaluator.py)
- **Current Program section**: Contains the current implementation that needs improvement

---

**STEP 0: Understand the TASK (MOST IMPORTANT - DO THIS FIRST)**

Read the Problem Objective section carefully to understand:
- What is the problem asking you to do?
- What is the goal or objective? (maximize, minimize, optimize)
- What are the inputs and outputs?
- What needs to be improved? (variables/decisions that affect the goal)
- What constraints exist?

**CRITICAL:** Analyze the Current Program NOW - understand what algorithm it uses and why it achieves score {best_score:.6f}. This is your starting point.

---

**STEP 1: Analyze the Evaluator Code**

Read the evaluator code line by line to understand:
- How solutions are scored (what metrics are computed)
- What constraints must be satisfied
- How instances/data are processed (sequentially or globally)

---

**STEP 2: Identify Metrics**

From the Evaluator Code, list:
- What metrics are computed? (combined_score, accuracy, performance, etc.)
- What is the primary metric? Secondary metrics?
- How are they calculated?
- If variance/std is penalized, the program needs consistency across scenarios

---

**STEP 3: Identify Constraints**

From the Evaluator Code, list:
- What conditions must be satisfied?
- What validation happens?
- What causes failure?
- What numerical/domain/structural constraints exist?

---

**STEP 4: Identify Problem Structure**

From the Evaluator Code, determine:
- **Processing pattern:** Sequential (one after another) or Global (all together)?
- **CRITICAL:** What data does your program receive vs what the evaluator uses? Check if evaluator uses different data (e.g., `workloads[i+1]` vs `workloads[i]`)
- **CRITICAL:** How are metrics computed across components? Independently then aggregated, or jointly?
- **Dependencies:** If order matters, WHY? What creates dependencies between items?
- **Decision variables:** DISCRETE (assignments, selections) or CONTINUOUS (real-valued)?

---

**STEP 5: Determine Appropriate Approach**

Based on your analysis:
- **Sequential processing** -> per-instance dynamic approaches (NOT static global optimization)
- **Global processing** -> global optimization approaches
- **Dependencies/constraints between decisions** -> analyze ALL dependencies upfront, build dependency graph/DAG, use topological sort (scipy.sparse.csgraph.topological_sort), minimize critical path
- **Discrete variables (no dependencies)** -> greedy heuristics, local search, scipy.optimize.linprog for linear constraints, scipy.optimize.linear_sum_assignment for one-to-one assignment. **Do NOT use scipy.optimize.minimize for discrete problems.**
- **Continuous with constraints** -> **PREFER scipy.optimize.minimize** with constraint-handling methods. Alternative: geometric approaches or reformulations. **Do NOT use multi-stage optimization.**
- **Continuous without constraints** -> scipy.optimize.minimize, geometric structures, spatial queries

---

**STEP 6: Identify Improvement Opportunities**

Based on your analysis:
- What would increase each metric?
- What would satisfy constraints better?
- If metrics are computed jointly -> optimize all components together
- If Current Program uses graph/network structures -> consider graph-based optimization
- If evaluator measures repair quality -> use structural relationships to detect violations and infer corrections
- What library techniques apply to this problem type?
"""


# =============================================================================
# PROMPT BUILDING - TECHNIQUES & GUIDANCE
# =============================================================================

def build_techniques_section() -> str:
    """Build the proven techniques section."""
    return """
## Task

Generate 2-3 breakthrough ideas of DIFFERENT TYPES that can improve the score.

### Proven Effective Techniques

**For Repair/Reconstruction Problems** (evaluator compares to ground truth):
Use heuristic-based approaches that exploit structural relationships. A complete repair approach should: (1) **detect** violations by checking structural constraints, (2) **correct** violations using simple heuristics (averaging, interpolation, copying from consistent parts), (3) **estimate confidence** from data consistency. **Key insight:** Optimization solvers often fail on repair problems because they over-constrain - prefer rule-based heuristics.

**For Dependency/Ordering Problems** (order matters due to conflicts):
Analyze dependency structure from evaluator logic, build dependency graph/DAG from conflict/constraint analysis, use topological sort with critical path optimization (scipy.sparse.csgraph.topological_sort), exploit parallelism by grouping non-conflicting items. For sophisticated global ordering: consider pairwise preference ranking and spectral ranking (numpy.linalg.eig on preference matrix to get global ordering).

**For Graph/Network Problems** (Current Program uses NetworkX):
Match graph algorithms to problem structure - tree algorithms (steiner_tree, minimum_spanning_tree) for shared structure, flow algorithms (min_cost_flow, network_simplex) for resource distribution, shortest path for routing. **Key insight:** Tree-based approaches can significantly reduce aggregate cost by exploiting shared structure.

**For Continuous Optimization with Constraints:**
**PREFER scipy.optimize.minimize** with constraint-handling methods (optimizes all variables together, most reliable). Key: multiple initial guesses, feasible starting point, flat array structure, clear constraint formulation. **Alternative:** Geometric approaches (scipy.spatial.Voronoi) or reformulations. **CRITICAL:** Do NOT use multi-stage optimization.

**For Filtering/Noise Reduction:**
Use methods that handle outliers better than mean-based (median, percentile-based statistics). Use filtering functions directly (scipy.signal.medfilt) - do NOT use scipy.optimize.minimize to tune filter parameters.

**For Performance-Critical Operations:**
Use numpy vectorization (broadcasting, vectorized operations) instead of Python loops.

**For Sequential/Streaming Data:**
Per-instance processing with efficient data structures. For simple ordering: prefer numpy.argsort over complex graph algorithms. For sophisticated ordering: spectral ranking (numpy.linalg.eig) or pandas.DataFrame.groupby for signature-based grouping.

**For Discrete Assignment (without dependencies):**
Greedy heuristics, local search, scipy.optimize.linprog for linear constraints, scipy.optimize.linear_sum_assignment for one-to-one assignment. For load balancing: numpy.argsort for priority-based greedy, numpy.argpartition for top-K, numpy.percentile for threshold-based decisions. **Key insight:** Simple greedy with good ordering often beats complex optimization. **CRITICAL:** Do NOT use scipy.optimize.minimize for discrete problems.

**For Discrete Optimization with Non-Linear Objectives (ratios):**
Evaluate the PROSPECTIVE objective (future state after placement) after each choice. For ratio objectives: evaluate the FULL ratio directly, NOT just the numerator. For MIN-MAX ratio: binary search on target + greedy placement + local improvements (swaps).
"""


def build_library_section() -> str:
    """Build the library recommendations section."""
    return """
### Libraries to Use

- Use any libraries already imported in Current Program
- If no specific libraries, use: numpy, scipy.*, pandas (if DataFrames used)

**Effective Techniques:**
- `scipy.optimize.minimize` - constrained continuous optimization
- `scipy.signal` (medfilt, savgol_filter, wiener) - robust filtering/noise reduction
- `numpy.argsort` - ordering/sequencing, priority-based greedy assignment
- `numpy.linalg.eig` - spectral ranking for global ordering
- `pandas.DataFrame.groupby` - signature-based caching/grouping
- `scipy.optimize.linprog` - linear programming
- `scipy.optimize.linear_sum_assignment` - one-to-one assignment (Hungarian algorithm)
- `numpy.percentile` - robust threshold-based allocation
- `numpy.argpartition` - efficient top-K selection
- `itertools.combinations` - pairwise swaps and local search

**AVOID:** DEAP, genetic algorithm libraries, domain-specific complex libraries, custom research algorithms, or any library requiring additional `pip install`
"""


def build_guidance_section() -> str:
    """Build the implementation guidance section."""
    return """
### Implementation Guidance

**Core Principles:**
- Prefer straightforward approaches that can be implemented clearly
- **Each idea MUST be a single-function library call** - no multi-stage processing
- **MULTI-STAGE OPTIMIZATION IS FORBIDDEN** - do NOT call one function then optimize its output
- Simple direct optimization often beats complex multi-step when constraints are tight

**scipy.optimize.minimize Rules:**
- CORRECT: Use to solve the problem directly (optimize positions, minimize objective)
- WRONG: Do NOT use for hyperparameter tuning or to find parameters for another function
- WRONG: Do NOT use to optimize output signal/data vectors directly

**Match Approach to Problem Structure:**
- **FOR CONTINUOUS WITH CONSTRAINTS:** PREFER scipy.optimize.minimize (optimizes all variables together)
- **FOR DISCRETE PROBLEMS:** Use discrete heuristics, linprog, linear_sum_assignment, or numpy.percentile. **Do NOT use scipy.optimize.minimize.**
- **FOR DATA PROCESSING:** Use scipy.signal.* functions directly. **Do NOT use multi-stage optimization.**

**Learning from Success:**
When an approach succeeds, think: what principle made it work? Learn and think of better ideas, don't just add complexity. If breakthrough patterns are known, prioritize approaches that match them.
"""


def build_diversity_section() -> str:
    """Build the diversity requirements section."""
    return """
### Diversity Requirement

Before generating ideas, explicitly think:
- Idea 1: [Type A - e.g., algorithmic refinement or library-based approach]
- Idea 2: [Type B - e.g., structural change or processing pattern - DIFFERENT from A]
- Idea 3: [Type C - e.g., different technique or optimization method - DIFFERENT from A and B]

**Verify:** Are these DIFFERENT types? NOT variations of the same approach.

Each idea must:
- Use DIFFERENT libraries/techniques than failed attempts
- Target DIFFERENT metrics/aspects from the evaluator
- Be independently implementable
- Prefer clear implementations (different != more complex)

### Be Specific and Actionable

Not vague: "Try optimization"
Specific: "Use scipy.optimize.minimize with SLSQP method"

- Include exact library names, function names, methods, parameters
- Provide step-by-step implementation guide
- Focus on core logic that implements the idea correctly
- Handle edge cases and avoid errors/warnings
"""


def build_output_format_section() -> str:
    """Build the output format section."""
    return '''
### Example Paradigm Structure

```json
{
  "idea": "Use scipy.optimize.minimize with SLSQP",
  "description": "Apply scipy.optimize.minimize directly to optimize all variables together. Use SLSQP method with constraint dict. This addresses [METRIC] by [HOW]. Ensure output format matches evaluator requirements.",
  "what_to_optimize": "[METRIC_FROM_EVALUATOR]",
  "cautions": "[KEY_IMPLEMENTATION_DETAILS]",
  "approach_type": "scipy.optimize.minimize"
}
```

### Output Format

Return a JSON array. Each idea should have:
- `idea`: Clear, direct idea with library/technique name
- `description`: Detailed implementation guide (5-10 sentences) with exact library names, methods, code structure
- `what_to_optimize`: What metrics/areas to focus on (from evaluator analysis)
- `cautions`: Important implementation details to watch out for
- `approach_type`: **CRITICAL:** Must be exact "[LIBRARY].[FUNCTION]" format (e.g., "scipy.optimize.minimize", "scipy.signal.medfilt", "numpy.argsort"). **DO NOT use "Other"**.

Complete ALL analysis steps (Steps 0-6), review proven techniques, follow guidance, ensure diversity, and generate ideas that match the problem structure. Return ONLY the JSON array, no other text.'''


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

async def generate_paradigms(
    config_path: str,
    evaluator_path: str,
    initial_program_path: Optional[str] = None,
    api_key: str = None,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-5-mini",
    temperature: float = 0.5,
    output_path: Optional[str] = None,
    best_program_code: Optional[str] = None,
    best_score: float = 0.0,
    previously_tried_ideas: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate breakthrough paradigms using LLM analysis.
    
    Args:
        config_path: Path to config.yaml
        evaluator_path: Path to evaluator.py
        initial_program_path: Path to initial_program.py (optional)
        api_key: OpenAI API key
        api_base: API base URL
        model: Model name (default: gpt-5-mini)
        temperature: Sampling temperature
        output_path: Output JSONL file path
        best_program_code: Current best program code (optional)
        best_score: Current best score
        previously_tried_ideas: List of previously tried ideas with outcomes
        
    Returns:
        List of paradigm dicts
    """
    # -------------------------------------------------------------------------
    # Load inputs
    # -------------------------------------------------------------------------
    print(f"Reading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    system_message = config.get('prompt', {}).get('system_message', '')
    
    print(f"Reading evaluator from {evaluator_path}...")
    with open(evaluator_path, 'r') as f:
        evaluator_code = f.read()
    
    # Get program code (prefer best_program_code, fallback to initial_program_path)
    program_code = ""
    program_label = "Current Program"
    
    if best_program_code:
        program_code = best_program_code
        program_label = f"Current Best Program (score: {best_score:.6f})"
    elif initial_program_path and os.path.exists(initial_program_path):
        print(f"Reading initial program from {initial_program_path}...")
        with open(initial_program_path, 'r') as f:
            program_code = f.read()
        program_label = "Current Initial Program"
    
    # -------------------------------------------------------------------------
    # Build prompt from sections
    # -------------------------------------------------------------------------
    prompt_sections = [
        build_problem_context(system_message, evaluator_code, program_code, program_label),
        build_current_program_analysis(best_score),
        build_previously_tried_section(previously_tried_ideas),
        build_analysis_framework(best_score),
        build_techniques_section(),
        build_library_section(),
        build_guidance_section(),
        build_diversity_section(),
        build_output_format_section(),
    ]
    prompt = "\n".join(prompt_sections)
    
    # -------------------------------------------------------------------------
    # Call LLM
    # -------------------------------------------------------------------------
    # Ensure OpenAI API (not Gemini)
    if "generativelanguage.googleapis.com" in api_base:
        print("WARNING: Detected Gemini API. Switching to OpenAI API.")
        api_base = "https://api.openai.com/v1"
    
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=900,  # 15 minutes for reasoning models
    )
    
    print(f"Calling {model} to generate paradigms...")
    
    create_kwargs = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert researcher. Think carefully and deeply. "
                    "Analyze the problem thoroughly, understand the evaluation metric "
                    "by reading the evaluator code, and suggest useful ideas that are "
                    "correct, actionable, and will actually help improve the solution."
                )
            },
            {"role": "user", "content": prompt}
        ],
    }
    
    # Temperature setting (gpt-5-mini supports custom, others use default 1.0)
    if model == "gpt-5-mini" or not model.startswith("gpt-5"):
        create_kwargs["temperature"] = temperature
    
    response = client.chat.completions.create(**create_kwargs)
    response_text = response.choices[0].message.content.strip()
    
    # -------------------------------------------------------------------------
    # Parse response
    # -------------------------------------------------------------------------
    if not response_text:
        raise ValueError("Empty response from LLM")
    
    # Extract JSON from markdown code blocks if present
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        response_text = response_text.split("```")[1].split("```")[0].strip()
    
    try:
        paradigms = json.loads(response_text.strip())
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        print(f"Response (first 500 chars): {response_text[:500]}")
        raise ValueError(f"Failed to parse JSON response: {e}")
    
    if not isinstance(paradigms, list):
        raise ValueError(f"Expected list, got {type(paradigms)}")
    
    # -------------------------------------------------------------------------
    # Output results
    # -------------------------------------------------------------------------
    if output_path:
        print(f"Writing {len(paradigms)} paradigms to {output_path}...")
        with open(output_path, 'w') as f:
            for paradigm in paradigms:
                f.write(json.dumps(paradigm) + '\n')
        
        print(f"\n✓ Successfully generated {len(paradigms)} paradigms!")
        print(f"Output: {output_path}\n")
        
        print("Generated Ideas:")
        print("=" * 80)
        for i, p in enumerate(paradigms, 1):
            print(f"\n{i}. {p.get('idea', 'unnamed')}")
            print(f"   Description: {p.get('description', 'N/A')}")
            print(f"   Optimize: {p.get('what_to_optimize', 'N/A')}")
            print(f"   Cautions: {p.get('cautions', 'N/A')}")
    
    return paradigms


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """Command-line interface for paradigm generation."""
    parser = argparse.ArgumentParser(
        description="Generate breakthrough paradigms for an OpenEvolve example"
    )
    parser.add_argument(
        "example_dir", type=str,
        help="Path to example directory (e.g., examples/circle_packing)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Config file name (default: config.yaml)"
    )
    parser.add_argument(
        "--evaluator", type=str, default="evaluator.py",
        help="Evaluator file name (default: evaluator.py)"
    )
    parser.add_argument(
        "--initial-program", type=str, default="initial_program.py",
        help="Initial program file name (default: initial_program.py)"
    )
    parser.add_argument(
        "--api-key", type=str,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--api-base", type=str, default="https://api.openai.com/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5-mini",
        help="Model name (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--output", type=str, default="breakthrough_paradigms.jsonl",
        help="Output JSONL file path"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    example_dir = Path(args.example_dir)
    config_path = example_dir / args.config
    evaluator_path = example_dir / args.evaluator
    initial_program_path = example_dir / args.initial_program if args.initial_program else None
    
    # Validate paths
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    if not evaluator_path.exists():
        print(f"ERROR: Evaluator file not found: {evaluator_path}")
        sys.exit(1)
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)
    
    # Ensure OpenAI API base
    api_base = args.api_base or "https://api.openai.com/v1"
    if "generativelanguage.googleapis.com" in api_base:
        print("WARNING: Detected Gemini API. Switching to OpenAI API.")
        api_base = "https://api.openai.com/v1"
    
    # Run
    asyncio.run(generate_paradigms(
        config_path=str(config_path),
        evaluator_path=str(evaluator_path),
        initial_program_path=str(initial_program_path) if initial_program_path and initial_program_path.exists() else None,
        api_key=api_key,
        api_base=api_base,
        model=args.model,
        output_path=args.output,
    ))


if __name__ == "__main__":
    main()
