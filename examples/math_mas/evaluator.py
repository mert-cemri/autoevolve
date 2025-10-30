"""
Evaluator for Multi-Agent Math Solving System

This evaluator tests the multi-agent system on math problems and calculates
accuracy using the get_success function from eval_utils.py
"""

import importlib.util
import json
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Import the evaluation utilities
from eval_utils import get_success_olympiad, extract_answer, strip_string, math_equal

logger = logging.getLogger(__name__)

# ===== COMBINED SCORE HYPERPARAMETERS =====
ACCURACY_WEIGHT = 0.7        # Correctness (primary metric)
EFFICIENCY_WEIGHT = 0.25     # LLM call efficiency (fewer calls = higher score)
TIME_WEIGHT = 0.05           # Speed (faster = higher score)
# Total = 1.0

# Efficiency scoring
TARGET_LLM_CALLS = 3.0       # Ideal: Solver + Verifier + optional Refiner
MAX_PENALTY_CALLS = 10.0     # Full penalty beyond this


def load_math_dataset(num_samples: int = 50, seed: int = 42) -> List[Dict[str, str]]:
    """
    Load math problems from OlympiadBench dataset.

    Args:
        num_samples: Number of problems to sample (default: 50)
        seed: Random seed for reproducibility (default: 42)
              - All strategies use the same seed for fair comparison
              - All iterations see the same problems (good for reproducibility)
              - Set num_samples=-1 to use all 675 problems

    Returns:
        List of problem dictionaries with question, final_answer, and answer
    """
    try:
        olympiad_path = '/Users/mertcemri/Desktop/research/autoevolve/olympiadbench/test.jsonl'

        # Load all problems from JSONL
        problems = []
        with open(olympiad_path, 'r') as f:
            for line in f:
                problems.append(json.loads(line))

        # Use all problems if num_samples is -1
        if num_samples < 0:
            logger.info(f"Using all {len(problems)} problems from OlympiadBench dataset")
            return [{
                'question': item['question'],
                'final_answer': item['final_answer'],
                'answer': item['final_answer']
            } for item in problems]

        # Randomly sample problems with fixed seed for reproducibility
        logger.info(f"Randomly sampling {num_samples} problems from OlympiadBench dataset (seed={seed})")
        random.seed(seed)
        sampled = random.sample(problems, min(num_samples, len(problems)))

        return [{
            'question': item['question'],
            'final_answer': item['final_answer'],
            'answer': item['final_answer']
        } for item in sampled]

    except Exception as e:
        print(f"Warning: Using synthetic problems (dataset load failed: {e})")
        return [
            {'question': 'What is $2 + 2$?', 'final_answer': '4', 'answer': '4'},
            {'question': 'Compute $3! + 4!$', 'final_answer': '30', 'answer': '30'},
            {'question': 'What is $\\sqrt{16}$?', 'final_answer': '4', 'answer': '4'},
            {'question': 'If $x + 5 = 12$, what is $x$?', 'final_answer': '7', 'answer': '7'},
            {'question': 'What is $10 \\times 10$?', 'final_answer': '100', 'answer': '100'}
        ][:num_samples]


def run_with_timeout(program_module, problems: List[Dict], timeout_seconds: int = 300):
    """Run program evaluation with timeout protection."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(program_module.run_evaluation_sample, problems, len(problems))
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Timed out after {timeout_seconds}s")


def calculate_accuracy(results: List[Dict]) -> float:
    """Calculate accuracy using get_success_olympiad from eval_utils."""
    if not results:
        return 0.0

    # Format samples for get_success_olympiad
    # It expects: pred (as list), answer (ground truth), completions (list)
    formatted_samples = [{
        'pred': [r.get('pred', r.get('answer', ''))],  # Wrap in list
        'answer': r.get('gold_answer', ''),
        'completions': [r.get('pred', r.get('answer', ''))]  # For 'no solution' check
    } for r in results]

    try:
        accuracy, failed_comparisons = get_success_olympiad(formatted_samples, tqdm_bar=False)
        return float(accuracy)
    except Exception as e:
        print(f"get_success_olympiad failed ({e}), using fallback")
        correct = sum(
            1 for s in formatted_samples
            if math_equal(
                strip_string(extract_answer(s['pred'][0]), skip_unit=False),
                strip_string(extract_answer(s['answer']), skip_unit=False)
            )
        )
        return correct / len(formatted_samples)


def evaluate(program_path: str) -> Dict[str, float]:
    """Main evaluation function called by OpenEvolve."""
    try:
        # Load program
        spec = importlib.util.spec_from_file_location("program", program_path)
        if not spec or not spec.loader:
            return {"accuracy": 0.0, "avg_llm_calls": 0.0, "combined_score": 0.0, "error": "Load failed"}

        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        if not hasattr(program, "run_evaluation_sample"):
            return {"accuracy": 0.0, "avg_llm_calls": 0.0, "combined_score": 0.0, "error": "Missing run_evaluation_sample"}

        # Load problems
        num_problems = int(os.environ.get("MATH_EVAL_PROBLEMS", "10"))
        problems = load_math_dataset(num_samples=num_problems, seed=42)
        if not problems:
            return {"accuracy": 0.0, "avg_llm_calls": 0.0, "combined_score": 0.0, "error": "No problems"}

        # Run evaluation
        start_time = time.time()
        results = run_with_timeout(program, problems, timeout_seconds=600)
        eval_time = time.time() - start_time

        # Calculate accuracy
        accuracy = calculate_accuracy(results)

        # Calculate time efficiency
        time_per_problem = eval_time / len(problems) if problems else 999
        time_score = min(1.0, 30.0 / max(time_per_problem, 1.0))

        # Calculate LLM call efficiency
        total_calls = sum(r.get('llm_calls', 0) for r in results)
        avg_calls = total_calls / len(results) if results else 0

        if avg_calls <= TARGET_LLM_CALLS:
            efficiency_score = 1.0
        elif avg_calls >= MAX_PENALTY_CALLS:
            efficiency_score = 0.0
        else:
            efficiency_score = 1.0 - (avg_calls - TARGET_LLM_CALLS) / (MAX_PENALTY_CALLS - TARGET_LLM_CALLS)

        # Combined score
        combined_score = (
            ACCURACY_WEIGHT * accuracy +
            EFFICIENCY_WEIGHT * efficiency_score +
            TIME_WEIGHT * time_score
        )

        print(f"\nEvaluation Results:")
        print(f"  Accuracy: {accuracy:.2%} (weight: {ACCURACY_WEIGHT})")
        print(f"  Avg LLM calls: {avg_calls:.1f} → efficiency: {efficiency_score:.3f} (weight: {EFFICIENCY_WEIGHT})")
        print(f"  Time/problem: {time_per_problem:.1f}s → speed: {time_score:.3f} (weight: {TIME_WEIGHT})")
        print(f"  Combined score: {combined_score:.4f}")

        return {
            "accuracy": float(accuracy),
            "avg_llm_calls": float(avg_calls),
            "efficiency_score": float(efficiency_score),
            "time_per_problem": float(time_per_problem),
            "time_score": float(time_score),
            "combined_score": float(combined_score),
            "num_problems": len(problems),
            "total_llm_calls": int(total_calls),
            "eval_time": float(eval_time)
        }

    except TimeoutError:
        return {"accuracy": 0.0, "avg_llm_calls": 0.0, "combined_score": 0.0, "error": "timeout"}
    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        return {"accuracy": 0.0, "avg_llm_calls": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage1(program_path: str) -> Dict[str, float]:
    """Stage 1: Quick validation with 2 simple problems."""
    try:
        # Load program
        spec = importlib.util.spec_from_file_location("program", program_path)
        if not spec or not spec.loader:
            return {"validation_passed": 0.0, "combined_score": 0.0, "error": "Load failed"}

        program = importlib.util.module_from_spec(spec)
        sys.modules["program"] = program
        spec.loader.exec_module(program)

        if not hasattr(program, "run_evaluation_sample"):
            return {"validation_passed": 0.0, "combined_score": 0.0, "error": "Missing function"}

        # Test with 2 simple problems
        test_problems = [
            {'question': 'What is $2 + 2$?', 'final_answer': '4', 'answer': '4'},
            {'question': 'What is $5 \\times 3$?', 'final_answer': '15', 'answer': '15'}
        ]

        start_time = time.time()
        results = run_with_timeout(program, test_problems, timeout_seconds=60)
        eval_time = time.time() - start_time

        has_errors = any('error' in r for r in results)
        accuracy = 0.0 if has_errors else calculate_accuracy(results)
        total_calls = sum(r.get('llm_calls', 0) for r in results)
        avg_calls = total_calls / len(results) if results else 0.0
        combined_score = accuracy

        print(f"\nStage 1: {'PASS' if not has_errors else 'FAIL'} | Accuracy: {accuracy:.2%}")

        return {
            "validation_passed": 1.0 if not has_errors else 0.0,
            "accuracy": float(accuracy),
            "avg_llm_calls": float(avg_calls),
            "combined_score": float(combined_score),
            "stage1_time": float(eval_time)
        }

    except Exception as e:
        print(f"Stage 1 failed: {e}")
        return {"validation_passed": 0.0, "accuracy": 0.0, "avg_llm_calls": 0.0, "combined_score": 0.0, "error": str(e)}


def evaluate_stage2(program_path: str) -> Dict[str, float]:
    """Stage 2: Medium evaluation with 5 problems."""
    original = os.environ.get("MATH_EVAL_PROBLEMS", "10")
    os.environ["MATH_EVAL_PROBLEMS"] = "5"
    try:
        return evaluate(program_path)
    finally:
        os.environ["MATH_EVAL_PROBLEMS"] = original


def evaluate_stage3(program_path: str) -> Dict[str, float]:
    """Stage 3: Full evaluation (same as main evaluate)."""
    return evaluate(program_path)


if __name__ == "__main__":
    program_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "initial_program.py"
    )

    print(f"Evaluating: {program_path}\n")

    stage1 = evaluate_stage1(program_path)
    print(f"Stage 1: {json.dumps(stage1, indent=2)}\n")

    if stage1.get("combined_score", 0) > 0.3:
        full = evaluate(program_path)
        print(f"\nFull: {json.dumps(full, indent=2)}")
    else:
        print("Stage 1 failed (score ≤ 0.3), skipping full evaluation")
