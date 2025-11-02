#!/usr/bin/env python3
"""
Standalone Test Script for Multi-Agent Math Solving Programs

This script evaluates any program on a specified number of math problems
with a configurable random seed for reproducible test/train splits.

Usage:
    python test_program.py path/to/program.py --num-problems 100 --seed 42
    python test_program.py openevolve_output/best/best_program.py --seed 99
    python test_program.py initial_program.py --num-problems 200 --seed 1234
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import evaluation utilities
from eval_utils import get_success_olympiad, extract_answer, strip_string, math_equal


def load_math_dataset(num_samples: int = 100, seed: int = 42) -> List[Dict[str, str]]:
    """
    Load math problems from OlympiadBench dataset.

    Args:
        num_samples: Number of problems to sample (-1 for all 675)
        seed: Random seed for reproducibility

    Returns:
        List of problem dictionaries
    """
    import random

    olympiad_path = '/Users/mertcemri/Desktop/research/autoevolve/olympiadbench/test.jsonl'

    if not os.path.exists(olympiad_path):
        print(f"Error: OlympiadBench dataset not found at {olympiad_path}")
        print("Using fallback synthetic problems...")
        return [
            {'question': 'What is $2 + 2$?', 'final_answer': '4', 'answer': '4'},
            {'question': 'What is $5 \\times 3$?', 'final_answer': '15', 'answer': '15'},
        ][:num_samples]

    # Load all problems
    problems = []
    with open(olympiad_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line))

    def extract_final_answer(item):
        """Extract answer, handling both list and string formats"""
        final_ans = item['final_answer']
        if isinstance(final_ans, list):
            return final_ans[0] if final_ans else ''
        return final_ans

    # Use all problems if requested
    if num_samples < 0:
        print(f"Using all {len(problems)} problems from OlympiadBench dataset (seed={seed})")
        return [{
            'question': item['question'],
            'final_answer': extract_final_answer(item),
            'answer': extract_final_answer(item)
        } for item in problems]

    # Sample problems with specified seed
    print(f"Sampling {num_samples} problems from OlympiadBench dataset (seed={seed})")
    random.seed(seed)
    sampled = random.sample(problems, min(num_samples, len(problems)))

    return [{
        'question': item['question'],
        'final_answer': extract_final_answer(item),
        'answer': extract_final_answer(item)
    } for item in sampled]


def load_program(program_path: str):
    """Load a Python program as a module."""
    if not os.path.exists(program_path):
        raise FileNotFoundError(f"Program not found: {program_path}")

    spec = importlib.util.spec_from_file_location("test_program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load program: {program_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["test_program"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "run_evaluation_sample"):
        raise AttributeError(
            f"Program {program_path} does not contain 'run_evaluation_sample' function"
        )

    return module


def calculate_accuracy(results: List[Dict]) -> float:
    """Calculate accuracy using get_success_olympiad."""
    if not results:
        return 0.0

    formatted_samples = [{
        'pred': [r.get('pred', r.get('answer', ''))],
        'answer': r.get('gold_answer', ''),
        'completions': [r.get('pred', r.get('answer', ''))]
    } for r in results]

    try:
        accuracy, failed_comparisons = get_success_olympiad(formatted_samples, tqdm_bar=True)
        return float(accuracy)
    except Exception as e:
        print(f"Warning: get_success_olympiad failed ({e}), using fallback")
        correct = sum(
            1 for s in formatted_samples
            if math_equal(
                strip_string(extract_answer(s['pred'][0]), skip_unit=False),
                strip_string(extract_answer(s['answer']), skip_unit=False)
            )
        )
        return correct / len(formatted_samples)


def evaluate_program(program_path: str, num_problems: int, seed: int) -> Dict:
    """
    Evaluate a program on math problems.

    Args:
        program_path: Path to the program file
        num_problems: Number of problems to evaluate
        seed: Random seed for problem sampling

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Testing Program: {program_path}")
    print(f"{'='*60}")
    print(f"Number of problems: {num_problems}")
    print(f"Random seed: {seed}")
    print(f"{'='*60}\n")

    # Load program
    print("Loading program...")
    program = load_program(program_path)
    print("✓ Program loaded successfully\n")

    # Load problems
    print("Loading problems...")
    problems = load_math_dataset(num_samples=num_problems, seed=seed)
    print(f"✓ Loaded {len(problems)} problems\n")

    # Set environment variables (in case program uses them)
    os.environ['MATH_EVAL_PROBLEMS'] = str(num_problems)
    os.environ['OPENEVOLVE_ITERATION'] = '-1'  # Indicate this is testing, not evolution

    # Run evaluation
    print("Running evaluation...")
    start_time = time.time()

    try:
        results = program.run_evaluation_sample(problems, len(problems))
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e),
            'accuracy': 0.0,
            'num_problems': len(problems),
            'seed': seed
        }

    eval_time = time.time() - start_time
    print(f"✓ Evaluation complete in {eval_time:.1f}s\n")

    # Calculate metrics
    print("Calculating metrics...")
    accuracy = calculate_accuracy(results)

    total_calls = sum(r.get('llm_calls', 0) for r in results)
    avg_calls = total_calls / len(results) if results else 0

    correct = sum(
        1 for r in results
        if r.get('pred', '').strip() and not r.get('error')
    )

    errors = sum(1 for r in results if r.get('error'))

    # Compile results
    evaluation_results = {
        'program_path': program_path,
        'timestamp': datetime.now().isoformat(),
        'seed': seed,
        'num_problems': len(problems),
        'accuracy': float(accuracy),
        'correct': correct,
        'errors': errors,
        'total_llm_calls': int(total_calls),
        'avg_llm_calls': float(avg_calls),
        'eval_time': float(eval_time),
        'time_per_problem': float(eval_time / len(problems)) if problems else 0,
    }

    return evaluation_results


def print_results(results: Dict):
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Program: {results['program_path']}")
    print(f"Seed: {results['seed']}")
    print(f"Problems: {results['num_problems']}")
    print(f"{'='*60}")
    print(f"Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['num_problems']})")
    print(f"Errors: {results['errors']}")
    print(f"Avg LLM calls: {results['avg_llm_calls']:.1f}")
    print(f"Total LLM calls: {results['total_llm_calls']}")
    print(f"Eval time: {results['eval_time']:.1f}s")
    print(f"Time per problem: {results['time_per_problem']:.1f}s")
    print(f"{'='*60}\n")


def save_results(results: Dict, output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test a multi-agent math solving program",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test initial program with 100 problems, seed 42
  python test_program.py initial_program.py

  # Test evolved program with different seed for test set
  python test_program.py openevolve_output/best/best_program.py --seed 99

  # Test with 200 problems
  python test_program.py path/to/program.py --num-problems 200

  # Test on full dataset (675 problems)
  python test_program.py path/to/program.py --num-problems -1

  # Save results to specific file
  python test_program.py path/to/program.py --output results.json
"""
    )

    parser.add_argument(
        'program_path',
        help='Path to the program file to test'
    )

    parser.add_argument(
        '--num-problems', '-n',
        type=int,
        default=100,
        help='Number of problems to evaluate (default: 100, use -1 for all 675)'
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for problem sampling (default: 42)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file for results (default: test_results_<timestamp>.json)'
    )

    args = parser.parse_args()

    # Run evaluation
    try:
        results = evaluate_program(args.program_path, args.num_problems, args.seed)

        if 'error' in results:
            print("\n✗ Evaluation failed")
            sys.exit(1)

        # Print results
        print_results(results)

        # Save results
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"test_results_{timestamp}.json"

        save_results(results, args.output)

        print(f"\n✓ Testing complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
