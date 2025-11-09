#!/usr/bin/env python3
"""
Debug script to test the initial program on a few problems

This helps identify why the program is timing out during evolution.
Run this BEFORE starting evolution to catch issues early.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from eval_utils import get_success_olympiad
import importlib.util

def load_program(program_path):
    """Load the program module."""
    spec = importlib.util.spec_from_file_location("test_program", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_program"] = module
    spec.loader.exec_module(module)
    return module

def test_simple_problems():
    """Test on the 2 simple Stage 1 problems."""
    print("\n" + "="*60)
    print("Stage 1 Test: 2 Simple Problems")
    print("="*60)

    problems = [
        {'question': 'What is $2 + 2$?', 'final_answer': '4', 'answer': '4'},
        {'question': 'What is $5 \\times 3$?', 'final_answer': '15', 'answer': '15'}
    ]

    program = load_program("initial_program.py")

    start_time = time.time()
    results = program.run_evaluation_sample(problems, 2)
    elapsed = time.time() - start_time

    print(f"\n✓ Completed in {elapsed:.1f}s ({elapsed/2:.1f}s per problem)")

    for i, (problem, result) in enumerate(zip(problems, results)):
        print(f"\nProblem {i+1}: {problem['question']}")
        print(f"  Expected: {problem['answer']}")
        print(f"  Got: {result.get('pred', 'ERROR')}")
        print(f"  LLM calls: {result.get('llm_calls', 0)}")
        print(f"  Correct: {'✓' if result.get('pred') == problem['answer'] else '✗'}")

    total_calls = sum(r.get('llm_calls', 0) for r in results)
    print(f"\nTotal LLM calls: {total_calls} ({total_calls/2:.1f} per problem)")

    if elapsed > 60:
        print(f"\n⚠️  WARNING: Stage 1 took {elapsed:.1f}s (expected < 60s)")
        print("   This will cause timeouts in Stage 2/3!")

    return elapsed < 60

def test_single_real_problem():
    """Test on a single real OlympiadBench problem."""
    print("\n" + "="*60)
    print("Single Real Problem Test")
    print("="*60)

    import json
    olympiad_path = '/Users/mertcemri/Desktop/research/autoevolve/olympiadbench/test.jsonl'

    with open(olympiad_path, 'r') as f:
        problem = json.loads(f.readline())

    problem_formatted = {
        'question': problem['question'],
        'final_answer': problem['final_answer'][0] if isinstance(problem['final_answer'], list) else problem['final_answer'],
        'answer': problem['final_answer'][0] if isinstance(problem['final_answer'], list) else problem['final_answer']
    }

    print(f"\nProblem: {problem_formatted['question'][:100]}...")
    print(f"Expected answer: {problem_formatted['answer']}")

    program = load_program("initial_program.py")

    print("\n⏱️  Running (timeout after 120s)...")
    start_time = time.time()

    try:
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Timed out after 120s")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(120)  # 120 second timeout

        results = program.run_evaluation_sample([problem_formatted], 1)

        signal.alarm(0)  # Cancel alarm
        elapsed = time.time() - start_time

        result = results[0]
        print(f"\n✓ Completed in {elapsed:.1f}s")
        print(f"  Got: {result.get('pred', 'ERROR')}")
        print(f"  LLM calls: {result.get('llm_calls', 0)}")
        print(f"  Correct: {'✓' if result.get('pred') == problem_formatted['answer'] else '✗'}")

        if elapsed > 60:
            print(f"\n⚠️  WARNING: Single problem took {elapsed:.1f}s")
            print("   Stage 2 (5 problems) will take ~{:.0f}s".format(elapsed * 5))
            print("   Stage 3 (100 problems) will take ~{:.0f}s".format(elapsed * 100))
            print("   Default timeout is 600s - this will fail!")

        return elapsed < 60

    except TimeoutError as e:
        print(f"\n✗ TIMEOUT: Problem took > 120s")
        print("   This program will definitely timeout in Stage 2/3")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("Initial Program Diagnostic")
    print("="*60)
    print("\nThis script tests if your initial program is suitable for evolution.")
    print("It should:")
    print("  1. Solve simple problems correctly")
    print("  2. Complete within reasonable time (< 60s per problem)")
    print("  3. Not have infinite loops or hangs")

    try:
        # Test Stage 1
        stage1_ok = test_simple_problems()

        if not stage1_ok:
            print("\n" + "="*60)
            print("⛔ RECOMMENDATION: Fix Stage 1 performance before evolution")
            print("="*60)
            print("\nThe initial program is too slow on simple problems.")
            print("Evolution will fail because all evaluations will timeout.")
            print("\nPossible issues:")
            print("  - Too many verification rounds (MAX_REVISION_ROUNDS)")
            print("  - Slow LLM model")
            print("  - Inefficient prompt/response parsing")
            return 1

        # Test a real problem
        real_ok = test_single_real_problem()

        if not real_ok:
            print("\n" + "="*60)
            print("⛔ RECOMMENDATION: Fix performance on real problems")
            print("="*60)
            print("\nThe program times out on OlympiadBench problems.")
            print("\nOptions:")
            print("  1. Reduce MAX_REVISION_ROUNDS (currently checking your code)")
            print("  2. Use faster LLM model (gpt-5-nano instead of gpt-4o)")
            print("  3. Increase evaluator timeout in config.yaml")
            print("  4. Simplify the multi-agent workflow")
            return 1

        print("\n" + "="*60)
        print("✅ Initial program looks good for evolution!")
        print("="*60)
        return 0

    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
