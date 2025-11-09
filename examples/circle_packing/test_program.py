#!/usr/bin/env python3
"""
Test and visualize circle packing programs

Usage:
    python test_program.py <program_path>
    python test_program.py initial_program.py
    python test_program.py openevolve_output/best/best_program.py
"""

import sys
import os
import argparse
import importlib.util
import numpy as np
import time
import json
from pathlib import Path


def load_program(program_path):
    """Load a program from file path"""
    spec = importlib.util.spec_from_file_location("program", program_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load program from {program_path}")

    program = importlib.util.module_from_spec(spec)
    sys.modules["program"] = program
    spec.loader.exec_module(program)

    return program


def validate_packing(centers, radii):
    """Validate that circles don't overlap and are inside the unit square"""
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any() or np.isnan(radii).any():
        print("❌ NaN values detected")
        return False

    # Check if radii are non-negative
    if (radii < 0).any():
        print("❌ Negative radii detected")
        return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            print(f"❌ Circle {i} at ({x:.4f}, {y:.4f}) with radius {r:.4f} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                print(f"❌ Circles {i} and {j} overlap: dist={dist:.4f}, r1+r2={radii[i]+radii[j]:.4f}")
                return False

    return True


def evaluate_program(program_path, visualize=True):
    """
    Evaluate a circle packing program

    Args:
        program_path: Path to the program file
        visualize: Whether to display visualization

    Returns:
        Dictionary with evaluation results
    """
    TARGET_VALUE = 2.635  # AlphaEvolve result for n=26

    print(f"\n{'='*60}")
    print(f"Evaluating: {program_path}")
    print(f"{'='*60}\n")

    try:
        # Load program
        program = load_program(program_path)

        # Run packing
        start_time = time.time()
        centers, radii, sum_radii = program.run_packing()
        eval_time = time.time() - start_time

        # Ensure numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)

        # Validate
        valid = validate_packing(centers, radii)

        # Check shape
        shape_valid = centers.shape == (26, 2) and radii.shape == (26,)
        if not shape_valid:
            print(f"❌ Invalid shapes: centers={centers.shape}, radii={radii.shape}")
            valid = False

        # Calculate metrics
        actual_sum = np.sum(radii) if valid else 0.0
        target_ratio = actual_sum / TARGET_VALUE if valid else 0.0

        # Display results
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"Valid packing:      {'✅ Yes' if valid else '❌ No'}")
        print(f"Number of circles:  {len(centers)}")
        print(f"Sum of radii:       {actual_sum:.6f}")
        print(f"Target (AlphaEvolve): {TARGET_VALUE}")
        print(f"Achievement ratio:  {target_ratio:.2%}")
        print(f"Evaluation time:    {eval_time:.3f}s")
        print(f"{'='*60}\n")

        # Statistics
        if valid:
            print(f"Circle Statistics:")
            print(f"  Min radius:       {np.min(radii):.6f}")
            print(f"  Max radius:       {np.max(radii):.6f}")
            print(f"  Mean radius:      {np.mean(radii):.6f}")
            print(f"  Std radius:       {np.std(radii):.6f}")
            print()

        # Visualize if requested
        if visualize and valid:
            try:
                program.visualize(centers, radii)
            except Exception as e:
                print(f"Note: Could not visualize: {e}")

        return {
            "valid": valid,
            "sum_radii": float(actual_sum),
            "target": TARGET_VALUE,
            "target_ratio": float(target_ratio),
            "eval_time": float(eval_time),
            "num_circles": len(centers),
            "min_radius": float(np.min(radii)) if valid else 0.0,
            "max_radius": float(np.max(radii)) if valid else 0.0,
            "mean_radius": float(np.mean(radii)) if valid else 0.0,
        }

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "valid": False,
            "sum_radii": 0.0,
            "target": TARGET_VALUE,
            "target_ratio": 0.0,
            "eval_time": 0.0,
            "error": str(e),
        }


def compare_programs(program_paths):
    """Compare multiple programs side by side"""
    results = []

    for path in program_paths:
        result = evaluate_program(path, visualize=False)
        result["program"] = path
        results.append(result)

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"COMPARISON")
    print(f"{'='*80}")
    print(f"{'Program':<40} {'Sum Radii':<12} {'Ratio':<10} {'Time':<8}")
    print(f"{'-'*80}")

    for result in results:
        program_name = Path(result["program"]).name
        print(f"{program_name:<40} {result['sum_radii']:<12.6f} {result['target_ratio']:<10.2%} {result['eval_time']:<8.3f}s")

    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test and visualize circle packing programs"
    )

    parser.add_argument(
        "program_path",
        nargs="+",
        help="Path(s) to the program file(s) to test"
    )

    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Single or multiple programs
    if len(args.program_path) == 1:
        result = evaluate_program(args.program_path[0], visualize=not args.no_visualize)
        results = [result]
    else:
        results = compare_programs(args.program_path)

    # Save to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
