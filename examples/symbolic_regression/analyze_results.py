#!/usr/bin/env python3
"""
Analyze and compare results from symbolic regression experiments across different search strategies.

This script collects results from all experiments and creates a comprehensive DataFrame
comparing performance across Beam Search, MCTS, Best-of-N, and MAP-Elites strategies.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# Define splits and their configurations (from scripts.sh)
SPLITS = {
    "bio_pop_growth": {"count": 24, "prefix": "BPG"},
    "chem_react": {"count": 36, "prefix": "CRK"},
    "matsci": {"count": 25, "prefix": "MatSci"},
    "phys_osc": {"count": 44, "prefix": "PO"},
}

# Search strategies to look for
# Note: "openevolve" refers to the default MAP-Elites strategy (no subfolder in output)
STRATEGIES = ["beam_search", "mcts", "best_of_n", "openevolve"]


def find_best_program_info(
    base_dir: str, split_name: str, problem_id: str, strategy: str
) -> Optional[Path]:
    """
    Find the best_program_info.json file for a given experiment.

    Args:
        base_dir: Base directory for problems
        split_name: Name of the split (e.g., 'bio_pop_growth')
        problem_id: Problem ID (e.g., 'BPG0')
        strategy: Strategy name (e.g., 'beam_search', 'mcts', 'best_of_n', 'openevolve')

    Returns:
        Path to best_program_info.json if found, None otherwise
    """
    problem_dir = Path(base_dir) / split_name / problem_id

    # Special case: 'openevolve' is the default MAP-Elites strategy
    # Path: openevolve_output/best/best_program_info.json (no strategy subfolder)
    if strategy == "openevolve":
        info_file = problem_dir / "openevolve_output_gradient" / "best" / "best_program_info.json"
    else:
        # Other strategies: openevolve_output/{strategy}/best/best_program_info.json
        info_file = problem_dir / "openevolve_output_gradient" / strategy / "best" / "best_program_info.json"

    if info_file.exists():
        return info_file

    return None


def load_result(info_file: Path) -> Optional[Dict]:
    """
    Load and parse a best_program_info.json file.

    Args:
        info_file: Path to the JSON file

    Returns:
        Dictionary with result data, or None if error
    """
    try:
        with open(info_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {info_file}: {e}")
        return None


def extract_metrics(result: Dict) -> Dict:
    """
    Extract relevant metrics from a result dictionary.

    Args:
        result: Result dictionary from best_program_info.json

    Returns:
        Dictionary with extracted metrics
    """
    metrics = {}

    # Extract common metrics
    if 'metrics' in result:
        m = result['metrics']
        metrics['combined_score'] = m.get('combined_score', np.nan)
        metrics['negative_mse'] = m.get('negative_mse', np.nan)
        metrics['mse_train_score'] = m.get('mse_train_score', np.nan)
        metrics['raw_mse_train'] = m.get('raw_mse_train', np.nan)
        metrics['can_run'] = m.get('can_run', np.nan)
        metrics['optimization_success'] = m.get('optimization_success', False)

    # Extract iteration info
    metrics['iteration'] = result.get('iteration', np.nan)
    metrics['generation'] = result.get('generation', np.nan)
    metrics['program_id'] = result.get('id', 'unknown')

    # Extract parent info if available
    metrics['parent_id'] = result.get('parent_id', None)

    return metrics


def collect_all_results(base_dir: str = "./problems") -> pd.DataFrame:
    """
    Collect results from all experiments across all splits, problems, and strategies.

    Args:
        base_dir: Base directory containing problem folders

    Returns:
        DataFrame with all results
    """
    results = []

    print("Collecting results from all experiments...")
    print(f"Base directory: {base_dir}")
    print()

    for split_name, split_config in SPLITS.items():
        count = split_config['count']
        prefix = split_config['prefix']

        print(f"Processing split: {split_name} ({count} problems)")

        for i in range(count):
            problem_id = f"{prefix}{i}"

            for strategy in STRATEGIES:
                info_file = find_best_program_info(base_dir, split_name, problem_id, strategy)

                if info_file:
                    result = load_result(info_file)

                    if result:
                        metrics = extract_metrics(result)

                        # Add metadata
                        metrics['split'] = split_name
                        metrics['problem_id'] = problem_id
                        metrics['problem_index'] = i
                        # Use descriptive name for display
                        display_strategy = strategy
                        if strategy == "openevolve":
                            display_strategy = "map_elites"  # More descriptive name for display
                        metrics['strategy'] = display_strategy
                        metrics['strategy_raw'] = strategy  # Keep original for reference
                        metrics['result_file'] = str(info_file)

                        results.append(metrics)

    print(f"\nTotal results collected: {len(results)}")

    if not results:
        print("WARNING: No results found!")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns for better readability
    column_order = [
        'split', 'problem_id', 'problem_index', 'strategy', 'strategy_raw',
        'combined_score', 'mse_train_score', 'raw_mse_train', 'negative_mse',
        'iteration', 'generation', 'can_run', 'optimization_success',
        'program_id', 'parent_id', 'result_file'
    ]

    # Only include columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]

    return df


def print_summary_statistics(df: pd.DataFrame):
    """
    Print summary statistics comparing strategies.

    Args:
        df: DataFrame with results
    """
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if df.empty:
        print("No results to summarize.")
        return

    # Overall statistics by strategy
    print("\n1. Results Count by Strategy:")
    print("-" * 40)
    strategy_counts = df['strategy'].value_counts()
    print(strategy_counts)

    # Performance metrics by strategy
    print("\n2. Combined Score by Strategy (mean ± std):")
    print("-" * 40)
    score_stats = df.groupby('strategy')['combined_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(score_stats)

    print("\n3. MSE Train Score by Strategy (mean ± std):")
    print("-" * 40)
    mse_stats = df.groupby('strategy')['mse_train_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(mse_stats)

    print("\n4. Success Rate by Strategy:")
    print("-" * 40)
    success_rate = df.groupby('strategy')['optimization_success'].mean() * 100
    print(success_rate.apply(lambda x: f"{x:.2f}%"))

    # Per-split statistics
    print("\n5. Combined Score by Split and Strategy:")
    print("-" * 40)
    pivot = df.pivot_table(
        values='combined_score',
        index='split',
        columns='strategy',
        aggfunc='mean'
    )
    print(pivot)

    # Best performing strategy per split
    print("\n6. Best Strategy per Split (by mean combined score):")
    print("-" * 40)
    best_strategy = pivot.idxmax(axis=1)
    for split, strategy in best_strategy.items():
        score = pivot.loc[split, strategy]
        print(f"  {split:20s}: {strategy:15s} (score: {score:.4f})")

    # Overall winner
    print("\n7. Overall Best Strategy (by mean combined score):")
    print("-" * 40)
    overall_means = df.groupby('strategy')['combined_score'].mean()
    best_overall = overall_means.idxmax()
    best_score = overall_means[best_overall]
    print(f"  Winner: {best_overall} (mean score: {best_score:.4f})")

    print("\n" + "=" * 80)


def print_detailed_comparison(df: pd.DataFrame, top_n: int = 10):
    """
    Print detailed comparison of top results.

    Args:
        df: DataFrame with results
        top_n: Number of top results to show
    """
    print("\n" + "=" * 80)
    print(f"TOP {top_n} RESULTS (by combined score)")
    print("=" * 80)

    if df.empty:
        print("No results available.")
        return

    top_results = df.nlargest(top_n, 'combined_score')

    for idx, row in top_results.iterrows():
        print(f"\nRank {idx + 1}:")
        print(f"  Problem: {row['split']}/{row['problem_id']}")
        print(f"  Strategy: {row['strategy']}")
        print(f"  Combined Score: {row['combined_score']:.6f}")
        print(f"  MSE Train Score: {row['mse_train_score']:.6f}")
        print(f"  Raw MSE: {row['raw_mse_train']:.2e}")
        print(f"  Iteration: {row['iteration']}")

    print("\n" + "=" * 80)


def save_results(df: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Save results to CSV files.

    Args:
        df: DataFrame with results
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full results
    full_path = os.path.join(output_dir, "all_results.csv")
    df.to_csv(full_path, index=False)
    print(f"\nFull results saved to: {full_path}")

    # Save summary by strategy
    summary = df.groupby('strategy').agg({
        'combined_score': ['count', 'mean', 'std', 'min', 'max'],
        'mse_train_score': ['mean', 'std'],
        'optimization_success': 'mean'
    })
    summary_path = os.path.join(output_dir, "summary_by_strategy.csv")
    summary.to_csv(summary_path)
    print(f"Summary by strategy saved to: {summary_path}")

    # Save summary by split and strategy
    pivot = df.pivot_table(
        values='combined_score',
        index=['split', 'problem_id'],
        columns='strategy',
        aggfunc='mean'
    )
    pivot_path = os.path.join(output_dir, "scores_by_problem_and_strategy.csv")
    pivot.to_csv(pivot_path)
    print(f"Scores by problem and strategy saved to: {pivot_path}")


def main():
    """Main function to run the analysis."""
    print("=" * 80)
    print("SYMBOLIC REGRESSION EXPERIMENTS ANALYSIS")
    print("=" * 80)

    # Collect all results
    df = collect_all_results()

    if df.empty:
        print("\nNo results found. Please check that experiments have been run.")
        return

    # Print summary statistics
    print_summary_statistics(df)

    # Print detailed comparison
    print_detailed_comparison(df, top_n=10)

    # Save results
    save_results(df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return df


if __name__ == "__main__":
    df = main()
