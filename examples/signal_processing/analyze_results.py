#!/usr/bin/env python3
"""
Analyze and compare results from signal processing experiments across different search strategies.

This script collects results from experiments comparing Beam Search, MCTS, Best-of-N,
and MAP-Elites strategies on the signal processing task.

Supports multiple independent runs (openevolve_output_round1, openevolve_output_round2, etc.)
to compute mean and standard deviation across runs.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# Search strategies to analyze
STRATEGIES = {
    "map_elites": {"path": "openevolve_output", "display_name": "MAP-Elites"},
    "best_of_n": {"path": "openevolve_output/best_of_n", "display_name": "Best-of-N"},
    "beam_search": {"path": "openevolve_output/beam_search", "display_name": "Beam Search"},
    "mcts": {"path": "openevolve_output/mcts", "display_name": "MCTS"}
}


def find_round_directories(base_path: Path = Path(".")) -> List[Path]:
    """
    Find all round directories (openevolve_output, openevolve_output_round1, etc.).

    Args:
        base_path: Base directory to search in

    Returns:
        List of paths to round directories, sorted by round number
    """
    round_dirs = []

    # Check for main output directory
    main_output = base_path / "openevolve_output"
    if main_output.exists():
        round_dirs.append(("main", main_output))

    # Check for round directories
    for item in base_path.iterdir():
        if item.is_dir():
            match = re.match(r'openevolve_output_round(\d+)', item.name)
            if match:
                round_num = int(match.group(1))
                round_dirs.append((f"round{round_num}", item))

    # Sort by round number (main first, then round1, round2, etc.)
    def sort_key(x):
        if x[0] == "main":
            return 0
        else:
            return int(x[0].replace("round", ""))

    round_dirs.sort(key=sort_key)

    return round_dirs


def find_checkpoints(output_dir: Path) -> List[Path]:
    """
    Find all checkpoint directories in an output directory.

    Args:
        output_dir: Path to the output directory

    Returns:
        List of checkpoint directory paths, sorted by checkpoint number
    """
    checkpoints_dir = output_dir / "checkpoints"

    if not checkpoints_dir.exists():
        return []

    checkpoints = []
    for item in checkpoints_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint_"):
            checkpoints.append(item)

    # Sort by checkpoint number
    checkpoints.sort(key=lambda p: int(p.name.split("_")[1]))

    return checkpoints


def load_checkpoint_metadata(checkpoint_dir: Path) -> Optional[Dict]:
    """
    Load metadata from a checkpoint.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        Metadata dictionary or None if not found
    """
    metadata_file = checkpoint_dir / "metadata.json"

    if not metadata_file.exists():
        return None

    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {metadata_file}: {e}")
        return None


def load_best_program(output_dir: Path) -> Optional[Dict]:
    """
    Load the best program information.

    Args:
        output_dir: Path to the output directory

    Returns:
        Best program info dictionary or None if not found
    """
    best_info_file = output_dir / "best" / "best_program_info.json"

    if not best_info_file.exists():
        return None

    try:
        with open(best_info_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {best_info_file}: {e}")
        return None


def extract_metrics_from_program(program: Dict) -> Dict:
    """
    Extract metrics from a program dictionary.

    Args:
        program: Program dictionary

    Returns:
        Dictionary of extracted metrics
    """
    metrics = {}

    if 'metrics' in program:
        m = program['metrics']

        # Primary metrics from evaluator
        metrics['overall_score'] = m.get('overall_score', np.nan)  # Primary selection metric
        metrics['composite_score'] = m.get('composite_score', np.nan)

        # Component scores
        metrics['smoothness_score'] = m.get('smoothness_score', np.nan)
        metrics['responsiveness_score'] = m.get('responsiveness_score', np.nan)
        metrics['accuracy_score'] = m.get('accuracy_score', np.nan)
        metrics['efficiency_score'] = m.get('efficiency_score', np.nan)

        # Raw values
        metrics['slope_changes'] = m.get('slope_changes', np.nan)
        metrics['lag_error'] = m.get('lag_error', np.nan)
        metrics['avg_error'] = m.get('avg_error', np.nan)
        metrics['false_reversals'] = m.get('false_reversals', np.nan)
        metrics['correlation'] = m.get('correlation', np.nan)
        metrics['noise_reduction'] = m.get('noise_reduction', np.nan)

        # Other metrics
        metrics['execution_time'] = m.get('execution_time', np.nan)
        metrics['success_rate'] = m.get('success_rate', np.nan)

    # Iteration info
    metrics['iteration'] = program.get('iteration', np.nan)
    metrics['generation'] = program.get('generation', np.nan)
    metrics['program_id'] = program.get('id', 'unknown')

    return metrics


def collect_strategy_evolution(strategy_name: str, output_dir: Path) -> pd.DataFrame:
    """
    Collect evolution history for a single strategy.

    Args:
        strategy_name: Name of the strategy
        output_dir: Path to the strategy's output directory

    Returns:
        DataFrame with evolution history
    """
    if not output_dir.exists():
        print(f"Output directory not found for {strategy_name}: {output_dir}")
        return pd.DataFrame()

    checkpoints = find_checkpoints(output_dir)

    if not checkpoints:
        print(f"No checkpoints found for {strategy_name}")
        return pd.DataFrame()

    results = []

    for checkpoint_dir in checkpoints:
        metadata = load_checkpoint_metadata(checkpoint_dir)

        if metadata and 'best_program' in metadata:
            metrics = extract_metrics_from_program(metadata['best_program'])
            metrics['strategy'] = strategy_name
            metrics['checkpoint'] = checkpoint_dir.name
            metrics['checkpoint_iteration'] = metadata.get('iteration', np.nan)

            results.append(metrics)

    return pd.DataFrame(results)


def collect_final_results_all_rounds(base_path: Path = Path(".")) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect final best results from all strategies across multiple rounds.

    Args:
        base_path: Base directory containing round directories

    Returns:
        Tuple of (all_results_df, aggregated_stats_df)
        - all_results_df: All individual results from each round
        - aggregated_stats_df: Mean, std, min, max across rounds for each strategy
    """
    all_results = []

    # Find all round directories
    round_dirs = find_round_directories(base_path)

    if not round_dirs:
        print("No round directories found. Looking for single openevolve_output directory...")
        round_dirs = [("main", Path("./openevolve_output"))]

    print(f"Found {len(round_dirs)} round(s) of results")
    print()

    for round_name, round_path in round_dirs:
        print(f"=" * 80)
        print(f"Processing {round_name}: {round_path}")
        print("=" * 80)

        for strategy_key, strategy_info in STRATEGIES.items():
            display_name = strategy_info['display_name']

            # Handle MAP-Elites special case (no subdirectory)
            if strategy_key == "map_elites":
                output_dir = round_path
            else:
                # Extract relative path from strategy_info
                rel_path = strategy_info['path'].replace("openevolve_output/", "")
                output_dir = round_path / rel_path

            print(f"  {display_name:15s}: ", end="")

            best_program = load_best_program(output_dir)

            if best_program:
                metrics = extract_metrics_from_program(best_program)
                metrics['strategy'] = display_name
                metrics['strategy_key'] = strategy_key
                metrics['round'] = round_name
                metrics['output_dir'] = str(output_dir)

                all_results.append(metrics)
                score = metrics.get('overall_score', np.nan)
                if not pd.isna(score):
                    print(f"✓ score={score:.4f}")
                else:
                    print("✓ (no score)")
            else:
                print("✗ No best program found")

        print()

    print(f"Total results collected: {len(all_results)}")

    if not all_results:
        return pd.DataFrame(), pd.DataFrame()

    # Create DataFrame with all results
    df_all = pd.DataFrame(all_results)

    # Compute aggregate statistics across rounds
    metric_cols = ['overall_score', 'composite_score', 'smoothness_score', 'responsiveness_score',
                   'accuracy_score', 'efficiency_score', 'slope_changes', 'lag_error',
                   'avg_error', 'false_reversals', 'correlation', 'noise_reduction',
                   'execution_time', 'success_rate', 'iteration', 'generation']

    agg_stats = []

    for strategy in df_all['strategy'].unique():
        strategy_data = df_all[df_all['strategy'] == strategy]
        n_runs = len(strategy_data)

        stats = {'strategy': strategy, 'n_runs': n_runs}

        for col in metric_cols:
            if col in strategy_data.columns:
                values = strategy_data[col].dropna()
                if len(values) > 0:
                    stats[f'{col}_mean'] = values.mean()
                    stats[f'{col}_std'] = values.std() if len(values) > 1 else 0.0
                    stats[f'{col}_min'] = values.min()
                    stats[f'{col}_max'] = values.max()
                else:
                    stats[f'{col}_mean'] = np.nan
                    stats[f'{col}_std'] = np.nan
                    stats[f'{col}_min'] = np.nan
                    stats[f'{col}_max'] = np.nan

        agg_stats.append(stats)

    df_agg = pd.DataFrame(agg_stats)

    return df_all, df_agg


def collect_all_evolution_histories(base_dir: Path = Path("./openevolve_output")) -> pd.DataFrame:
    """
    Collect full evolution history from all strategies.

    Args:
        base_dir: Base directory containing all strategy outputs

    Returns:
        DataFrame with evolution histories from all strategies
    """
    all_data = []

    print("\nCollecting evolution histories from all strategies...")

    for strategy_key, strategy_info in STRATEGIES.items():
        strategy_path = strategy_info['path']
        display_name = strategy_info['display_name']

        # Handle MAP-Elites special case
        if strategy_key == "map_elites":
            output_dir = base_dir
        else:
            output_dir = base_dir.parent / strategy_path

        print(f"\nProcessing {display_name}...")
        df = collect_strategy_evolution(display_name, output_dir)

        if not df.empty:
            all_data.append(df)
            print(f"  Collected {len(df)} checkpoints")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal evolution records collected: {len(combined_df)}")

    return combined_df


def print_aggregated_statistics(df_agg: pd.DataFrame):
    """
    Print aggregated statistics (mean ± std) across multiple runs.

    Args:
        df_agg: DataFrame with aggregated statistics
    """
    print("\n" + "=" * 80)
    print("AGGREGATED STATISTICS - SIGNAL PROCESSING")
    print("(Mean ± Std across multiple runs)")
    print("=" * 80)

    if df_agg.empty:
        print("No results to summarize.")
        return

    print("\n1. Overall Score by Strategy (Mean ± Std):")
    print("-" * 80)
    print(f"{'Strategy':<15s} {'Runs':>5s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 80)

    # Sort by mean overall score
    df_sorted = df_agg.sort_values('overall_score_mean', ascending=False, na_position='last')

    for _, row in df_sorted.iterrows():
        strategy = row['strategy']
        n_runs = int(row['n_runs'])
        mean = row.get('overall_score_mean', np.nan)
        std = row.get('overall_score_std', np.nan)
        min_val = row.get('overall_score_min', np.nan)
        max_val = row.get('overall_score_max', np.nan)

        if not pd.isna(mean):
            print(f"{strategy:<15s} {n_runs:>5d} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")
        else:
            print(f"{strategy:<15s} {n_runs:>5d} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}")

    print("\n2. Component Scores by Strategy (Mean ± Std):")
    print("-" * 80)
    component_cols = ['smoothness_score', 'responsiveness_score', 'accuracy_score', 'efficiency_score']
    component_names = {'smoothness_score': 'Smoothness', 'responsiveness_score': 'Responsiveness',
                       'accuracy_score': 'Accuracy', 'efficiency_score': 'Efficiency'}

    for _, row in df_sorted.iterrows():
        print(f"\n{row['strategy']} (n={int(row['n_runs'])}):")
        for col in component_cols:
            mean_col = f'{col}_mean'
            std_col = f'{col}_std'
            if mean_col in row and not pd.isna(row[mean_col]):
                mean = row[mean_col]
                std = row[std_col]
                name = component_names[col]
                print(f"  {name:20s}: {mean:8.4f} ± {std:7.4f}")

    print("\n3. Raw Metrics by Strategy (Mean ± Std):")
    print("-" * 80)
    raw_cols = ['slope_changes', 'lag_error', 'avg_error', 'false_reversals', 'correlation', 'noise_reduction']
    raw_names = {'slope_changes': 'Slope Changes', 'lag_error': 'Lag Error', 'avg_error': 'Avg Error',
                 'false_reversals': 'False Reversals', 'correlation': 'Correlation', 'noise_reduction': 'Noise Reduction'}

    for _, row in df_sorted.iterrows():
        print(f"\n{row['strategy']} (n={int(row['n_runs'])}):")
        for col in raw_cols:
            mean_col = f'{col}_mean'
            std_col = f'{col}_std'
            if mean_col in row and not pd.isna(row[mean_col]):
                mean = row[mean_col]
                std = row[std_col]
                name = raw_names[col]
                print(f"  {name:20s}: {mean:8.4f} ± {std:7.4f}")

    print("\n4. Convergence Information (Mean ± Std Iterations):")
    print("-" * 80)
    for _, row in df_sorted.iterrows():
        mean_iter = row.get('iteration_mean', np.nan)
        std_iter = row.get('iteration_std', np.nan)

        if not pd.isna(mean_iter):
            print(f"  {row['strategy']:15s}: {mean_iter:6.1f} ± {std_iter:5.1f}")
        else:
            print(f"  {row['strategy']:15s}: N/A")

    # Best strategy
    print("\n5. Best Strategy (by mean overall score):")
    print("-" * 80)

    if 'overall_score_mean' in df_sorted.columns:
        valid_scores = df_sorted['overall_score_mean'].notna()
        if valid_scores.any():
            best_row = df_sorted[valid_scores].iloc[0]
            mean = best_row['overall_score_mean']
            std = best_row['overall_score_std']
            print(f"  Winner: {best_row['strategy']} (mean score: {mean:.4f} ± {std:.4f})")
        else:
            print("  No valid scores available")
    else:
        print("  overall_score_mean column not found in results")

    print("\n" + "=" * 80)


def save_results(df_all: pd.DataFrame, df_agg: pd.DataFrame, df_history: pd.DataFrame,
                output_dir: str = "./results_analysis"):
    """
    Save results to CSV files.

    Args:
        df_all: DataFrame with all individual results from each round
        df_agg: DataFrame with aggregated statistics (mean, std, etc.)
        df_history: DataFrame with evolution histories
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save all individual results
    if not df_all.empty:
        all_results_path = os.path.join(output_dir, "all_results_by_round.csv")
        df_all.to_csv(all_results_path, index=False)
        print(f"\nAll individual results saved to: {all_results_path}")

    # Save aggregated statistics
    if not df_agg.empty:
        agg_path = os.path.join(output_dir, "aggregated_statistics.csv")
        df_agg.to_csv(agg_path, index=False)
        print(f"Aggregated statistics saved to: {agg_path}")

    # Save evolution history
    if not df_history.empty:
        history_path = os.path.join(output_dir, "evolution_history.csv")
        df_history.to_csv(history_path, index=False)
        print(f"Evolution history saved to: {history_path}")


def main():
    """Main function to run the analysis."""
    print("=" * 80)
    print("SIGNAL PROCESSING EXPERIMENTS ANALYSIS")
    print("=" * 80)

    # Use current directory as base path
    base_path = Path(".")

    # Collect results from all rounds
    df_all, df_agg = collect_final_results_all_rounds(base_path)

    # Collect evolution histories (from first round if available)
    round_dirs = find_round_directories(base_path)
    df_history = pd.DataFrame()
    if round_dirs:
        first_round_path = round_dirs[0][1]
        df_history = collect_all_evolution_histories(first_round_path)

    if df_all.empty:
        print("\nNo results found. Please run experiments first.")
        print("\nExpected directory structure:")
        print("  openevolve_output/best/best_program_info.json  (MAP-Elites)")
        print("  openevolve_output/best_of_n/best/best_program_info.json")
        print("  openevolve_output/beam_search/best/best_program_info.json")
        print("  openevolve_output/mcts/best/best_program_info.json")
        print("\nOr for multiple rounds:")
        print("  openevolve_output_round1/, openevolve_output_round2/, etc.")
        return None, None, None

    # Print aggregated statistics
    print_aggregated_statistics(df_agg)

    # Save results
    save_results(df_all, df_agg, df_history)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    return df_all, df_agg, df_history


if __name__ == "__main__":
    df_all, df_agg, df_history = main()
