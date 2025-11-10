#!/usr/bin/env python3
"""
Visualize and compare results from signal processing experiments.

This script creates plots comparing performance across different search strategies
(MAP-Elites, Best-of-N, Beam Search, MCTS) on the signal processing task.
"""

import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the analyze_results module to reuse data collection
import analyze_results


def plot_final_comparison(df: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Create comparison plots for final results from different strategies.

    Args:
        df: DataFrame with final results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have any valid scores
    if 'overall_score' not in df.columns or df['overall_score'].isna().all():
        print("Warning: No valid overall scores found. Skipping final comparison plots.")
        return

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Signal Processing: Search Strategy Comparison', fontsize=16, fontweight='bold')

    # Sort strategies by overall score (NaN values will be at the end)
    df_sorted = df.sort_values('overall_score', ascending=False, na_position='last')
    strategy_order = df_sorted['strategy'].tolist()

    # 1. Overall Score Comparison (Bar Plot)
    ax1 = axes[0, 0]
    colors = sns.color_palette("husl", len(df_sorted))
    # Replace NaN with 0 for plotting
    scores_for_plot = df_sorted['overall_score'].fillna(0)
    bars = ax1.bar(range(len(df_sorted)), scores_for_plot, color=colors)
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels(df_sorted['strategy'], rotation=45, ha='right')
    ax1.set_title('Final Overall Score by Strategy', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Overall Score')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars (only for non-NaN values)
    for i, (bar, score) in enumerate(zip(bars, df_sorted['overall_score'])):
        if not pd.isna(score):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=10)
        else:
            # Label for missing data
            ax1.text(bar.get_x() + bar.get_width()/2., 0.01,
                    'N/A',
                    ha='center', va='bottom', fontsize=10, color='red')

    # 2. Component Scores Comparison (Grouped Bar Plot)
    ax2 = axes[0, 1]
    component_cols = ['smoothness_score', 'responsiveness_score', 'accuracy_score', 'efficiency_score']
    component_labels = ['Smoothness', 'Responsiveness', 'Accuracy', 'Efficiency']

    x = np.arange(len(df_sorted))
    width = 0.2

    for i, (col, label) in enumerate(zip(component_cols, component_labels)):
        if col in df_sorted.columns:
            values = df_sorted[col].fillna(0)
            ax2.bar(x + i*width, values, width, label=label)

    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Score')
    ax2.set_title('Component Scores by Strategy', fontweight='bold', fontsize=12)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(df_sorted['strategy'], rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Raw Metrics Comparison (Grouped Bar Plot)
    ax3 = axes[1, 0]
    raw_cols = ['slope_changes', 'lag_error', 'avg_error', 'false_reversals']
    raw_labels = ['Slope Changes', 'Lag Error', 'Avg Error', 'False Reversals']

    # Normalize raw metrics to 0-1 scale for comparison
    df_normalized = df_sorted.copy()
    for col in raw_cols:
        if col in df_normalized.columns:
            col_max = df_normalized[col].max()
            if col_max > 0:
                df_normalized[col] = df_normalized[col] / col_max

    x = np.arange(len(df_sorted))

    for i, (col, label) in enumerate(zip(raw_cols, raw_labels)):
        if col in df_normalized.columns:
            values = df_normalized[col].fillna(0)
            ax3.bar(x + i*width, values, width, label=label)

    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Normalized Value (0-1)')
    ax3.set_title('Raw Metrics by Strategy (Normalized)', fontweight='bold', fontsize=12)
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(df_sorted['strategy'], rotation=45, ha='right')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Convergence Information (Bar Plot)
    ax4 = axes[1, 1]
    if 'iteration' in df_sorted.columns:
        bars = ax4.bar(range(len(df_sorted)), df_sorted['iteration'], color=colors)
        ax4.set_xticks(range(len(df_sorted)))
        ax4.set_xticklabels(df_sorted['strategy'], rotation=45, ha='right')
        ax4.set_title('Convergence: Iteration of Best Solution', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Iteration Number')
        ax4.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, iteration in zip(bars, df_sorted['iteration']):
            height = bar.get_height()
            if not pd.isna(iteration):
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(iteration)}',
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nStrategy comparison plot saved to: {plot_path}")
    plt.close()


def plot_evolution_history(df: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Plot evolution history over time for all strategies.

    Args:
        df: DataFrame with evolution history
        output_dir: Directory to save plots
    """
    if df.empty:
        print("No evolution history available for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Evolution History: Performance Over Time', fontsize=16, fontweight='bold')

    # 1. Overall Score Evolution
    ax1 = axes[0, 0]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('checkpoint_iteration')
        if 'overall_score' in strategy_data.columns:
            ax1.plot(strategy_data['checkpoint_iteration'], strategy_data['overall_score'],
                    marker='o', label=strategy, linewidth=2, markersize=4)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Overall Score')
    ax1.set_title('Overall Score Evolution', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Smoothness Score Evolution
    ax2 = axes[0, 1]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('checkpoint_iteration')
        if 'smoothness_score' in strategy_data.columns:
            ax2.plot(strategy_data['checkpoint_iteration'], strategy_data['smoothness_score'],
                    marker='o', label=strategy, linewidth=2, markersize=4)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Smoothness Score')
    ax2.set_title('Smoothness Score Evolution', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Responsiveness Score Evolution
    ax3 = axes[1, 0]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('checkpoint_iteration')
        if 'responsiveness_score' in strategy_data.columns:
            ax3.plot(strategy_data['checkpoint_iteration'], strategy_data['responsiveness_score'],
                    marker='o', label=strategy, linewidth=2, markersize=4)

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Responsiveness Score')
    ax3.set_title('Responsiveness Score Evolution', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Accuracy Score Evolution
    ax4 = axes[1, 1]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy].sort_values('checkpoint_iteration')
        if 'accuracy_score' in strategy_data.columns:
            ax4.plot(strategy_data['checkpoint_iteration'], strategy_data['accuracy_score'],
                    marker='o', label=strategy, linewidth=2, markersize=4)

    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Accuracy Score')
    ax4.set_title('Accuracy Score Evolution', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'evolution_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Evolution history plot saved to: {plot_path}")
    plt.close()


def plot_radar_comparison(df: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Create radar/spider plot comparing component scores across strategies.

    Args:
        df: DataFrame with final results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    component_cols = ['smoothness_score', 'responsiveness_score', 'accuracy_score', 'efficiency_score']
    component_labels = ['Smoothness', 'Responsiveness', 'Accuracy', 'Efficiency']

    # Check if we have all component scores
    missing_cols = [col for col in component_cols if col not in df.columns]
    if missing_cols:
        print(f"Skipping radar plot: missing columns {missing_cols}")
        return

    # Number of variables
    num_vars = len(component_cols)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle('Component Score Comparison (Radar Plot)', fontsize=16, fontweight='bold', y=0.95)

    # Plot each strategy
    colors = sns.color_palette("husl", len(df))

    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[col] for col in component_cols]
        values += values[:1]  # Complete the circle

        ax.plot(angles, values, 'o-', linewidth=2, label=row['strategy'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(component_labels, size=11)

    # Set y-axis limits
    ax.set_ylim(0, 1)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'radar_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Radar comparison plot saved to: {plot_path}")
    plt.close()


def create_summary_report(df_all: pd.DataFrame, df_history: pd.DataFrame,
                         output_dir: str = "./results_analysis"):
    """
    Create a text summary report.

    Args:
        df_all: DataFrame with all results (possibly from multiple rounds)
        df_history: DataFrame with evolution history
        output_dir: Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'summary_report.txt')

    # Check if we have multiple rounds
    has_multiple_rounds = 'round' in df_all.columns and df_all['round'].nunique() > 1

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SIGNAL PROCESSING EXPERIMENTS - SUMMARY REPORT\n")
        if has_multiple_rounds:
            f.write(f"({df_all['round'].nunique()} independent runs)\n")
        f.write("=" * 80 + "\n\n")

        if has_multiple_rounds:
            # Compute aggregated statistics
            f.write("1. AGGREGATED RESULTS BY STRATEGY (Mean ± Std)\n")
            f.write("-" * 80 + "\n\n")

            for strategy in sorted(df_all['strategy'].unique()):
                strategy_data = df_all[df_all['strategy'] == strategy]
                n_runs = len(strategy_data)

                f.write(f"{strategy} ({n_runs} runs):\n")

                # Overall score
                if 'overall_score' in strategy_data.columns:
                    scores = strategy_data['overall_score'].dropna()
                    if len(scores) > 0:
                        f.write(f"  Overall Score:       {scores.mean():.4f} ± {scores.std():.4f}\n")

                # Component scores
                for metric in ['smoothness_score', 'responsiveness_score', 'accuracy_score', 'efficiency_score']:
                    if metric in strategy_data.columns:
                        values = strategy_data[metric].dropna()
                        if len(values) > 0:
                            f.write(f"  {metric.replace('_', ' ').title():20s}: {values.mean():.4f} ± {values.std():.4f}\n")

                f.write("\n")
        else:
            # Single round results
            f.write("1. FINAL RESULTS BY STRATEGY\n")
            f.write("-" * 80 + "\n")

            df_sorted = df_all.sort_values('overall_score', ascending=False, na_position='last')

            for idx, (_, row) in enumerate(df_sorted.iterrows(), 1):
                f.write(f"\nRank {idx}: {row['strategy']}\n")
                if pd.notna(row.get('overall_score')):
                    f.write(f"  Overall Score: {row['overall_score']:.6f}\n")
                else:
                    f.write(f"  Overall Score: N/A (experiment not completed or failed)\n")

                if pd.notna(row.get('composite_score')):
                    f.write(f"  Composite Score: {row['composite_score']:.6f}\n")

                if 'smoothness_score' in row and pd.notna(row['smoothness_score']):
                    f.write(f"  Smoothness Score: {row['smoothness_score']:.6f}\n")
                if 'responsiveness_score' in row and pd.notna(row['responsiveness_score']):
                    f.write(f"  Responsiveness Score: {row['responsiveness_score']:.6f}\n")
                if 'accuracy_score' in row and pd.notna(row['accuracy_score']):
                    f.write(f"  Accuracy Score: {row['accuracy_score']:.6f}\n")
                if 'efficiency_score' in row and pd.notna(row['efficiency_score']):
                    f.write(f"  Efficiency Score: {row['efficiency_score']:.6f}\n")

                if 'iteration' in row and not pd.isna(row['iteration']):
                    f.write(f"  Converged at Iteration: {int(row['iteration'])}\n")
                if 'generation' in row and not pd.isna(row['generation']):
                    f.write(f"  Generation: {int(row['generation'])}\n")

        # Raw Metrics
        f.write("\n\n2. RAW METRICS BY STRATEGY\n")
        f.write("-" * 80 + "\n")

        for _, row in df_sorted.iterrows():
            f.write(f"\n{row['strategy']}:\n")

            if 'slope_changes' in row and not pd.isna(row['slope_changes']):
                f.write(f"  Slope Changes: {row['slope_changes']:.4f}\n")
            if 'lag_error' in row and not pd.isna(row['lag_error']):
                f.write(f"  Lag Error: {row['lag_error']:.4f}\n")
            if 'avg_error' in row and not pd.isna(row['avg_error']):
                f.write(f"  Avg Error: {row['avg_error']:.4f}\n")
            if 'false_reversals' in row and not pd.isna(row['false_reversals']):
                f.write(f"  False Reversals: {row['false_reversals']:.4f}\n")
            if 'correlation' in row and not pd.isna(row['correlation']):
                f.write(f"  Correlation: {row['correlation']:.4f}\n")
            if 'noise_reduction' in row and not pd.isna(row['noise_reduction']):
                f.write(f"  Noise Reduction: {row['noise_reduction']:.4f}\n")

        # Winner
        f.write("\n\n3. BEST STRATEGY\n")
        f.write("-" * 80 + "\n")

        # Check if there are any valid scores
        if 'overall_score' in df_sorted.columns:
            valid_scores = df_sorted['overall_score'].notna()
            if valid_scores.any():
                # Get the first row with a valid score
                best_row = df_sorted[valid_scores].iloc[0]
                f.write(f"Winner: {best_row['strategy']}\n")
                f.write(f"Overall Score: {best_row['overall_score']:.6f}\n")
            else:
                f.write("No valid results available (all experiments may have failed or not completed)\n")
        else:
            f.write("overall_score column not found in results\n")

        # Convergence Analysis
        if not df_history.empty:
            f.write("\n\n4. CONVERGENCE ANALYSIS\n")
            f.write("-" * 80 + "\n")

            for strategy in df_history['strategy'].unique():
                strategy_data = df_history[df_history['strategy'] == strategy]
                n_checkpoints = len(strategy_data)

                f.write(f"\n{strategy}:\n")
                f.write(f"  Number of checkpoints: {n_checkpoints}\n")

                if 'overall_score' in strategy_data.columns:
                    max_score = strategy_data['overall_score'].max()
                    final_score = strategy_data.iloc[-1]['overall_score']
                    if pd.notna(max_score):
                        f.write(f"  Max score achieved: {max_score:.6f}\n")
                    if pd.notna(final_score):
                        f.write(f"  Final score: {final_score:.6f}\n")
                else:
                    f.write(f"  Score data not available\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report saved to: {report_path}")


def plot_aggregated_comparison(df_agg: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Create comparison plots with error bars for aggregated results.

    Args:
        df_agg: DataFrame with aggregated statistics (mean, std)
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if we have any valid scores
    if 'overall_score_mean' not in df_agg.columns or df_agg['overall_score_mean'].isna().all():
        print("Warning: No valid overall scores found. Skipping aggregated comparison plots.")
        return

    # Set style
    sns.set_style("whitegrid")

    # Sort by mean overall score
    df_sorted = df_agg.sort_values('overall_score_mean', ascending=False, na_position='last')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Signal Processing: Search Strategy Comparison (Mean ± Std)', fontsize=16, fontweight='bold')

    # 1. Overall Score with error bars
    ax1 = axes[0, 0]
    x_pos = np.arange(len(df_sorted))
    means = df_sorted['overall_score_mean'].fillna(0)
    stds = df_sorted['overall_score_std'].fillna(0)
    colors = sns.color_palette("husl", len(df_sorted))

    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df_sorted['strategy'], rotation=45, ha='right')
    ax1.set_title('Overall Score by Strategy (Mean ± Std)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Overall Score')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        if not pd.isna(mean) and mean > 0:
            ax1.text(i, mean + std, f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=9)

    # 2. Component Scores comparison
    ax2 = axes[0, 1]
    component_cols = ['smoothness_score', 'responsiveness_score', 'accuracy_score', 'efficiency_score']
    component_labels = ['Smoothness', 'Responsiveness', 'Accuracy', 'Efficiency']

    x = np.arange(len(df_sorted))
    width = 0.2

    for i, (col, label) in enumerate(zip(component_cols, component_labels)):
        mean_col = f'{col}_mean'
        if mean_col in df_sorted.columns:
            values = df_sorted[mean_col].fillna(0)
            ax2.bar(x + i*width, values, width, label=label, alpha=0.8)

    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Score')
    ax2.set_title('Component Scores by Strategy (Mean)', fontweight='bold', fontsize=12)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(df_sorted['strategy'], rotation=45, ha='right')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Raw Metrics (lower is better for these)
    ax3 = axes[1, 0]
    raw_cols = ['slope_changes', 'lag_error', 'false_reversals']
    raw_labels = ['Slope Changes', 'Lag Error', 'False Reversals']

    x = np.arange(len(df_sorted))

    for i, (col, label) in enumerate(zip(raw_cols, raw_labels)):
        mean_col = f'{col}_mean'
        std_col = f'{col}_std'
        if mean_col in df_sorted.columns:
            means_raw = df_sorted[mean_col].fillna(0)
            stds_raw = df_sorted[std_col].fillna(0)
            ax3.bar(x + i*width, means_raw, width, yerr=stds_raw, capsize=3, label=label, alpha=0.8)

    ax3.set_xlabel('Strategy')
    ax3.set_ylabel('Value')
    ax3.set_title('Raw Metrics by Strategy (Mean ± Std, lower is better)', fontweight='bold', fontsize=12)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(df_sorted['strategy'], rotation=45, ha='right')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # 4. Convergence iterations
    ax4 = axes[1, 1]
    if 'iteration_mean' in df_sorted.columns:
        means_iter = df_sorted['iteration_mean'].fillna(0)
        stds_iter = df_sorted['iteration_std'].fillna(0)

        bars = ax4.bar(x_pos, means_iter, yerr=stds_iter, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(df_sorted['strategy'], rotation=45, ha='right')
        ax4.set_title('Convergence Iteration (Mean ± Std)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Iteration Number')
        ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'aggregated_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nAggregated comparison plot saved to: {plot_path}")
    plt.close()


def main():
    """Main function to create all visualizations."""
    print("=" * 80)
    print("CREATING VISUALIZATIONS FOR SIGNAL PROCESSING EXPERIMENTS")
    print("=" * 80)

    # Collect data
    df_all, df_agg, df_history = analyze_results.main()

    if df_all is None or df_all.empty:
        print("\nNo results found. Please run experiments first.")
        return

    output_dir = "./results_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Create plots
    print("\nGenerating plots...")

    # Plot aggregated results with error bars
    if df_agg is not None and not df_agg.empty:
        plot_aggregated_comparison(df_agg, output_dir)

    # Plot individual round results for comparison
    if 'round' in df_all.columns and df_all['round'].nunique() > 1:
        print("Multiple rounds detected - plotting individual round results...")
        # Use first round for detailed plots
        first_round = df_all['round'].iloc[0]
        df_first_round = df_all[df_all['round'] == first_round].copy()
        df_first_round = df_first_round.rename(columns={col: col.replace('_mean', '').replace('_std', '')
                                                        for col in df_first_round.columns})
        plot_final_comparison(df_first_round, output_dir)

    if df_history is not None and not df_history.empty:
        plot_evolution_history(df_history, output_dir)

    if df_agg is not None and not df_agg.empty:
        # Create radar plot from aggregated means
        df_radar = df_agg.copy()
        df_radar = df_radar.rename(columns={
            'smoothness_score_mean': 'smoothness_score',
            'responsiveness_score_mean': 'responsiveness_score',
            'accuracy_score_mean': 'accuracy_score',
            'efficiency_score_mean': 'efficiency_score'
        })
        plot_radar_comparison(df_radar, output_dir)

    # Create summary report
    print("\nGenerating summary report...")
    create_summary_report(df_all, df_history, output_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print(f"All results saved to: {output_dir}/")
    print("=" * 80)

    return df_all, df_agg, df_history


if __name__ == "__main__":
    try:
        df_final, df_history = main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMissing dependencies. Install with:")
        print("  pip install matplotlib seaborn pandas numpy")
