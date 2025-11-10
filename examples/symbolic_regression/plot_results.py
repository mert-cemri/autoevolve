#!/usr/bin/env python3
"""
Visualize and compare results from symbolic regression experiments.

This script creates plots comparing performance across different search strategies.
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


def plot_strategy_comparison(df: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Create comparison plots for different strategies.

    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Symbolic Regression: Search Strategy Comparison', fontsize=16, fontweight='bold')

    # 1. Box plot: Combined Score by Strategy
    ax1 = axes[0, 0]
    strategy_order = df.groupby('strategy')['combined_score'].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x='strategy', y='combined_score', order=strategy_order, ax=ax1)
    ax1.set_title('Combined Score Distribution by Strategy', fontweight='bold')
    ax1.set_xlabel('Search Strategy')
    ax1.set_ylabel('Combined Score')
    ax1.grid(axis='y', alpha=0.3)

    # 2. Violin plot: Combined Score by Strategy
    ax2 = axes[0, 1]
    sns.violinplot(data=df, x='strategy', y='combined_score', order=strategy_order, ax=ax2)
    ax2.set_title('Combined Score Distribution (Violin Plot)', fontweight='bold')
    ax2.set_xlabel('Search Strategy')
    ax2.set_ylabel('Combined Score')
    ax2.grid(axis='y', alpha=0.3)

    # 3. Bar plot: Mean Score by Split and Strategy
    ax3 = axes[1, 0]
    pivot = df.pivot_table(values='combined_score', index='split', columns='strategy', aggfunc='mean')
    pivot.plot(kind='bar', ax=ax3)
    ax3.set_title('Mean Combined Score by Split and Strategy', fontweight='bold')
    ax3.set_xlabel('Problem Split')
    ax3.set_ylabel('Mean Combined Score')
    ax3.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(axis='y', alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Count plot: Number of Experiments per Strategy
    ax4 = axes[1, 1]
    strategy_counts = df['strategy'].value_counts().reindex(strategy_order)
    strategy_counts.plot(kind='bar', ax=ax4, color='skyblue')
    ax4.set_title('Number of Completed Experiments by Strategy', fontweight='bold')
    ax4.set_xlabel('Search Strategy')
    ax4.set_ylabel('Count')
    ax4.grid(axis='y', alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nStrategy comparison plot saved to: {plot_path}")
    plt.close()


def plot_split_analysis(df: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Create detailed analysis plots for each problem split.

    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    splits = df['split'].unique()
    n_splits = len(splits)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance Analysis by Problem Split', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for idx, split in enumerate(sorted(splits)):
        if idx >= 4:
            break

        ax = axes[idx]
        split_data = df[df['split'] == split]

        # Box plot for this split
        strategy_order = split_data.groupby('strategy')['combined_score'].mean().sort_values(ascending=False).index
        sns.boxplot(data=split_data, x='strategy', y='combined_score', order=strategy_order, ax=ax)

        ax.set_title(f'{split.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Combined Score')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'split_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Split analysis plot saved to: {plot_path}")
    plt.close()


def plot_convergence_analysis(df: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Plot convergence (iteration) analysis.

    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')

    # 1. Scatter: Score vs Iteration by Strategy
    ax1 = axes[0]
    for strategy in df['strategy'].unique():
        strategy_data = df[df['strategy'] == strategy]
        ax1.scatter(strategy_data['iteration'], strategy_data['combined_score'],
                   label=strategy, alpha=0.6, s=30)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Combined Score')
    ax1.set_title('Score vs Iteration Number', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Box plot: Iteration distribution by Strategy
    ax2 = axes[1]
    strategy_order = df.groupby('strategy')['iteration'].median().sort_values().index
    sns.boxplot(data=df, x='strategy', y='iteration', order=strategy_order, ax=ax2)
    ax2.set_title('Iteration Distribution by Strategy', fontweight='bold')
    ax2.set_xlabel('Search Strategy')
    ax2.set_ylabel('Iteration at Best Solution')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'convergence_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Convergence analysis plot saved to: {plot_path}")
    plt.close()


def create_statistical_summary(df: pd.DataFrame, output_dir: str = "./results_analysis"):
    """
    Create detailed statistical summary tables.

    Args:
        df: DataFrame with results
        output_dir: Directory to save summary
    """
    summary_path = os.path.join(output_dir, 'statistical_summary.txt')

    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SUMMARY OF SYMBOLIC REGRESSION EXPERIMENTS\n")
        f.write("=" * 80 + "\n\n")

        # Overall statistics
        f.write("1. OVERALL STATISTICS BY STRATEGY\n")
        f.write("-" * 80 + "\n")
        overall_stats = df.groupby('strategy')['combined_score'].describe()
        f.write(overall_stats.to_string())
        f.write("\n\n")

        # Per-split statistics
        f.write("2. STATISTICS BY SPLIT AND STRATEGY\n")
        f.write("-" * 80 + "\n")
        for split in sorted(df['split'].unique()):
            f.write(f"\n{split.upper()}:\n")
            split_stats = df[df['split'] == split].groupby('strategy')['combined_score'].describe()
            f.write(split_stats.to_string())
            f.write("\n")

        # Win rates (how often each strategy wins per problem)
        f.write("\n3. WIN RATES (Best strategy per problem)\n")
        f.write("-" * 80 + "\n")

        # Find best strategy for each problem
        best_per_problem = df.loc[df.groupby('problem_id')['combined_score'].idxmax()]
        win_counts = best_per_problem['strategy'].value_counts()
        total_problems = len(df['problem_id'].unique())

        f.write(f"Total unique problems: {total_problems}\n\n")
        for strategy, wins in win_counts.items():
            win_rate = (wins / total_problems) * 100
            f.write(f"{strategy:15s}: {wins:3d} wins ({win_rate:5.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Statistical summary saved to: {summary_path}")


def main():
    """Main function to create all visualizations."""
    print("=" * 80)
    print("CREATING VISUALIZATIONS FOR SYMBOLIC REGRESSION EXPERIMENTS")
    print("=" * 80)

    # Collect data
    df = analyze_results.collect_all_results()

    if df.empty:
        print("\nNo results found. Please run experiments first.")
        return

    output_dir = "./results_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Create plots
    print("\nGenerating plots...")
    plot_strategy_comparison(df, output_dir)
    plot_split_analysis(df, output_dir)
    plot_convergence_analysis(df, output_dir)

    # Create statistical summary
    print("\nGenerating statistical summary...")
    create_statistical_summary(df, output_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print(f"All results saved to: {output_dir}/")
    print("=" * 80)

    return df


if __name__ == "__main__":
    try:
        df = main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMissing dependencies. Install with:")
        print("  pip install matplotlib seaborn")
