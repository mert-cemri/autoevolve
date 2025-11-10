#!/usr/bin/env python3
"""
Compare results between memory-enabled and non-memory experiments.
Creates a comparison matrix showing average scores by strategy and split.
"""

import pandas as pd
import numpy as np
import os


def load_results(results_dir):
    """Load all_results.csv from a directory."""
    csv_path = os.path.join(results_dir, "all_results.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def create_comparison_matrix():
    """Create comparison matrix of average scores by strategy and split."""

    # Load both datasets
    print("Loading results...")
    df_no_memory = load_results("./results_analysis_no_memory")
    df_memory = load_results("./results_analysis")

    if df_no_memory.empty and df_memory.empty:
        print("Error: No results found in either directory!")
        return

    # Add memory indicator to each dataset
    if not df_no_memory.empty:
        df_no_memory['memory_enabled'] = False
    if not df_memory.empty:
        df_memory['memory_enabled'] = True

    # Combine datasets
    df_combined = pd.concat([df_no_memory, df_memory], ignore_index=True)

    print(f"\nTotal results loaded: {len(df_combined)}")
    print(f"  - Without memory: {len(df_no_memory)}")
    print(f"  - With memory: {len(df_memory)}")

    # Create strategy labels with memory suffix
    df_combined['strategy_full'] = df_combined.apply(
        lambda row: f"{row['strategy']}_memory" if row['memory_enabled'] else row['strategy'],
        axis=1
    )

    # Create pivot table: rows=strategy, columns=split, values=mean combined_score
    pivot = df_combined.pivot_table(
        values='combined_score',
        index='strategy_full',
        columns='split',
        aggfunc='mean'
    )

    # Calculate row and column means
    pivot['AVERAGE'] = pivot.mean(axis=1)
    pivot.loc['AVERAGE'] = pivot.mean(axis=0)

    # Sort rows: first non-memory, then memory versions
    strategy_order = [
        'map_elites',
        'beam_search',
        'mcts',
        'best_of_n',
        'map_elites_memory',
        'beam_search_memory',
        'mcts_memory',
        'best_of_n_memory',
        'AVERAGE'
    ]

    # Only include strategies that exist
    strategy_order = [s for s in strategy_order if s in pivot.index]
    pivot = pivot.reindex(strategy_order)

    # Sort columns alphabetically (except AVERAGE)
    cols = sorted([c for c in pivot.columns if c != 'AVERAGE']) + ['AVERAGE']
    pivot = pivot[cols]

    return pivot, df_combined


def print_comparison_matrix(pivot):
    """Print the comparison matrix in a nice format."""
    print("\n" + "=" * 100)
    print("COMPARISON MATRIX: Average Combined Score by Strategy and Split")
    print("=" * 100)
    print()

    # Print with nice formatting
    print(pivot.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\n" + "=" * 100)
    print("LEGEND:")
    print("  - Strategies without '_memory' suffix: Standard evolution (no memory)")
    print("  - Strategies with '_memory' suffix: Memory-augmented evolution")
    print("  - AVERAGE row/column: Mean across all strategies/splits")
    print("=" * 100)


def create_improvement_matrix(pivot):
    """Calculate improvement from memory for each strategy."""
    print("\n" + "=" * 100)
    print("MEMORY IMPROVEMENT ANALYSIS")
    print("=" * 100)
    print()

    improvements = {}

    strategies = ['map_elites', 'beam_search', 'mcts', 'best_of_n']
    splits = [c for c in pivot.columns if c != 'AVERAGE']

    improvement_data = []

    for strategy in strategies:
        memory_strategy = f"{strategy}_memory"

        if strategy in pivot.index and memory_strategy in pivot.index:
            for split in splits:
                if split in pivot.columns:
                    base_score = pivot.loc[strategy, split]
                    memory_score = pivot.loc[memory_strategy, split]

                    if pd.notna(base_score) and pd.notna(memory_score):
                        improvement = memory_score - base_score
                        pct_improvement = (improvement / abs(base_score)) * 100 if base_score != 0 else 0

                        improvement_data.append({
                            'strategy': strategy,
                            'split': split,
                            'base_score': base_score,
                            'memory_score': memory_score,
                            'improvement': improvement,
                            'pct_improvement': pct_improvement
                        })

    if improvement_data:
        improvement_df = pd.DataFrame(improvement_data)

        # Pivot for improvement values
        improvement_pivot = improvement_df.pivot_table(
            values='improvement',
            index='strategy',
            columns='split',
            aggfunc='mean'
        )

        # Add average column
        improvement_pivot['AVERAGE'] = improvement_pivot.mean(axis=1)

        print("Absolute Improvement (Memory - Base):")
        print(improvement_pivot.to_string(float_format=lambda x: f'{x:+.4f}'))

        print("\n")

        # Pivot for percentage improvement
        pct_pivot = improvement_df.pivot_table(
            values='pct_improvement',
            index='strategy',
            columns='split',
            aggfunc='mean'
        )

        pct_pivot['AVERAGE'] = pct_pivot.mean(axis=1)

        print("Percentage Improvement (%):")
        print(pct_pivot.to_string(float_format=lambda x: f'{x:+.2f}%'))

        print("\n" + "=" * 100)
        print("SUMMARY:")
        avg_improvement = improvement_df['improvement'].mean()
        avg_pct = improvement_df['pct_improvement'].mean()
        positive_count = (improvement_df['improvement'] > 0).sum()
        total_count = len(improvement_df)

        print(f"  - Average absolute improvement: {avg_improvement:+.4f}")
        print(f"  - Average percentage improvement: {avg_pct:+.2f}%")
        print(f"  - Memory wins: {positive_count}/{total_count} ({100*positive_count/total_count:.1f}%)")
        print("=" * 100)
    else:
        print("No comparable data found for improvement analysis.")


def create_ranking_analysis(pivot):
    """Analyze rankings of strategies."""
    print("\n" + "=" * 100)
    print("STRATEGY RANKINGS (by split)")
    print("=" * 100)
    print()

    splits = [c for c in pivot.columns if c != 'AVERAGE' and c != 'AVERAGE']

    # Remove AVERAGE row for ranking
    pivot_no_avg = pivot.drop('AVERAGE', errors='ignore')

    ranking_data = []

    for split in splits:
        if split in pivot_no_avg.columns:
            # Sort strategies by score for this split (descending)
            sorted_strategies = pivot_no_avg[split].sort_values(ascending=False)

            print(f"{split}:")
            for rank, (strategy, score) in enumerate(sorted_strategies.items(), 1):
                if pd.notna(score):
                    print(f"  {rank}. {strategy:25s}: {score:.4f}")
                    ranking_data.append({
                        'split': split,
                        'rank': rank,
                        'strategy': strategy,
                        'score': score
                    })
            print()

    # Calculate average rank per strategy
    if ranking_data:
        ranking_df = pd.DataFrame(ranking_data)
        avg_ranks = ranking_df.groupby('strategy')['rank'].mean().sort_values()

        print("Average Rank Across All Splits:")
        for strategy, avg_rank in avg_ranks.items():
            print(f"  {strategy:25s}: {avg_rank:.2f}")

    print("=" * 100)


def save_comparison_csv(pivot, df_combined):
    """Save comparison results to CSV."""
    output_dir = "./results_comparison"
    os.makedirs(output_dir, exist_ok=True)

    # Save pivot table
    pivot_path = os.path.join(output_dir, "comparison_matrix.csv")
    pivot.to_csv(pivot_path)
    print(f"\nComparison matrix saved to: {pivot_path}")

    # Save combined raw data
    combined_path = os.path.join(output_dir, "combined_results.csv")
    df_combined.to_csv(combined_path, index=False)
    print(f"Combined raw data saved to: {combined_path}")


def main():
    """Main function."""
    print("=" * 100)
    print("MEMORY VS NO-MEMORY RESULTS COMPARISON")
    print("=" * 100)

    # Create comparison matrix
    pivot, df_combined = create_comparison_matrix()

    if pivot is None or pivot.empty:
        print("No data to compare.")
        return

    # Print main comparison matrix
    print_comparison_matrix(pivot)

    # Create improvement analysis
    create_improvement_matrix(pivot)

    # Create ranking analysis
    create_ranking_analysis(pivot)

    # Save results
    save_comparison_csv(pivot, df_combined)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
