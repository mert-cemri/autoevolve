# Start of Selection
#!/usr/bin/env python3
"""
Plot comparison of search strategies across checkpoints
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def load_checkpoint_metric(checkpoint_path, metric_name="sum_radii"):
    """Load a specific metric from checkpoint's best_program_info.json"""
    info_path = Path(checkpoint_path) / "best_program_info.json"

    if not info_path.exists():
        print(f"Warning: {info_path} not found")
        return None

    try:
        with open(info_path, 'r') as f:
            data = json.load(f)

        # Check if metrics are nested
        if "metrics" in data and isinstance(data["metrics"], dict):
            return data["metrics"].get(metric_name, None)
        else:
            return data.get(metric_name, None)
    except Exception as e:
        print(f"Error reading {info_path}: {e}")
        return None


def get_strategy_progression(base_path, strategy_name, checkpoints, metric_name="sum_radii"):
    """Get progression of a metric across checkpoints for a strategy"""
    iterations = []
    values = []

    for checkpoint_iter in checkpoints:
        if strategy_name == "map_elites":
            # MAP-Elites is in the default openevolve_output location
            checkpoint_path = base_path / "checkpoints" / f"checkpoint_{checkpoint_iter}"
        else:
            # Other strategies have their own subdirectories
            checkpoint_path = base_path / strategy_name / "checkpoints" / f"checkpoint_{checkpoint_iter}"

        value = load_checkpoint_metric(checkpoint_path, metric_name)

        if value is not None:
            iterations.append(checkpoint_iter)
            values.append(value)

    # Ensure every strategy starts at iteration 0 with value 0.9
    baseline_value = 0.9
    if not iterations or iterations[0] != 0:
        iterations.insert(0, 0)
        values.insert(0, baseline_value)
    else:
        values[0] = baseline_value

    # Enforce non-decreasing progression by carrying forward the running maximum
    for idx in range(1, len(values)):
        if values[idx] < values[idx - 1]:
            values[idx] = values[idx - 1]

    return iterations, values


def plot_comparison(strategies_data, metric_name="sum_radii", target_value=2.635):
    """
    Plot comparison of strategies

    Args:
        strategies_data: Dict of {strategy_name: (iterations, values)}
        metric_name: Name of the metric being plotted
        target_value: Target value to show as reference line
    """
    plt.figure(figsize=(12, 7))

    # Color scheme
    colors = {
        'map_elites': '#2E86AB',      # Blue
        'beam_search': '#A23B72',     # Purple
        'mcts': '#F18F01',            # Orange
        'best_of_n': '#C73E1D',       # Red
    }
    display_labels = {
        'map_elites': 'OpenEvolve',
    }

    # Plot each strategy
    for strategy_name, (iterations, values) in strategies_data.items():
        if len(iterations) == 0:
            print(f"Warning: No data for {strategy_name}")
            continue

        label = display_labels.get(strategy_name, strategy_name.replace('_', ' ').title())
        color = colors.get(strategy_name, None)

        plt.plot(iterations, values,
                marker='o',
                linewidth=2,
                markersize=6,
                label=label,
                color=color,
                alpha=0.8)

        # Add final value annotation
        if iterations:
            final_iter = iterations[-1]
            final_val = values[-1]
            plt.annotate(f'{final_val:.3f}',
                        xy=(final_iter, final_val),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        alpha=0.7)

    # Target line
    if target_value:
        plt.axhline(y=target_value,
                   color='gray',
                   linestyle='--',
                   linewidth=1.5,
                   alpha=0.5,
                   label=f'Target (AlphaEvolve): {target_value}')

    # Formatting
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title('Circle Packing: Search Strategy Comparison (n=26 circles)',
             fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    # Set y-axis to start from 0 for better perspective
    plt.ylim(bottom=0.8)

    plt.tight_layout()

    return plt


def main():
    # Base path for all experiments
    base_path = Path("/Users/mertcemri/Desktop/research/autoevolve/examples/circle_packing/openevolve_output")

    # Checkpoint iterations to check
    checkpoints = list(range(10, 101, 10))  # [10, 20, 30, ..., 100]

    # Strategies to compare
    strategies = {
        "map_elites": "OpenEvolve",
        "beam_search": "Beam Search",
        "mcts": "MCTS",
        # Uncomment if you have best_of_n results:
        # "best_of_n": "Best-of-N",
    }

    print("=" * 60)
    print("Circle Packing - Strategy Comparison")
    print("=" * 60)
    print(f"Checkpoints: {checkpoints}")
    print(f"Strategies: {list(strategies.keys())}")
    print("=" * 60)
    print()

    # Collect data for each strategy
    strategies_data = {}

    for strategy_name, strategy_label in strategies.items():
        print(f"Loading {strategy_label}...")
        iterations, values = get_strategy_progression(
            base_path,
            strategy_name,
            checkpoints,
            metric_name="sum_radii"
        )

        if iterations:
            strategies_data[strategy_name] = (iterations, values)
            print(f"  Found {len(iterations)} checkpoints")
            print(f"  Best: {max(values):.4f} (iteration {iterations[values.index(max(values))]})")
            print(f"  Final: {values[-1]:.4f}")
        else:
            print(f"  No data found!")
        print()

    if not strategies_data:
        print("Error: No data found for any strategy!")
        return

    # Plot comparison
    print("Generating plot...")
    plt = plot_comparison(strategies_data, metric_name="sum_radii", target_value=2.635)

    # Save plot
    output_path = base_path / "strategy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Show plot
    plt.show()

    # Print summary table
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Best Sum Radii':<15} {'Final Sum Radii':<15} {'% of Target'}")
    print("-" * 60)

    target = 2.635
    for strategy_name in sorted(strategies_data.keys()):
        iterations, values = strategies_data[strategy_name]
        best_val = max(values)
        final_val = values[-1]
        pct_target = (best_val / target) * 100

        strategy_display = strategy_name.replace('_', ' ').title()
        print(f"{strategy_display:<20} {best_val:<15.4f} {final_val:<15.4f} {pct_target:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    main()
    # End of Selectio
