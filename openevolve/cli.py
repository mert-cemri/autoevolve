"""
Command-line interface for OpenEvolve
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional

from openevolve import OpenEvolve
from openevolve.config import Config, load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="OpenEvolve - Evolutionary coding agent")

    parser.add_argument("initial_program", help="Path to the initial program file")

    parser.add_argument(
        "evaluation_file", help="Path to the evaluation file containing an 'evaluate' function"
    )

    parser.add_argument("--config", "-c", help="Path to configuration file (YAML)", default=None)

    parser.add_argument("--output", "-o", help="Output directory for results", default=None)

    parser.add_argument(
        "--iterations", "-i", help="Maximum number of iterations", type=int, default=None
    )

    parser.add_argument(
        "--target-score", "-t", help="Target score to reach", type=float, default=None
    )

    parser.add_argument(
        "--log-level",
        "-l",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        help="Path to checkpoint directory to resume from (e.g., openevolve_output/checkpoints/checkpoint_50)",
        default=None,
    )

    parser.add_argument("--api-base", help="Base URL for the LLM API", default=None)

    parser.add_argument("--primary-model", help="Primary LLM model name", default=None)

    parser.add_argument("--secondary-model", help="Secondary LLM model name", default=None)

    # Search strategy selection
    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--best-of-n",
        action="store_true",
        help="Use Best-of-N search strategy (N independent lineages)"
    )
    strategy_group.add_argument(
        "--beam-search",
        action="store_true",
        help="Use Beam Search strategy (keep top M, branch to N)"
    )
    strategy_group.add_argument(
        "--mcts",
        action="store_true",
        help="Use MCTS strategy (Monte Carlo Tree Search with UCT)"
    )

    return parser.parse_args()


async def main_async() -> int:
    """
    Main asynchronous entry point

    Returns:
        Exit code
    """
    args = parse_args()

    # Check if files exist
    if not os.path.exists(args.initial_program):
        print(f"Error: Initial program file '{args.initial_program}' not found")
        return 1

    if not os.path.exists(args.evaluation_file):
        print(f"Error: Evaluation file '{args.evaluation_file}' not found")
        return 1

    # Determine search strategy from CLI arguments
    strategy_name = None
    if args.best_of_n:
        strategy_name = "best_of_n"
    elif args.beam_search:
        strategy_name = "beam_search"
    elif args.mcts:
        strategy_name = "mcts"

    # If strategy is specified, auto-select config file if not provided
    if strategy_name and not args.config:
        import os
        base_dir = os.path.dirname(args.initial_program)
        default_config = os.path.join(base_dir, f"config_{strategy_name}.yaml")
        if os.path.exists(default_config):
            args.config = default_config
            print(f"Auto-selected config: {default_config}")

    # Create config object with command-line overrides
    config = None
    if args.api_base or args.primary_model or args.secondary_model:
        # Load base config from file or defaults
        config = load_config(args.config)

        # Apply command-line overrides
        if args.api_base:
            config.llm.api_base = args.api_base
            print(f"Using API base: {config.llm.api_base}")

        if args.primary_model:
            config.llm.primary_model = args.primary_model
            print(f"Using primary model: {config.llm.primary_model}")

        if args.secondary_model:
            config.llm.secondary_model = args.secondary_model
            print(f"Using secondary model: {config.llm.secondary_model}")

        # Rebuild models list to apply CLI overrides
        if args.primary_model or args.secondary_model:
            config.llm.rebuild_models()
            print(f"Applied CLI model overrides - active models:")
            for i, model in enumerate(config.llm.models):
                print(f"  Model {i+1}: {model.name} (weight: {model.weight})")

    # Initialize with appropriate strategy
    try:
        if strategy_name:
            # Use strategy-aware controller
            from openevolve.strategy_controller import OpenEvolveWithStrategy

            print(f"\n🔍 Using search strategy: {strategy_name.upper().replace('_', '-')}")
            openevolve = OpenEvolveWithStrategy(
                initial_program_path=args.initial_program,
                evaluation_file=args.evaluation_file,
                strategy_name=strategy_name,
                config=config,
                config_path=args.config if config is None else None,
                output_dir=args.output,
            )
        else:
            # Use default MAP-Elites
            print(f"\n🔍 Using default MAP-Elites search strategy")
            openevolve = OpenEvolve(
                initial_program_path=args.initial_program,
                evaluation_file=args.evaluation_file,
                config=config,
                config_path=args.config if config is None else None,
                output_dir=args.output,
            )

        # Load from checkpoint if specified
        if args.checkpoint:
            if not os.path.exists(args.checkpoint):
                print(f"Error: Checkpoint directory '{args.checkpoint}' not found")
                return 1
            print(f"Loading checkpoint from {args.checkpoint}")
            if hasattr(openevolve, 'strategy') and hasattr(openevolve.strategy, 'load'):
                openevolve.strategy.load(args.checkpoint)
            else:
                openevolve.database.load(args.checkpoint)
            print(f"Checkpoint loaded successfully")

        # Override log level if specified
        if args.log_level:
            logging.getLogger().setLevel(getattr(logging, args.log_level))

        # Run evolution
        best_program = await openevolve.run(
            iterations=args.iterations,
            target_score=args.target_score,
            checkpoint_path=args.checkpoint,
        )

        # Get the checkpoint path
        checkpoint_dir = os.path.join(openevolve.output_dir, "checkpoints")
        latest_checkpoint = None
        if os.path.exists(checkpoint_dir):
            checkpoints = [
                os.path.join(checkpoint_dir, d)
                for d in os.listdir(checkpoint_dir)
                if os.path.isdir(os.path.join(checkpoint_dir, d))
            ]
            if checkpoints:
                latest_checkpoint = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
                )[-1]

        print(f"\nEvolution complete!")
        print(f"Best program metrics:")
        for name, value in best_program.metrics.items():
            # Handle mixed types: format numbers as floats, others as strings
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.4f}")
            else:
                print(f"  {name}: {value}")

        if latest_checkpoint:
            print(f"\nLatest checkpoint saved at: {latest_checkpoint}")
            print(f"To resume, use: --checkpoint {latest_checkpoint}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    """
    Main entry point

    Returns:
        Exit code
    """
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
