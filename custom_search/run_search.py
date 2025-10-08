#!/usr/bin/env python3
"""
CLI for running custom search strategies
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from custom_search.best_of_n import BestOfNSearch
from custom_search.beam_search import BeamSearch
from custom_search.mcts_search import MCTSSearch


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_output_dir(strategy_name: str, base_dir: str = "custom_search/results") -> str:
    """
    Generate timestamped output directory

    Format: {base_dir}/{strategy_name}/run_{YYYY-MM-DD_HH-MM-SS}
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(base_dir) / strategy_name / f"run_{timestamp}"
    return str(output_dir)


def setup_logging():
    """Configure verbose logging"""
    # Create formatter with more detail
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # Suppress overly verbose openai logs
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def run_best_of_n(config: dict):
    """Run Best of N search"""
    search = BestOfNSearch(
        initial_program_path=config['initial_program'],
        evaluator_path=config['evaluator'],
        output_dir=config['output_dir'],
        strategy_name="best_of_n",
        num_eval_problems=config.get('num_eval_problems', 10),
        model=config.get('model', 'gpt-5'),
        agent_model=config.get('agent_model', 'gpt-5-mini'),
        temperature=config.get('temperature', 0.8),
        max_tokens=config.get('max_tokens', 16000)
    )

    best = search.search(
        n=config.get('n', 4),
        iterations=config.get('iterations', 10)
    )

    print(f"\n{'=' * 60}")
    print(f"Best of N Search Complete!")
    print(f"Best Program Score: {best.score:.4f}")
    print(f"Best Program Metrics: {best.metrics}")
    print(f"Results saved to: {config['output_dir']}")
    print(f"{'=' * 60}\n")


def run_beam_search(config: dict):
    """Run Beam Search"""
    search = BeamSearch(
        initial_program_path=config['initial_program'],
        evaluator_path=config['evaluator'],
        output_dir=config['output_dir'],
        strategy_name="beam_search",
        num_eval_problems=config.get('num_eval_problems', 10),
        model=config.get('model', 'gpt-5'),
        agent_model=config.get('agent_model', 'gpt-5-mini'),
        temperature=config.get('temperature', 0.8),
        max_tokens=config.get('max_tokens', 16000)
    )

    best = search.search(
        beam_width=config.get('beam_width', 4),
        branch_factor=config.get('branch_factor', 8),
        iterations=config.get('iterations', 10)
    )

    print(f"\n{'=' * 60}")
    print(f"Beam Search Complete!")
    print(f"Best Program Score: {best.score:.4f}")
    print(f"Best Program Metrics: {best.metrics}")
    print(f"Results saved to: {config['output_dir']}")
    print(f"{'=' * 60}\n")


def run_mcts(config: dict):
    """Run MCTS"""
    search = MCTSSearch(
        initial_program_path=config['initial_program'],
        evaluator_path=config['evaluator'],
        output_dir=config['output_dir'],
        strategy_name="mcts",
        num_eval_problems=config.get('num_eval_problems', 10),
        model=config.get('model', 'gpt-5'),
        agent_model=config.get('agent_model', 'gpt-5-mini'),
        temperature=config.get('temperature', 0.8),
        max_tokens=config.get('max_tokens', 16000)
    )

    best = search.search(
        iterations=config.get('iterations', 50),
        expansion_width=config.get('expansion_width', 3),
        exploration_constant=config.get('exploration_constant', 1.414)
    )

    print(f"\n{'=' * 60}")
    print(f"MCTS Complete!")
    print(f"Best Program Score: {best.score:.4f}")
    print(f"Best Program Metrics: {best.metrics}")
    print(f"Results saved to: {config['output_dir']}")
    print(f"Tree stats saved to: {config['output_dir']}/mcts_tree_stats.json")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run custom search strategies for code evolution"
    )

    parser.add_argument(
        "strategy",
        choices=["best_of_n", "beam_search", "mcts"],
        help="Search strategy to use"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (defaults to config/<strategy>_config.yaml)"
    )

    parser.add_argument(
        "--initial-program",
        type=str,
        help="Path to initial program (overrides config)"
    )

    parser.add_argument(
        "--evaluator",
        type=str,
        help="Path to evaluator (overrides config)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = Path(__file__).parent / "config" / f"{args.strategy}_config.yaml"

    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Get custom_search directory (where this script is located)
    custom_search_dir = Path(__file__).parent

    # Override with command line arguments
    if args.initial_program:
        config['initial_program'] = args.initial_program
    if args.evaluator:
        config['evaluator'] = args.evaluator

    # Generate timestamped output directory if not explicitly provided
    if args.output_dir:
        config['output_dir'] = args.output_dir
    else:
        config['output_dir'] = generate_output_dir(args.strategy)

    # Resolve paths relative to custom_search directory
    initial_program_path = custom_search_dir / config['initial_program']
    evaluator_path = custom_search_dir / config['evaluator']

    # Validate paths
    if not initial_program_path.exists():
        print(f"Error: Initial program not found: {initial_program_path}")
        sys.exit(1)

    if not evaluator_path.exists():
        print(f"Error: Evaluator not found: {evaluator_path}")
        sys.exit(1)

    # Update config with absolute paths
    config['initial_program'] = str(initial_program_path)
    config['evaluator'] = str(evaluator_path)

    # Setup logging
    setup_logging()

    # Run search
    print(f"\n{'=' * 60}")
    print(f"Running {args.strategy.upper()}")
    print(f"Initial Program: {config['initial_program']}")
    print(f"Evaluator: {config['evaluator']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"{'=' * 60}\n")

    if args.strategy == "best_of_n":
        run_best_of_n(config)
    elif args.strategy == "beam_search":
        run_beam_search(config)
    elif args.strategy == "mcts":
        run_mcts(config)


if __name__ == "__main__":
    main()
