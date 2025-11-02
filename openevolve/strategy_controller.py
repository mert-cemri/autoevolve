"""
Strategy-aware OpenEvolve controller

Minimal wrapper around OpenEvolve that uses different search strategies
while reusing all existing infrastructure (prompts, parallel execution, etc.)
"""

import logging
import os
import time
import uuid
from typing import Optional

from openevolve.config import Config, load_config
from openevolve.controller import OpenEvolve
from openevolve.database import Program
from openevolve.search_strategies import (
    BestOfNStrategy,
    BeamSearchStrategy,
    MCTSStrategy,
)
from openevolve.strategy_parallel import StrategyParallelController

logger = logging.getLogger(__name__)


class OpenEvolveWithStrategy(OpenEvolve):
    """
    OpenEvolve controller using pluggable search strategies.

    Inherits from OpenEvolve to reuse all infrastructure, but replaces
    the database and parallel controller with strategy-aware versions.
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        strategy_name: str,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize with specified search strategy.

        Args:
            initial_program_path: Path to initial program
            evaluation_file: Path to evaluator
            strategy_name: One of: "best_of_n", "beam_search", "mcts"
            config_path: Path to config YAML
            config: Config object (if provided, config_path ignored)
            output_dir: Output directory
        """
        # Load config first
        if config is not None:
            self.config = config
        else:
            self.config = load_config(config_path)

        # Store strategy name
        self.strategy_name = strategy_name

        # Set up output directory with strategy-specific subdirectory
        if output_dir:
            self.output_dir = output_dir
        else:
            # Default: openevolve_output/<strategy_name>
            base_output = os.path.join(
                os.path.dirname(initial_program_path), "openevolve_output"
            )
            self.output_dir = os.path.join(base_output, strategy_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # Call parent's setup methods (but skip database initialization)
        self._setup_logging()

        # Set random seed
        if self.config.random_seed is not None:
            self._set_random_seed()

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()

        # Extract language and file extension
        from openevolve.utils.code_utils import extract_code_language
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code)

        self.file_extension = os.path.splitext(initial_program_path)[1] or ".py"
        if not self.file_extension.startswith("."):
            self.file_extension = f".{self.file_extension}"

        if not hasattr(self.config, 'file_suffix') or self.config.file_suffix == ".py":
            self.config.file_suffix = self.file_extension

        # Initialize LLM ensembles
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.prompt.sampler import PromptSampler
        from openevolve.evaluator import Evaluator

        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
        self.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models)

        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler.set_templates("evaluator_system_message")

        # Create search strategy instead of database
        self.strategy = self._create_strategy(strategy_name)

        # For compatibility, expose strategy as database
        self.database = self.strategy

        # Initialize evaluator
        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=None,  # Strategy handles storage
            suffix=self.file_extension,
        )
        self.evaluation_file = evaluation_file

        logger.info(f"Initialized OpenEvolve with {strategy_name} strategy")

        # Initialize evolution tracer
        self.evolution_tracer = None
        if self.config.evolution_trace.enabled:
            from openevolve.evolution_trace import EvolutionTracer
            trace_output_path = self.config.evolution_trace.output_path
            if not trace_output_path:
                trace_output_path = os.path.join(
                    self.output_dir,
                    f"evolution_trace.{self.config.evolution_trace.format}"
                )
            self.evolution_tracer = EvolutionTracer(
                output_path=trace_output_path,
                format=self.config.evolution_trace.format,
                include_code=self.config.evolution_trace.include_code,
                include_prompts=self.config.evolution_trace.include_prompts,
                enabled=True,
                buffer_size=self.config.evolution_trace.buffer_size,
                compress=self.config.evolution_trace.compress
            )

        # Initialize parallel controller
        self.parallel_controller = None

    def _create_strategy(self, strategy_name: str):
        """Create the appropriate search strategy."""
        if strategy_name == "best_of_n":
            return BestOfNStrategy(self.config)
        elif strategy_name == "beam_search":
            return BeamSearchStrategy(self.config)
        elif strategy_name == "mcts":
            return MCTSStrategy(self.config)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        import hashlib

        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

        base_seed = str(self.config.random_seed).encode("utf-8")
        llm_seed = int(hashlib.md5(base_seed + b"llm").hexdigest()[:8], 16) % (2**31)

        self.config.llm.random_seed = llm_seed
        for model_cfg in self.config.llm.models:
            if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                model_cfg.random_seed = llm_seed
        for model_cfg in self.config.llm.evaluator_models:
            if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                model_cfg.random_seed = llm_seed

        logger.info(f"Set random seed to {self.config.random_seed}")

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:
        """
        Run evolution with strategy-aware parallel controller.

        This overrides the parent's run() method to use StrategyParallelController
        instead of ProcessParallelController.
        """
        max_iterations = iterations or self.config.max_iterations

        # Determine starting iteration
        start_iteration = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = getattr(self.strategy, 'last_iteration', 0) + 1
            logger.info(f"Resuming from checkpoint at iteration {start_iteration}")

        # Add initial program if starting fresh
        should_add_initial = (
            start_iteration == 0
            and len(self.strategy.programs) == 0
        )

        if should_add_initial:
            logger.info("Adding initial program")
            initial_program_id = str(uuid.uuid4())

            # Evaluate initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id, iteration=start_iteration
            )

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                language=self.config.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
            )

            self.strategy.add_program(initial_program, iteration=start_iteration)
            logger.info(f"Initial program: {initial_program.id} (metrics: {initial_metrics})")

        # Initialize parallel controller with strategy
        try:
            self.parallel_controller = StrategyParallelController(
                self.config,
                self.evaluation_file,
                self.strategy,
                self.evolution_tracer,
                file_suffix=self.config.file_suffix
            )

            # Set up signal handlers
            import signal
            def signal_handler(signum, frame):
                logger.info("Graceful shutdown requested...")
                self.parallel_controller.request_shutdown()

                def force_exit_handler(signum, frame):
                    logger.info("Force exit - terminating immediately")
                    import sys
                    sys.exit(0)

                signal.signal(signal.SIGINT, force_exit_handler)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            self.parallel_controller.start()

            # Run evolution
            evolution_start = start_iteration
            evolution_iterations = max_iterations

            if should_add_initial and start_iteration == 0:
                evolution_start = 1

            await self._run_evolution_with_checkpoints(
                evolution_start, evolution_iterations, target_score
            )

        finally:
            if self.parallel_controller:
                self.parallel_controller.stop()
                self.parallel_controller = None

            if self.evolution_tracer:
                self.evolution_tracer.close()

        # Get best program
        best_program = self.strategy.get_best_program()

        if best_program:
            from openevolve.utils.format_utils import format_metrics_safe
            logger.info(
                f"Evolution complete. Best program: {format_metrics_safe(best_program.metrics)}"
            )
            self._save_best_program(best_program)
            return best_program
        else:
            logger.warning("No valid programs found")
            return None

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:
        """Run evolution with checkpoint support."""
        logger.info(f"Running evolution with {self.strategy_name} strategy")

        # Run the evolution
        await self.parallel_controller.run_evolution(
            start_iteration,
            max_iterations,
            target_score,
            checkpoint_callback=self._save_checkpoint
        )

        # Save final checkpoint if needed
        final_iteration = start_iteration + max_iterations - 1
        if final_iteration > 0 and final_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint(final_iteration)

    def _save_checkpoint(self, iteration: int) -> None:
        """Save checkpoint with strategy state."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save strategy snapshot
        import json
        snapshot_path = os.path.join(checkpoint_path, "strategy.json")
        with open(snapshot_path, 'w') as f:
            json.dump(self.strategy.get_snapshot(), f, indent=2)

        # Save best program
        best_program = self.strategy.get_best_program()
        if best_program:
            best_program_path = os.path.join(checkpoint_path, f"best_program{self.file_extension}")
            with open(best_program_path, "w") as f:
                f.write(best_program.code)

            info_path = os.path.join(checkpoint_path, "best_program_info.json")
            with open(info_path, "w") as f:
                json.dump({
                    "id": best_program.id,
                    "generation": best_program.generation,
                    "iteration": best_program.iteration_found,
                    "current_iteration": iteration,
                    "metrics": best_program.metrics,
                    "strategy": self.strategy_name,
                    "timestamp": time.time(),
                }, f, indent=2)

        logger.info(f"Saved checkpoint at iteration {iteration}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint (basic implementation)."""
        import json
        snapshot_path = os.path.join(checkpoint_path, "strategy.json")
        if os.path.exists(snapshot_path):
            with open(snapshot_path, 'r') as f:
                snapshot = json.load(f)
            # Basic restoration - can be enhanced
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"No strategy snapshot found in {checkpoint_path}")
