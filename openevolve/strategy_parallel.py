"""
Strategy-aware parallel controller

Parallel execution that works with different search strategies.
"""

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openevolve.config import Config
from openevolve.database import Program
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""
    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None


def _worker_init(config_dict: dict, evaluation_file: str, parent_env: dict = None) -> None:
    """Initialize worker process with necessary components"""
    import os

    # Set environment from parent process
    if parent_env:
        os.environ.update(parent_env)

    global _worker_config
    global _worker_evaluation_file
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    # Store config for later use
    from openevolve.config import (
        Config,
        DatabaseConfig,
        EvaluatorConfig,
        LLMConfig,
        PromptConfig,
        LLMModelConfig,
    )

    # Reconstruct model objects
    models = [LLMModelConfig(**m) for m in config_dict["llm"]["models"]]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict["llm"]["evaluator_models"]]

    # Create LLM config with models
    llm_dict = config_dict["llm"].copy()
    llm_dict["models"] = models
    llm_dict["evaluator_models"] = evaluator_models
    llm_config = LLMConfig(**llm_dict)

    # Create other configs
    prompt_config = PromptConfig(**config_dict["prompt"])
    database_config = DatabaseConfig(**config_dict["database"])
    evaluator_config = EvaluatorConfig(**config_dict["evaluator"])

    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        **{
            k: v
            for k, v in config_dict.items()
            if k not in ["llm", "prompt", "database", "evaluator"]
        },
    )
    _worker_evaluation_file = evaluation_file

    # These will be lazily initialized on first use
    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None


def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use"""
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler

    if _worker_llm_ensemble is None:
        from openevolve.llm.ensemble import LLMEnsemble
        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        from openevolve.prompt.sampler import PromptSampler
        _worker_prompt_sampler = PromptSampler(_worker_config.prompt)

    if _worker_evaluator is None:
        from openevolve.evaluator import Evaluator
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.prompt.sampler import PromptSampler

        # Create evaluator-specific components
        evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
        evaluator_prompt = PromptSampler(_worker_config.prompt)
        evaluator_prompt.set_templates("evaluator_system_message")

        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            evaluator_llm,
            evaluator_prompt,
            database=None,
            suffix=getattr(_worker_config, 'file_suffix', '.py'),
        )


def _run_iteration_worker(
    iteration: int,
    strategy_snapshot: Dict[str, Any],
    parent_id: str,
    best_program_dicts: List[Dict[str, Any]],
    inspiration_dicts: List[Dict[str, Any]]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    try:
        # Lazy initialization
        _lazy_init_worker_components()

        # Reconstruct parent program
        parent = Program(**strategy_snapshot["programs"][parent_id])

        # Reconstruct context programs
        best_programs = [Program(**prog_dict) for prog_dict in best_program_dicts]
        inspirations = [Program(**prog_dict) for prog_dict in inspiration_dicts]

        # Build prompt
        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in best_programs],
            top_programs=[p.to_dict() for p in (best_programs + inspirations)],
            inspirations=[p.to_dict() for p in inspirations],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=None,
            feature_dimensions=strategy_snapshot.get("feature_dimensions", []),
        )

        iteration_start = time.time()

        # Generate code modification
        try:
            llm_response = asyncio.run(
                _worker_llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return SerializableResult(error=f"LLM generation failed: {str(e)}", iteration=iteration)

        if llm_response is None:
            return SerializableResult(error="LLM returned None response", iteration=iteration)

        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import extract_diffs, apply_diff, format_diff_summary
            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(error="No valid diffs found in response", iteration=iteration)
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite
            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(error="No valid code found in response", iteration=iteration)
            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration,
            )

        # Evaluate the child program
        import uuid
        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))

        # Get artifacts
        artifacts = _worker_evaluator.get_pending_artifacts(child_id)

        # Create child program
        child_program = Program(
            id=child_id,
            code=child_code,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
            },
        )

        iteration_time = time.time() - iteration_start

        return SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt,
            llm_response=llm_response,
            artifacts=artifacts,
            iteration=iteration,
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(error=str(e), iteration=iteration)


class StrategyParallelController:
    """Controller for process-based parallel evolution with search strategies"""

    def __init__(
        self,
        config: Config,
        evaluation_file: str,
        strategy,
        evolution_tracer=None,
        file_suffix: str = ".py"
    ):
        self.config = config
        self.evaluation_file = evaluation_file
        self.strategy = strategy
        self.evolution_tracer = evolution_tracer
        self.file_suffix = file_suffix

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations

        logger.info(f"Initialized strategy parallel controller with {self.num_workers} workers")
        logger.info(f"Using search strategy: {strategy.get_strategy_name()}")

    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        from dataclasses import asdict

        return {
            "llm": {
                "models": [asdict(m) for m in config.llm.models],
                "evaluator_models": [asdict(m) for m in config.llm.evaluator_models],
                "api_base": config.llm.api_base,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
            },
            "prompt": asdict(config.prompt),
            "database": asdict(config.database),
            "evaluator": asdict(config.evaluator),
            "max_iterations": config.max_iterations,
            "checkpoint_interval": config.checkpoint_interval,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "random_seed": config.random_seed,
            "diff_based_evolution": config.diff_based_evolution,
            "max_code_length": config.max_code_length,
            "language": config.language,
            "file_suffix": self.file_suffix,
        }

    def start(self) -> None:
        """Start the process pool"""
        config_dict = self._serialize_config(self.config)

        # Pass current environment to worker processes
        import os
        current_env = dict(os.environ)

        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file, current_env),
        )

        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
    ):
        """Run evolution with process-based parallelism"""
        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        # Track pending futures
        pending_futures: Dict[int, Future] = {}
        batch_size = min(self.num_workers * 2, max_iterations)

        # Submit initial batch
        current_iteration = start_iteration
        for _ in range(min(batch_size, max_iterations)):
            future = self._submit_iteration(current_iteration)
            if future:
                pending_futures[current_iteration] = future
            current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0

        # Early stopping tracking
        early_stopping_enabled = self.config.early_stopping_patience is not None
        if early_stopping_enabled:
            best_score = float("-inf")
            iterations_without_improvement = 0
            logger.info(
                f"Early stopping enabled: patience={self.config.early_stopping_patience}, "
                f"threshold={self.config.convergence_threshold}, "
                f"metric={self.config.early_stopping_metric}"
            )

        # Process results as they complete
        while (
            pending_futures
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break

            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            # Process completed result
            future = pending_futures.pop(completed_iteration)

            try:
                timeout_seconds = self.config.evaluator.timeout + 30
                result = future.result(timeout=timeout_seconds)

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Add to strategy
                    self.strategy.add_program(child_program, iteration=completed_iteration)

                    # Log evolution trace
                    if self.evolution_tracer and result.parent_id:
                        parent_program = self.strategy.programs.get(result.parent_id)
                        if parent_program:
                            self.evolution_tracer.log_trace(
                                iteration=completed_iteration,
                                parent_program=parent_program,
                                child_program=child_program,
                                prompt=result.prompt,
                                llm_response=result.llm_response,
                                artifacts=result.artifacts,
                                island_id=0,  # Strategy-specific
                                metadata={
                                    "iteration_time": result.iteration_time,
                                    "changes": child_program.metadata.get("changes", ""),
                                }
                            )

                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id[:8]}... "
                        f"(parent: {result.parent_id[:8] if result.parent_id else 'None'}...) "
                        f"completed in {result.iteration_time:.2f}s"
                    )

                    if child_program.metrics:
                        metrics_str = ", ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in child_program.metrics.items()
                            ]
                        )
                        logger.info(f"Metrics: {metrics_str}")

                    # Check for new best
                    if self.strategy.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id[:8]}..."
                        )

                    # Checkpoint callback
                    if (
                        completed_iteration > 0
                        and completed_iteration % self.config.checkpoint_interval == 0
                    ):
                        logger.info(f"Checkpoint interval reached at iteration {completed_iteration}")
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)

                    # Check target score
                    if target_score is not None and child_program.metrics:
                        numeric_metrics = [
                            v for v in child_program.metrics.values() if isinstance(v, (int, float))
                        ]
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(
                                    f"Target score {target_score} reached at iteration {completed_iteration}"
                                )
                                break

                    # Check early stopping
                    if early_stopping_enabled and child_program.metrics:
                        current_score = None
                        if self.config.early_stopping_metric in child_program.metrics:
                            current_score = child_program.metrics[self.config.early_stopping_metric]
                        elif self.config.early_stopping_metric == "combined_score":
                            current_score = safe_numeric_average(child_program.metrics)
                        else:
                            logger.warning(
                                f"Early stopping metric '{self.config.early_stopping_metric}' not found"
                            )
                            current_score = safe_numeric_average(child_program.metrics)

                        if current_score is not None and isinstance(current_score, (int, float)):
                            improvement = current_score - best_score
                            if improvement >= self.config.convergence_threshold:
                                best_score = current_score
                                iterations_without_improvement = 0
                            else:
                                iterations_without_improvement += 1

                            if iterations_without_improvement >= self.config.early_stopping_patience:
                                self.early_stopping_triggered = True
                                logger.info(
                                    f"🛑 Early stopping triggered at iteration {completed_iteration}: "
                                    f"No improvement for {iterations_without_improvement} iterations "
                                    f"(best score: {best_score:.4f})"
                                )
                                break

            except FutureTimeoutError:
                logger.error(
                    f"⏰ Iteration {completed_iteration} timed out after {timeout_seconds}s"
                )
                future.cancel()
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1

            # Submit next iteration
            if (
                next_iteration < total_iterations
                and not self.shutdown_event.is_set()
                and not self.early_stopping_triggered
            ):
                future = self._submit_iteration(next_iteration)
                if future:
                    pending_futures[next_iteration] = future
                    next_iteration += 1

        # Handle shutdown
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        # Log completion reason
        if self.early_stopping_triggered:
            logger.info("✅ Evolution completed - Early stopping triggered")
        elif self.shutdown_event.is_set():
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
            logger.info("✅ Evolution completed - Maximum iterations reached")

        return self.strategy.get_best_program()

    def _submit_iteration(self, iteration: int) -> Optional[Future]:
        """Submit an iteration to the process pool"""
        try:
            # Sample parent from strategy
            parent = self.strategy.sample_parent(iteration)

            if parent is None:
                logger.warning(f"Could not sample parent for iteration {iteration}")
                return None

            # Get context programs from strategy
            best_programs, inspirations = self.strategy.get_context_programs(parent, iteration)

            # Create strategy snapshot
            strategy_snapshot = self.strategy.get_snapshot()

            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                strategy_snapshot,
                parent.id,
                [p.to_dict() for p in best_programs],
                [p.to_dict() for p in inspirations],
            )

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
