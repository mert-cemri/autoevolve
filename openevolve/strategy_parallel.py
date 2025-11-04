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
from typing import Any, Dict, List, Optional, Tuple

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

    # Configure logging in worker to append to the same log file as controller
    try:
        import logging
        log_path = os.environ.get("OPENEVOLVE_LOG_FILE")
        if log_path:
            root_logger = logging.getLogger()
            # Avoid duplicate handlers if any
            have_file = any(
                isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', None) == log_path
                for h in root_logger.handlers
            )
            if not have_file:
                fh = logging.FileHandler(log_path)
                fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
                root_logger.addHandler(fh)
            # Ensure level is set according to config
            try:
                lvl = config_dict.get("log_level") or "INFO"
                root_logger.setLevel(getattr(logging, str(lvl).upper()))
            except Exception:
                root_logger.setLevel(logging.INFO)
    except Exception:
        # Never fail worker init due to logging setup
        pass

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

        # Log semantic search (supplied by controller) just before building prompt
        # Only log if memory is enabled (indicated by presence of semantic_parent_log in snapshot)
        sem_log = strategy_snapshot.get("semantic_parent_log") if isinstance(strategy_snapshot, dict) else None
        if sem_log is not None:
            logger.info(f"Memory (worker): Starting semantic search logging for iteration {iteration}, parent={parent.id}")
        try:
            if not isinstance(strategy_snapshot, dict):
                logger.warning(f"Memory (worker): strategy_snapshot is not a dict (type: {type(strategy_snapshot)})")
            elif sem_log is not None:
                # Debug: log what keys are in strategy_snapshot (only when memory is enabled)
                snapshot_keys = list(strategy_snapshot.keys())
                logger.info(f"Memory (worker): strategy_snapshot keys for iteration {iteration}: {snapshot_keys}")
                
                sem_details = strategy_snapshot.get("semantic_parent_details")
                logger.info(f"Memory (worker): sem_log type={type(sem_log)}, sem_details type={type(sem_details)}")
                
                if isinstance(sem_log, dict):
                    topk = sem_log.get("topk")
                    parents_list = sem_log.get("parents") or []
                    results_count = sem_log.get("results_count", 0)
                    if parents_list:
                        logger.info(
                            f"Memory (worker): Semantic search for parent={parent.id}, topk={topk}, found {results_count} results, parent_ids={parents_list}"
                        )
                    elif results_count > 0:
                        # Results found but no parent IDs (unusual but log it)
                        logger.info(
                            f"Memory (worker): Semantic search for parent={parent.id}, topk={topk}, found {results_count} results, but no parent_ids available"
                        )
                    else:
                        logger.info(f"Memory (worker): Semantic search for parent={parent.id}, topk={topk}, no results found")
                elif sem_log is not None:
                    logger.warning(f"Memory (worker): Semantic search log exists but is not a dict (type: {type(sem_log)}) for parent={parent.id}")
                # If sem_log is None, memory is disabled - don't log anything
                if isinstance(sem_details, list) and sem_details:
                    preview = [
                        {
                            "parent": d.get("parent_id"),
                            "child": d.get("child_id"),
                            "parent_combined": d.get("parent_combined_score"),
                            "child_combined": d.get("child_combined_score"),
                            "delta": d.get("delta_combined_score"),
                        }
                        for d in sem_details[: min(5, len(sem_details))]
                    ]
                    logger.info(f"Memory (worker): Semantic search details for iteration {iteration}: {len(sem_details)} entries, preview (first 5): {preview}")
                elif isinstance(sem_details, list) and not sem_details:
                    logger.info(f"Memory (worker): Semantic search details exists but is empty for iteration {iteration}")
                elif sem_details is not None:
                    logger.warning(f"Memory (worker): Semantic search details exists but is not a list (type: {type(sem_details)}) for iteration {iteration}")
        except Exception as e:
            logger.error(f"Memory (worker): Semantic search logging failed for iteration {iteration}: {e}", exc_info=True)

        # Build prompt
        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in best_programs],
            top_programs=[p.to_dict() for p in (best_programs + inspirations)],
            inspirations=[p.to_dict() for p in inspirations],
            similar_parent_changes=strategy_snapshot.get("semantic_parent_details", []) if isinstance(strategy_snapshot, dict) else [],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=None,
            feature_dimensions=strategy_snapshot.get("feature_dimensions", []) if isinstance(strategy_snapshot, dict) else [],
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
        
        # Track iteration context to log failures to memory with parent and strategy info
        self.iteration_context: Dict[int, Tuple[Optional[str], Optional[int]]] = {}
        
        # Memory store and logging path (set by controller)
        self.memory_store = None
        self.memory_log_path: Optional[str] = None

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
                    # Mark branch as failed in strategy (for beam search tracking)
                    if hasattr(self.strategy, 'mark_branch_failed'):
                        self.strategy.mark_branch_failed()
                    # Add failure record to memory so failures are searchable (failure_signature, status=fail)
                    try:
                        memory_store = getattr(self, "memory_store", None)
                        if memory_store is not None:
                            from memory.schemas import MemoryEntry
                            import uuid as _uuid
                            parent_program = self.strategy.programs.get(result.parent_id) if result.parent_id else None

                            generator_input = {
                                "code": parent_program.code if parent_program else "",
                                "metrics": (parent_program.metrics if parent_program else {}),
                            }
                            generator_output = {
                                "code": "",
                                "llm_response": result.llm_response,
                                "changes_summary": "",
                            }
                            validator_output = {"error": result.error, "artifacts": (result.artifacts or {})}

                            entry = MemoryEntry(
                                parent_program_id=result.parent_id or "",
                                child_program_id=str(_uuid.uuid4()),  # no valid child program id
                                generator_input=generator_input,
                                generator_output=generator_output,
                                validator_output=validator_output,
                                diff_summary_user="",
                                generator_prompt=result.prompt,
                                iteration=completed_iteration,
                                metadata={"status": "fail"},
                            )
                            memory_store.add(entry)
                            logger.info(f"Memory: Added failure entry for iteration {completed_iteration} (parent={result.parent_id}, error={result.error[:50]})")

                            # Log to memory JSONL as well
                            try:
                                import os
                                memory_log_path = getattr(self, "memory_log_path", None)
                                if memory_log_path:
                                    os.makedirs(os.path.dirname(memory_log_path), exist_ok=True)
                                    log_record = {
                                        "iteration": completed_iteration,
                                        "parent_id": result.parent_id,
                                        "child_id": entry.child_program_id,
                                        "generator_input": generator_input,
                                        "generator_output": generator_output,
                                        "validator_output": validator_output,
                                        "generator_prompt": result.prompt,
                                    }
                                    import json
                                    with open(memory_log_path, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                                    pretty_path = memory_log_path.replace(".jsonl", "_pretty.json")
                                    with open(pretty_path, "w", encoding="utf-8") as f:
                                        json.dump(log_record, f, ensure_ascii=False, indent=2)
                            except Exception:
                                logger.error("Failed to write failure memory logs", exc_info=True)
                    except Exception:
                        logger.error("Memory add/logging failed for iteration error", exc_info=True)
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Add to strategy
                    self.strategy.add_program(child_program, iteration=completed_iteration)

                    # Add-only memory integration (non-blocking enrichment happens inside store)
                    try:
                        memory_store = getattr(self, "memory_store", None)
                        if memory_store is not None:
                            from memory.schemas import MemoryEntry
                            parent_program = self.strategy.programs.get(result.parent_id) if result.parent_id else None

                            generator_input = {
                                "code": parent_program.code if parent_program else "",
                                "metrics": (parent_program.metrics if parent_program else {}),
                            }
                            generator_output = {
                                "code": child_program.code,
                                "llm_response": result.llm_response,
                                "changes_summary": child_program.metadata.get("changes", ""),
                            }
                            validator_output = {**(child_program.metrics or {}), "artifacts": (result.artifacts or {})}

                            entry = MemoryEntry(
                                parent_program_id=result.parent_id or "",
                                child_program_id=child_program.id,
                                generator_input=generator_input,
                                generator_output=generator_output,
                                validator_output=validator_output,
                                diff_summary_user=child_program.metadata.get("changes", ""),
                                generator_prompt=result.prompt,
                                iteration=completed_iteration,
                                metadata={},
                            )
                            memory_store.add(entry)
                            logger.info(f"Memory: Added entry for iteration {completed_iteration} (parent={result.parent_id}, child={child_program.id})")

                            # Structured logging of what we added
                            try:
                                import os
                                import json
                                memory_log_path = getattr(self, "memory_log_path", None)
                                if memory_log_path:
                                    os.makedirs(os.path.dirname(memory_log_path), exist_ok=True)
                                    log_record = {
                                        "iteration": completed_iteration,
                                        "parent_id": result.parent_id,
                                        "child_id": child_program.id,
                                        "generator_input": generator_input,
                                        "generator_output": generator_output,
                                        "validator_output": validator_output,
                                        "generator_prompt": result.prompt,
                                    }
                                    with open(memory_log_path, "a", encoding="utf-8") as f:
                                        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                                    # Also write a pretty single-file snapshot of the last record for quick viewing
                                    pretty_path = memory_log_path.replace(".jsonl", "_pretty.json")
                                    with open(pretty_path, "w", encoding="utf-8") as f:
                                        json.dump(log_record, f, ensure_ascii=False, indent=2)
                            except Exception:
                                logger.error("Failed to write memory logs", exc_info=True)
                    except Exception:
                        logger.error("Memory add/logging failed", exc_info=True)

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
                # Mark branch as failed in strategy (for beam search tracking)
                if hasattr(self.strategy, 'mark_branch_failed'):
                    self.strategy.mark_branch_failed()
                # Add timeout record to memory
                try:
                    memory_store = getattr(self, "memory_store", None)
                    if memory_store is not None:
                        from memory.schemas import MemoryEntry
                        import uuid as _uuid
                        parent_id, _ = self.iteration_context.get(completed_iteration, (None, None))
                        parent_program = self.strategy.programs.get(parent_id) if parent_id else None

                        generator_input = {
                            "code": parent_program.code if parent_program else "",
                            "metrics": (parent_program.metrics if parent_program else {}),
                        }
                        generator_output = {"code": "", "llm_response": "", "changes_summary": ""}
                        validator_output = {"error": f"Iteration timeout after {timeout_seconds}s"}

                        entry = MemoryEntry(
                            parent_program_id=parent_id or "",
                            child_program_id=str(_uuid.uuid4()),
                            generator_input=generator_input,
                            generator_output={"code": "", "llm_response": None, "changes_summary": ""},
                            validator_output=validator_output,
                            diff_summary_user="",
                            generator_prompt=None,
                            iteration=completed_iteration,
                            metadata={"status": "fail"},
                        )
                        memory_store.add(entry)
                        logger.info(f"Memory: Added timeout entry for iteration {completed_iteration} (parent={parent_id})")

                        # Also log to JSONL
                        try:
                            import os
                            import json
                            memory_log_path = getattr(self, "memory_log_path", None)
                            if memory_log_path:
                                os.makedirs(os.path.dirname(memory_log_path), exist_ok=True)
                                log_record = {
                                    "iteration": completed_iteration,
                                    "parent_id": parent_id,
                                    "child_id": entry.child_program_id,
                                    "generator_input": generator_input,
                                    "generator_output": {"code": "", "llm_response": None, "changes_summary": ""},
                                    "validator_output": validator_output,
                                    "generator_prompt": None,
                                }
                                with open(memory_log_path, "a", encoding="utf-8") as f:
                                    f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                                pretty_path = memory_log_path.replace(".jsonl", "_pretty.json")
                                with open(pretty_path, "w", encoding="utf-8") as f:
                                    json.dump(log_record, f, ensure_ascii=False, indent=2)
                        except Exception:
                            logger.error("Failed to write timeout memory logs", exc_info=True)
                except Exception:
                    logger.error("Memory add/logging failed for timeout", exc_info=True)
                future.cancel()
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")
                # Mark branch as failed in strategy (for beam search tracking)
                if hasattr(self.strategy, 'mark_branch_failed'):
                    self.strategy.mark_branch_failed()
                # Log processing error to memory
                try:
                    memory_store = getattr(self, "memory_store", None)
                    if memory_store is not None:
                        from memory.schemas import MemoryEntry
                        import uuid as _uuid
                        parent_id, _ = self.iteration_context.get(completed_iteration, (None, None))
                        parent_program = self.strategy.programs.get(parent_id) if parent_id else None
                        generator_input = {
                            "code": parent_program.code if parent_program else "",
                            "metrics": (parent_program.metrics if parent_program else {}),
                        }
                        validator_output = {"error": str(e)}
                        entry = MemoryEntry(
                            parent_program_id=parent_id or "",
                            child_program_id=str(_uuid.uuid4()),
                            generator_input=generator_input,
                            generator_output={"code": "", "llm_response": None, "changes_summary": ""},
                            validator_output=validator_output,
                            diff_summary_user="",
                            generator_prompt=None,
                            iteration=completed_iteration,
                            metadata={"status": "fail"},
                        )
                        memory_store.add(entry)
                        logger.info(f"Memory: Added processing exception entry for iteration {completed_iteration} (parent={parent_id})")
                except Exception:
                    logger.error("Memory add/logging failed for processing exception", exc_info=True)

            # Cleanup iteration context
            try:
                if completed_iteration in self.iteration_context:
                    del self.iteration_context[completed_iteration]
            except Exception:
                pass

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

            # Prepare semantic search info in snapshot for worker-side logging
            # Get semantic search topk from config (defaults to 3 if not available)
            sem_topk = 3
            try:
                if hasattr(self.config, 'memory') and hasattr(self.config.memory, 'semantic_search_topk'):
                    sem_topk = int(self.config.memory.semantic_search_topk)
                else:
                    # Fallback to environment variable for backward compatibility
                    import os
                    sem_topk = int(os.environ.get("MEMORY_SEMANTIC_TOPK", "3"))
            except Exception:
                sem_topk = 3
            sem_parents = []
            sem_results_count = 0
            sem_details: List[Dict[str, Any]] = []
            try:
                memory_store = getattr(self, "memory_store", None)
                if memory_store is not None and parent is not None and isinstance(parent.code, str) and parent.code:
                    sem_results = memory_store.search_parents_by_code(parent.code, topk=sem_topk)
                    sem_parents = [r.get("parent") for r in sem_results]
                    sem_results_count = len(sem_results)
                    logger.info(f"Memory: Found {sem_results_count} similar parent(s) for iteration {iteration} (parent={parent.id}, topk={sem_topk})")
                    
                    # Build detailed records: ids, codes, combined scores, and deltas
                    for r in sem_results:
                        try:
                            pid = r.get("parent")
                            cid = r.get("child")
                            parent_prog = self.strategy.programs.get(pid) if pid else None
                            child_prog = self.strategy.programs.get(cid) if cid else None
                            p_code = (
                                parent_prog.code
                                if parent_prog is not None and isinstance(parent_prog.code, str)
                                else (r.get("generator_input", {}).get("code") if isinstance(r.get("generator_input"), dict) else None)
                            )
                            c_code = (
                                child_prog.code
                                if child_prog is not None and isinstance(child_prog.code, str)
                                else (r.get("generator_output", {}).get("code") if isinstance(r.get("generator_output"), dict) else None)
                            )
                            # Gather metrics dictionaries if available for richer prompt rendering
                            parent_metrics_dict = None
                            if parent_prog and isinstance(parent_prog.metrics, dict):
                                parent_metrics_dict = parent_prog.metrics
                            elif isinstance(r.get("generator_input"), dict) and isinstance(r.get("generator_input", {}).get("metrics"), dict):
                                parent_metrics_dict = r.get("generator_input", {}).get("metrics")

                            child_metrics_dict = None
                            if child_prog and isinstance(child_prog.metrics, dict):
                                child_metrics_dict = child_prog.metrics
                            elif isinstance(r.get("validator_output"), dict):
                                # validator_output may already be a metrics-like dict
                                child_metrics_dict = r.get("validator_output")
                            def _num(x):
                                return float(x) if isinstance(x, (int, float)) else None
                            p_comb = None
                            if parent_prog and isinstance(parent_prog.metrics, dict):
                                p_comb = parent_prog.metrics.get("combined_score")
                            if p_comb is None and isinstance(r.get("generator_input"), dict):
                                pm = r.get("generator_input", {}).get("metrics")
                                if isinstance(pm, dict):
                                    p_comb = pm.get("combined_score")
                            c_comb = None
                            if child_prog and isinstance(child_prog.metrics, dict):
                                c_comb = child_prog.metrics.get("combined_score")
                            if c_comb is None and isinstance(r.get("validator_output"), dict):
                                c_comb = r.get("validator_output", {}).get("combined_score")
                            # Fallbacks to numeric averages
                            if c_comb is None and isinstance(r.get("validator_output"), dict):
                                try:
                                    nums = [v for v in r["validator_output"].values() if isinstance(v, (int, float))]
                                    c_comb = (sum(nums) / len(nums)) if nums else None
                                except Exception:
                                    c_comb = None
                            if p_comb is None and parent_prog and isinstance(parent_prog.metrics, dict):
                                try:
                                    nums = [v for v in parent_prog.metrics.values() if isinstance(v, (int, float))]
                                    p_comb = (sum(nums) / len(nums)) if nums else None
                                except Exception:
                                    p_comb = None
                            delta = None
                            if isinstance(p_comb, (int, float)) and isinstance(c_comb, (int, float)):
                                try:
                                    delta = float(c_comb) - float(p_comb)
                                except Exception:
                                    delta = None
                            # Try to surface a concise change summary for the child
                            change_summary = None
                            try:
                                if child_prog and isinstance(child_prog.metadata, dict):
                                    change_summary = child_prog.metadata.get("changes")
                                if not change_summary and isinstance(r.get("generator_output"), dict):
                                    change_summary = r.get("generator_output", {}).get("changes_summary")
                                if not change_summary:
                                    change_summary = r.get("diff_summary_user")
                            except Exception:
                                change_summary = None
                            sem_details.append(
                                {
                                    "parent_id": pid,
                                    "child_id": cid,
                                    "parent_code": p_code,
                                    "child_code": c_code,
                                    "parent_metrics": parent_metrics_dict,
                                    "child_metrics": child_metrics_dict,
                                    "change_summary": change_summary,
                                    "parent_combined_score": _num(p_comb),
                                    "child_combined_score": _num(c_comb),
                                    "delta_combined_score": _num(delta),
                                }
                            )
                        except Exception:
                            logger.error("Failed to build semantic parent detail record", exc_info=True)
                            continue
            except Exception:
                logger.error("Semantic search call failed", exc_info=True)

            # Save iteration context (parent and strategy info) for robust failure logging
            try:
                self.iteration_context[iteration] = (parent.id if parent else None, None)
            except Exception:
                pass

            # Create strategy snapshot
            strategy_snapshot = self.strategy.get_snapshot()

            # Only add semantic search data if memory is enabled
            memory_store = getattr(self, "memory_store", None)
            if memory_store is not None:
                strategy_snapshot["semantic_parent_log"] = {
                    "topk": sem_topk,
                    "parents": sem_parents,
                    "results_count": sem_results_count,
                }
                strategy_snapshot["semantic_parent_details"] = sem_details

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
