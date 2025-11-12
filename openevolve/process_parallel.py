"""
Process-based parallel controller for true parallelism
"""

import asyncio
import json
import logging
import multiprocessing as mp
import os
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
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
    # Reconstruct Config object from nested dictionaries
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
            database=None,  # No shared database in worker
            suffix=getattr(_worker_config, 'file_suffix', '.py'),
        )


def _run_iteration_worker(
    iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    try:
        # Lazy initialization
        _lazy_init_worker_components()

        # Reconstruct programs from snapshot
        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}

        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]

        # Get parent artifacts if available
        parent_artifacts = db_snapshot["artifacts"].get(parent_id)

        # Get island-specific programs for context
        parent_island = parent.metadata.get("island", db_snapshot["current_island"])
        island_programs = [
            programs[pid] for pid in db_snapshot["islands"][parent_island] if pid in programs
        ]

        # Sort by metrics for top programs
        island_programs.sort(
            key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
            reverse=True,
        )

        # Use config values for limits instead of hardcoding
        # Programs for LLM display (includes both top and diverse for inspiration)
        programs_for_prompt = island_programs[
            : _worker_config.prompt.num_top_programs + _worker_config.prompt.num_diverse_programs
        ]
        # Best programs only (for previous attempts section, focused on top performers)
        best_programs_only = island_programs[: _worker_config.prompt.num_top_programs]

        # Log semantic search (supplied by controller) just before building prompt
        # Only log if memory is enabled (indicated by presence of semantic_parent_log in snapshot)
        sem_log = db_snapshot.get("semantic_parent_log") if isinstance(db_snapshot, dict) else None
        if sem_log is not None:
            logger.info(f"Memory (worker): Starting semantic search logging for iteration {iteration}, parent={parent.id}")
        try:
            if not isinstance(db_snapshot, dict):
                logger.warning(f"Memory (worker): db_snapshot is not a dict (type: {type(db_snapshot)})")
            elif sem_log is not None:
                # Debug: log what keys are in db_snapshot (only when memory is enabled)
                snapshot_keys = list(db_snapshot.keys())
                logger.info(f"Memory (worker): db_snapshot keys for iteration {iteration}: {snapshot_keys}")

                sem_details = db_snapshot.get("semantic_parent_details")
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
            previous_programs=[p.to_dict() for p in best_programs_only],
            top_programs=[p.to_dict() for p in programs_for_prompt],
            inspirations=[p.to_dict() for p in inspirations],
            similar_parent_changes=db_snapshot.get("semantic_parent_details", []),
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            feature_dimensions=db_snapshot.get("feature_dimensions", []),
        )

        iteration_start = time.time()

        # Generate code modification (sync wrapper for async)
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

        # Check for None response
        if llm_response is None:
            return SerializableResult(error="LLM returned None response", iteration=iteration)

        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import extract_diffs, apply_diff, format_diff_summary

            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(
                    error=f"No valid diffs found in response", iteration=iteration
                )

            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(
                    error=f"No valid code found in response", iteration=iteration
                )

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
                "island": parent_island,
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


class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase, evolution_tracer=None, file_suffix: str = ".py"):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.evolution_tracer = evolution_tracer
        self.file_suffix = file_suffix

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations

        # Worker-to-island pinning for true island isolation
        self.num_islands = config.database.num_islands
        self.worker_island_map = {}

        # Distribute workers across islands using modulo
        for worker_id in range(self.num_workers):
            island_id = worker_id % self.num_islands
            self.worker_island_map[worker_id] = island_id

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")
        logger.info(f"Worker-to-island mapping: {self.worker_island_map}")

        # Track iteration context to log failures to memory with parent and island
        self.iteration_context: Dict[int, Tuple[Optional[str], Optional[int]]] = {}

        # Memory store and logging path (set by controller)
        self.memory_store = None
        self.memory_log_path: Optional[str] = None

    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly
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
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
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

    def _create_database_snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the database state"""
        # Only include necessary data for workers
        snapshot = {
            "programs": {pid: prog.to_dict() for pid, prog in self.database.programs.items()},
            "islands": [list(island) for island in self.database.islands],
            "current_island": self.database.current_island,
            "feature_dimensions": self.database.config.feature_dimensions,
            "artifacts": {},  # Will be populated selectively
        }

        # Include artifacts for programs that might be selected
        # IMPORTANT: This limits artifacts (execution outputs/errors) to first 100 programs only.
        # This does NOT affect program code - all programs are fully serialized above.
        # With max_artifact_bytes=20KB and population_size=1000, artifacts could be 20MB total,
        # which would significantly slow worker process initialization. The limit of 100 keeps
        # artifact data under 2MB while still providing execution context for recent programs.
        # Workers can still evolve properly as they have access to ALL program code.
        for pid in list(self.database.programs.keys())[:100]:
            artifacts = self.database.get_artifacts(pid)
            if artifacts:
                snapshot["artifacts"][pid] = artifacts

        return snapshot

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

        # Track pending futures by island to maintain distribution
        pending_futures: Dict[int, Future] = {}
        island_pending: Dict[int, List[int]] = {i: [] for i in range(self.num_islands)}
        batch_size = min(self.num_workers * 2, max_iterations)

        # Submit initial batch - distribute across islands
        batch_per_island = max(1, batch_size // self.num_islands) if batch_size > 0 else 0
        current_iteration = start_iteration

        # Round-robin distribution across islands
        for island_id in range(self.num_islands):
            for _ in range(batch_per_island):
                if current_iteration < total_iterations:
                    future = self._submit_iteration(current_iteration, island_id)
                    if future:
                        pending_futures[current_iteration] = future
                        island_pending[island_id].append(current_iteration)
                    current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0

        # Island management
        programs_per_island = max(1, max_iterations // (self.config.database.num_islands * 10))
        current_island_counter = 0

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
        else:
            logger.info("Early stopping disabled")

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
                # Use evaluator timeout + buffer to gracefully handle stuck processes
                timeout_seconds = self.config.evaluator.timeout + 30
                result = future.result(timeout=timeout_seconds)

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                    # Add failure record to memory so failures are searchable (failure_signature, status=fail)
                    try:
                        memory_store = getattr(self, "memory_store", None)
                        if memory_store is not None:
                            from memory.schemas import MemoryEntry
                            import uuid as _uuid
                            # Use iteration_context for robust parent/island lookup
                            parent_id, island_id = self.iteration_context.get(
                                result.iteration, (result.parent_id, self.database.current_island)
                            )
                            parent_program = self.database.get(parent_id) if parent_id else None

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
                                parent_program_id=parent_id or "",
                                child_program_id=str(_uuid.uuid4()),  # no valid child program id
                                generator_input=generator_input,
                                generator_output=generator_output,
                                validator_output=validator_output,
                                diff_summary_user="",
                                generator_prompt=result.prompt,
                                iteration=result.iteration,
                                metadata={"island": island_id, "status": "fail"},
                            )
                            memory_store.add(entry)
                            logger.info(f"Memory: Added failure entry for iteration {result.iteration} (parent={parent_id}, error={result.error[:50]})")

                            # Log to memory JSONL as well
                            try:
                                memory_log_path = getattr(self, "memory_log_path", None)
                                if memory_log_path:
                                    os.makedirs(os.path.dirname(memory_log_path), exist_ok=True)
                                    log_record = {
                                        "iteration": result.iteration,
                                        "parent_id": parent_id,
                                        "child_id": entry.child_program_id,
                                        "island": island_id,
                                        "generator_input": generator_input,
                                        "generator_output": generator_output,
                                        "validator_output": validator_output,
                                        "generator_prompt": result.prompt,
                                    }
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

                    # Add to database (will auto-inherit parent's island)
                    # No need to specify target_island - database will handle parent island inheritance
                    self.database.add(child_program, iteration=completed_iteration)

                    # Store artifacts
                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)

                    # Add-only memory integration (non-blocking enrichment happens inside store)
                    try:
                        memory_store = getattr(self, "memory_store", None)
                        if memory_store is not None:
                            from memory.schemas import MemoryEntry
                            parent_program = self.database.get(result.parent_id) if result.parent_id else None
                            island_id = child_program.metadata.get("island", self.database.current_island)

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

                            # Compute distance and gradient for gradient-based evolution
                            distance = None
                            gradient = None
                            if parent_program and hasattr(memory_store, 'compute_code_distance'):
                                try:
                                    # Compute semantic distance between parent and child
                                    distance = memory_store.compute_code_distance(
                                        parent_program.code, child_program.code
                                    )

                                    if distance is not None:
                                        # Compute delta score (child - parent)
                                        parent_score = parent_program.metrics.get("combined_score") if parent_program.metrics else None
                                        child_score = child_program.metrics.get("combined_score") if child_program.metrics else None

                                        if parent_score is not None and child_score is not None:
                                            delta = float(child_score) - float(parent_score)
                                            # Gradient = delta / distance (avoid division by zero)
                                            gradient = delta / max(distance, 0.01)

                                            # Update parent's gradient statistics
                                            parent_program.visit_count += 1
                                            parent_program.total_gradient += gradient
                                except Exception:
                                    # Non-blocking: gradient computation is optional
                                    pass

                            entry = MemoryEntry(
                                parent_program_id=result.parent_id or "",
                                child_program_id=child_program.id,
                                generator_input=generator_input,
                                generator_output=generator_output,
                                validator_output=validator_output,
                                diff_summary_user=child_program.metadata.get("changes", ""),
                                generator_prompt=result.prompt,
                                iteration=completed_iteration,
                                metadata={"island": island_id},
                                distance=distance,
                                gradient=gradient,
                            )
                            memory_store.add(entry)
                            logger.info(f"Memory: Added entry for iteration {completed_iteration} (parent={result.parent_id}, child={child_program.id})")

                            # Structured logging of what we added
                            try:
                                memory_log_path = getattr(self, "memory_log_path", None)
                                if memory_log_path:
                                    os.makedirs(os.path.dirname(memory_log_path), exist_ok=True)
                                    log_record = {
                                        "iteration": completed_iteration,
                                        "parent_id": result.parent_id,
                                        "child_id": child_program.id,
                                        "island": island_id,
                                        "generator_input": generator_input,
                                        "generator_output": generator_output,
                                        "validator_output": validator_output,
                                        "generator_prompt": result.prompt,
                                        "distance": distance,
                                        "gradient": gradient,
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
                        # Never break evolution due to memory logging, but surface as error
                        logger.error("Memory add/logging failed", exc_info=True)

                    # Log evolution trace
                    if self.evolution_tracer:
                        # Retrieve parent program for trace logging
                        parent_program = self.database.get(result.parent_id) if result.parent_id else None
                        if parent_program:
                            # Determine island ID
                            island_id = child_program.metadata.get("island", self.database.current_island)
                            
                            self.evolution_tracer.log_trace(
                                iteration=completed_iteration,
                                parent_program=parent_program,
                                child_program=child_program,
                                prompt=result.prompt,
                                llm_response=result.llm_response,
                                artifacts=result.artifacts,
                                island_id=island_id,
                                metadata={
                                    "iteration_time": result.iteration_time,
                                    "changes": child_program.metadata.get("changes", ""),
                                }
                            )

                    # Log prompts
                    if result.prompt:
                        self.database.log_prompt(
                            template_key=(
                                "full_rewrite_user"
                                if not self.config.diff_based_evolution
                                else "diff_user"
                            ),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )

                    # Island management
                    if (
                        completed_iteration > start_iteration
                        and current_island_counter >= programs_per_island
                    ):
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")

                    current_island_counter += 1
                    self.database.increment_island_generation()

                    # Check migration
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
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

                        # Check if this is the first program without combined_score
                        if not hasattr(self, "_warned_about_combined_score"):
                            self._warned_about_combined_score = False

                        if (
                            "combined_score" not in child_program.metrics
                            and not self._warned_about_combined_score
                        ):
                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True

                    # Check for new best
                    if self.database.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id}"
                        )

                    # Checkpoint callback
                    # Don't checkpoint at iteration 0 (that's just the initial program)
                    if (
                        completed_iteration > 0
                        and completed_iteration % self.config.checkpoint_interval == 0
                    ):
                        logger.info(
                            f"Checkpoint interval reached at iteration {completed_iteration}"
                        )
                        self.database.log_island_status()
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
                        # Get the metric to track for early stopping
                        current_score = None
                        if self.config.early_stopping_metric in child_program.metrics:
                            current_score = child_program.metrics[self.config.early_stopping_metric]
                        elif self.config.early_stopping_metric == "combined_score":
                            # Default metric not found, use safe average (standard pattern)
                            current_score = safe_numeric_average(child_program.metrics)
                        else:
                            # User specified a custom metric that doesn't exist
                            logger.warning(
                                f"Early stopping metric '{self.config.early_stopping_metric}' not found, using safe numeric average"
                            )
                            current_score = safe_numeric_average(child_program.metrics)

                        if current_score is not None and isinstance(current_score, (int, float)):
                            # Check for improvement
                            improvement = current_score - best_score
                            if improvement >= self.config.convergence_threshold:
                                best_score = current_score
                                iterations_without_improvement = 0
                                logger.debug(
                                    f"New best score: {best_score:.4f} (improvement: {improvement:+.4f})"
                                )
                            else:
                                iterations_without_improvement += 1
                                logger.debug(
                                    f"No improvement: {iterations_without_improvement}/{self.config.early_stopping_patience}"
                                )

                            # Check if we should stop
                            if (
                                iterations_without_improvement
                                >= self.config.early_stopping_patience
                            ):
                                self.early_stopping_triggered = True
                                logger.info(
                                    f"🛑 Early stopping triggered at iteration {completed_iteration}: "
                                    f"No improvement for {iterations_without_improvement} iterations "
                                    f"(best score: {best_score:.4f})"
                                )
                                break

            except FutureTimeoutError:
                logger.error(
                    f"⏰ Iteration {completed_iteration} timed out after {timeout_seconds}s "
                    f"(evaluator timeout: {self.config.evaluator.timeout}s + 30s buffer). "
                    f"Canceling future and continuing with next iteration."
                )
                # Cancel the future to clean up the process
                future.cancel()
                # Log timeout to memory
                try:
                    memory_store = getattr(self, "memory_store", None)
                    if memory_store is not None:
                        from memory.schemas import MemoryEntry
                        import uuid as _uuid
                        parent_id, island_id = self.iteration_context.get(
                            completed_iteration, (None, self.database.current_island)
                        )
                        parent_program = self.database.get(parent_id) if parent_id else None
                        generator_input = {
                            "code": parent_program.code if parent_program else "",
                            "metrics": (parent_program.metrics if parent_program else {}),
                        }
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
                            metadata={"island": island_id, "status": "fail"},
                        )
                        memory_store.add(entry)
                        logger.info(f"Memory: Added timeout entry for iteration {completed_iteration} (parent={parent_id})")

                        # Also log to JSONL
                        try:
                            memory_log_path = getattr(self, "memory_log_path", None)
                            if memory_log_path:
                                os.makedirs(os.path.dirname(memory_log_path), exist_ok=True)
                                log_record = {
                                    "iteration": completed_iteration,
                                    "parent_id": parent_id,
                                    "child_id": entry.child_program_id,
                                    "island": island_id,
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
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")
                # Log processing error to memory
                try:
                    memory_store = getattr(self, "memory_store", None)
                    if memory_store is not None:
                        from memory.schemas import MemoryEntry
                        import uuid as _uuid
                        parent_id, island_id = self.iteration_context.get(
                            completed_iteration, (None, self.database.current_island)
                        )
                        parent_program = self.database.get(parent_id) if parent_id else None
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
                            metadata={"island": island_id, "status": "fail"},
                        )
                        memory_store.add(entry)
                        logger.info(f"Memory: Added processing exception entry for iteration {completed_iteration} (parent={parent_id})")
                except Exception:
                    logger.error("Memory add/logging failed for processing exception", exc_info=True)

            completed_iterations += 1

            # Clean up iteration context to prevent memory leaks
            if completed_iteration in self.iteration_context:
                del self.iteration_context[completed_iteration]

            # Remove completed iteration from island tracking
            for island_id, iteration_list in island_pending.items():
                if completed_iteration in iteration_list:
                    iteration_list.remove(completed_iteration)
                    break

            # Submit next iterations maintaining island balance
            for island_id in range(self.num_islands):
                if (
                    len(island_pending[island_id]) < batch_per_island
                    and next_iteration < total_iterations
                    and not self.shutdown_event.is_set()
                ):
                    future = self._submit_iteration(next_iteration, island_id)
                    if future:
                        pending_futures[next_iteration] = future
                        island_pending[island_id].append(next_iteration)
                        next_iteration += 1
                        break  # Only submit one iteration per completion to maintain balance

        # Handle shutdown
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        # Log completion reason
        if self.early_stopping_triggered:
            logger.info("✅ Evolution completed - Early stopping triggered due to convergence")
        elif self.shutdown_event.is_set():
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
            logger.info("✅ Evolution completed - Maximum iterations reached")

        return self.database.get_best_program()

    def _submit_iteration(
        self, iteration: int, island_id: Optional[int] = None
    ) -> Optional[Future]:
        """Submit an iteration to the process pool, optionally pinned to a specific island"""
        try:
            # Use specified island or current island
            target_island = island_id if island_id is not None else self.database.current_island

            # Use thread-safe sampling that doesn't modify shared state
            # This fixes the race condition from GitHub issue #246
            parent, inspirations = self.database.sample_from_island(
                island_id=target_island,
                num_inspirations=self.config.prompt.num_top_programs
            )

            # Prepare semantic search info in snapshot for worker-side logging
            # Get semantic search topk from config (defaults to 3 if not available)
            sem_topk = 3
            try:
                if hasattr(self.config, 'memory') and hasattr(self.config.memory, 'semantic_search_topk'):
                    sem_topk = int(self.config.memory.semantic_search_topk)
                else:
                    # Fallback to environment variable for backward compatibility
                    sem_topk = int(os.environ.get("MEMORY_SEMANTIC_TOPK", "3"))
            except Exception:
                sem_topk = 3
            sem_parents = []
            sem_results_count = 0
            sem_details: List[Dict[str, Any]] = []
            try:
                memory_store = getattr(self, "memory_store", None)
                if memory_store is not None and parent is not None and isinstance(parent.code, str) and parent.code:
                    # Get more candidates for gradient-based filtering
                    initial_topk = max(20, sem_topk * 3)
                    sem_results_raw = memory_store.search_parents_by_code(parent.code, topk=initial_topk)

                    # Apply gradient-based scoring: info_score = similarity × |gradient|
                    for r in sem_results_raw:
                        similarity = r.get("similarity", 0.0)
                        gradient = r.get("gradient")

                        # If gradient is not available, compute it from delta/distance if possible
                        if gradient is None:
                            distance = r.get("distance")
                            # Try to compute delta from validator output
                            if distance is not None and distance > 0:
                                try:
                                    validator = r.get("validator_output", {})
                                    generator_input = r.get("generator_input", {})
                                    if isinstance(validator, dict) and isinstance(generator_input, dict):
                                        child_score = validator.get("combined_score")
                                        parent_score = generator_input.get("metrics", {}).get("combined_score")
                                        if child_score is not None and parent_score is not None:
                                            delta = float(child_score) - float(parent_score)
                                            gradient = delta / max(distance, 0.01)
                                except Exception:
                                    pass

                        # Compute information score (default to similarity if no gradient)
                        if gradient is not None:
                            r["info_score"] = similarity * abs(gradient)
                        else:
                            # Fallback: use similarity alone if gradient not available
                            r["info_score"] = similarity

                    # Sort by info_score (highest first) and take top-k
                    sem_results_raw.sort(key=lambda x: x.get("info_score", 0), reverse=True)
                    sem_results = sem_results_raw[:sem_topk]

                    sem_parents = [r.get("parent") for r in sem_results]
                    sem_results_count = len(sem_results)
                    logger.info(f"Memory: Found {sem_results_count} similar parent(s) for iteration {iteration} (parent={parent.id}, topk={sem_topk})")

                    # Build detailed records: ids, codes, combined scores, and deltas
                    for r in sem_results:
                        try:
                            pid = r.get("parent")
                            cid = r.get("child")
                            parent_prog = self.database.get(pid) if pid else None
                            child_prog = self.database.get(cid) if cid else None
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
                                    "gradient": r.get("gradient"),
                                    "distance": r.get("distance"),
                                }
                            )
                        except Exception:
                            logger.error("Failed to build semantic parent detail record", exc_info=True)
                else:
                    logger.debug(f"Memory: No similar parent search (memory_store={memory_store is not None}, parent={parent.id if parent else None})")
            except Exception:
                logger.error("Semantic search call failed", exc_info=True)

            # Save iteration context (parent and island) for robust failure logging
            try:
                self.iteration_context[iteration] = (parent.id if parent else None, target_island)
            except Exception:
                pass

            # Create database snapshot
            db_snapshot = self._create_database_snapshot()
            db_snapshot["sampling_island"] = target_island  # Mark which island this is for

            # Only add semantic search data if memory is enabled
            memory_store = getattr(self, "memory_store", None)
            if memory_store is not None:
                db_snapshot["semantic_parent_log"] = {
                    "topk": sem_topk,
                    "parents": sem_parents,
                    "results_count": sem_results_count,
                }
                db_snapshot["semantic_parent_details"] = sem_details

            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations],
            )

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
