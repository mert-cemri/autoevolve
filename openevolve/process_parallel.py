"""
Process-based parallel controller for true parallelism
"""

import asyncio
import logging
import multiprocessing as mp
import pickle
import signal
import time
import sys
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Use the exact helpers from openevolve.coach_helpers for prompt parity (hard dependency)
from openevolve.coach_helpers import (
    read_last_n_jsonl as probe_read_last_n_jsonl,
    build_minimal_context as probe_build_minimal_context,
    read_best_metrics as probe_read_best_metrics,
    read_task_context as probe_read_task_context,
    make_prompt as probe_make_prompt,
)
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
        # Determine evolution mode for this iteration (controller may override per snapshot)
        try:
            diff_mode = bool(db_snapshot.get("diff_based_evolution", getattr(_worker_config, "diff_based_evolution", True)))
        except Exception:
            diff_mode = getattr(_worker_config, "diff_based_evolution", True)

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

        # Build prompt
        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in best_programs_only],
            top_programs=[p.to_dict() for p in programs_for_prompt],
            inspirations=[p.to_dict() for p in inspirations],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=diff_mode,
            program_artifacts=parent_artifacts,
            feature_dimensions=db_snapshot.get("feature_dimensions", []),
        )
        # Inject a single neutral coach hint line at top of user prompt if provided
        try:
            coach_hint = db_snapshot.get("coach_hint_text", "")
            if isinstance(coach_hint, str) and coach_hint.strip():
                user_msg = prompt.get("user", "")
                prompt["user"] = f"{coach_hint.strip()}\n\n{user_msg}"
                prompt_injected = True
                injected_hint_text = coach_hint.strip()
            else:
                prompt_injected = False
                injected_hint_text = ""
        except Exception:
            prompt_injected = False
            injected_hint_text = ""

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
        # If we injected a coach hint, log the exact prompt and the model's raw response
        try:
            log_dir = getattr(_worker_config, "log_dir", None)
            if log_dir and prompt_injected:
                os.makedirs(log_dir, exist_ok=True)
                inj_path = os.path.join(log_dir, "coach_injection_calls.jsonl")
                inj_rec = {
                    "ts": time.time(),
                    "iteration": iteration,
                    "hint": injected_hint_text,
                    "prompt_system": prompt.get("system", ""),
                    "prompt_user": prompt.get("user", ""),
                    "llm_response": llm_response,
                }
                with open(inj_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(inj_rec, ensure_ascii=False) + "\n")
        except Exception as _e:
            logger.debug(f"Coach injection call log failed: {_e}")

        # Extract optional PLAN line (single-line JSON) if present
        def _extract_plan(text: str):
            try:
                for line in text.splitlines():
                    s = line.strip()
                    if s.startswith("PLAN:"):
                        payload = s[len("PLAN:"):].strip()
                        if payload:
                            try:
                                return json.loads(payload)
                            except Exception:
                                return {"raw": payload}
                return None
            except Exception:
                return None

        llm_plan = _extract_plan(llm_response)

        # Parse response based on evolution mode
        if diff_mode:
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
                "llm_plan": llm_plan if llm_plan is not None else {},
            },
        )

        iteration_time = time.time() - iteration_start

        # Compute and store primary metric improvement
        def _select_primary_metric(metrics: Dict[str, Any]) -> str:
            if isinstance(metrics, dict) and "combined_score" in metrics:
                return "combined_score"
            if isinstance(metrics, dict) and "performance_score" in metrics:
                return "performance_score"
            return ""

        primary_metric = _select_primary_metric(child_metrics if isinstance(child_metrics, dict) else {})
        if primary_metric:
            try:
                p_val = float(parent.metrics.get(primary_metric, 0.0))
                c_val = float(child_metrics.get(primary_metric, 0.0)) if isinstance(child_metrics, dict) else 0.0
                delta_val = c_val - p_val
                child_program.metadata["primary_metric"] = primary_metric
                child_program.metadata["primary_metric_delta"] = delta_val
            except Exception:
                pass

        # Append structured intent/result log (JSONL) per iteration
        try:
            log_dir = getattr(_worker_config, "log_dir", None)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                intent_log_path = os.path.join(log_dir, "intent_log.jsonl")
                entry = {
                    "iteration": iteration,
                    "parent_id": parent.id,
                    "child_id": child_id,
                    "intent": llm_plan if llm_plan is not None else {},
                    "validator_output": child_metrics if isinstance(child_metrics, dict) else {},
                    "primary_metric": child_program.metadata.get("primary_metric", ""),
                    "primary_metric_delta": child_program.metadata.get("primary_metric_delta", 0.0),
                    "timestamp": time.time(),
                }
                with open(intent_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Intent log append failed: {e}")

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
        self.num_islands = config.database.num_islands

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")
        # Coach/stagnancy runtime state (diagnosis-only)
        self._coach_cooldown: int = 0
        self._coach_hint_remaining: int = 0
        self._coach_hint_text: str = ""
        self._best_correctness_seen: Optional[float] = None
        self._last_coach_event_ts: float = 0.0
        # Best-program plateau tracking
        self._last_best_id: Optional[str] = None
        self._stable_best_count: int = 0
        # Coach/stagnancy runtime state (diagnosis-only)
        self._recent_deltas: List[float] = []
        self._recent_intents: List[str] = []
        self._coach_cooldown: int = 0
        self._coach_hint_remaining: int = 0
        self._coach_hint_text: str = ""
        self._best_correctness_seen: Optional[float] = None
        # Burst exploration state (macro-rewrite window)
        self._burst_active: bool = False
        self._burst_iters_left: int = 0
        self._pre_burst: Dict[str, Any] = {}

    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly

        # The asdict() call itself triggers the deepcopy which tries to serialize novelty_llm. Remove it first.
        config.database.novelty_llm = None
        
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
        # Per-iteration override: allow controller to force full rewrites during bursts
        try:
            snapshot["diff_based_evolution"] = bool((not self._burst_active) and getattr(self.config, "diff_based_evolution", True))
        except Exception:
            snapshot["diff_based_evolution"] = getattr(self.config, "diff_based_evolution", True)
        # If coach hint is active, include one-line hint for worker prompt injection
        if self._coach_hint_remaining > 0 and self._coach_hint_text:
            snapshot["coach_hint_text"] = self._coach_hint_text

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
        programs_per_island = self.config.database.programs_per_island or max(1, max_iterations // (self.config.database.num_islands * 10))
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
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Add to database (will auto-inherit parent's island)
                    # No need to specify target_island - database will handle parent island inheritance
                    self.database.add(child_program, iteration=completed_iteration)

                    # Store artifacts
                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)
                    
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
                    # Update coach/stagnancy state after metrics available
                    try:
                        self._update_coach_state(child_program)
                    except Exception as _e:
                        logger.debug(f"Coach state update failed: {_e}")

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
                        if ('combined_score' in child_program.metrics and
                            child_program.metrics['combined_score'] >= target_score):
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
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1

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

            # Create database snapshot
            db_snapshot = self._create_database_snapshot()
            db_snapshot["sampling_island"] = target_island  # Mark which island this is for

            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations],
            )

            # If we have a coach hint to inject this iteration, record an event
            try:
                hint = db_snapshot.get("coach_hint_text", "")
                if isinstance(hint, str) and hint.strip():
                    self._append_coach_event({
                        "event": "inject_hint",
                        "iteration": iteration,
                        "hint": hint,
                    })
            except Exception:
                pass

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None

    # --------------- Coach/stagnancy helpers (diagnosis-only, minimal) ---------------
    def _update_coach_state(self, child_program: Program) -> None:
        """Maintain recent window, detect stagnancy, manage hint lifecycle."""
        if not getattr(self.config, "coach", None) or not self.config.coach.enabled:
            return
        # Track best-program stability (plateau heuristic)
        try:
            current_best_id = getattr(self.database, "best_program_id", None)
            if current_best_id:
                if self._last_best_id == current_best_id:
                    self._stable_best_count += 1
                else:
                    self._last_best_id = current_best_id
                    self._stable_best_count = 1
        except Exception:
            pass
        # Track best correctness seen (for guard)
        correctness = None
        if isinstance(child_program.metrics, dict):
            c = child_program.metrics.get("correctness_score")
            if isinstance(c, (int, float)):
                correctness = float(c)
        if correctness is not None:
            if self._best_correctness_seen is None:
                self._best_correctness_seen = correctness
            else:
                self._best_correctness_seen = max(self._best_correctness_seen, correctness)
        # Stop hint immediately if correctness regresses from perfect
        if self._coach_hint_remaining > 0 and self._best_correctness_seen is not None and self._best_correctness_seen >= 1.0:
            if correctness is not None and correctness < 1.0:
                # Log stop due to correctness regression
                try:
                    self._append_coach_event({
                        "event": "stop_hint_correctness_drop",
                        "iteration": getattr(child_program, "iteration_found", None),
                        "best_correctness_seen": self._best_correctness_seen,
                        "current_correctness": correctness,
                    })
                except Exception:
                    pass
                self._coach_hint_remaining = 0
                return
        # Extract delta for stop-on-improvement guard (any positive delta)
        delta = 0.0
        try:
            meta = child_program.metadata or {}
            if "primary_metric_delta" in meta and isinstance(meta["primary_metric_delta"], (int, float)):
                delta = float(meta["primary_metric_delta"])
            else:
                # fallback on combined_score delta if present
                pm = child_program.metrics or {}
                parent_metrics = meta.get("parent_metrics", {})
                if isinstance(pm, dict) and isinstance(parent_metrics, dict) and "combined_score" in pm and "combined_score" in parent_metrics:
                    delta = float(pm["combined_score"]) - float(parent_metrics["combined_score"])
        except Exception:
            delta = 0.0
        # Decrement hint if active
        if self._coach_hint_remaining > 0:
            self._coach_hint_remaining -= 1
            # Stop early if we saw a meaningful improvement
            # During bursts, ignore tiny gains; require a clear breakthrough
            improvement_threshold = 0.01
            if delta is not None and delta >= improvement_threshold:
                try:
                    self._append_coach_event({
                        "event": "stop_hint_improvement",
                        "iteration": getattr(child_program, "iteration_found", None),
                        "delta": delta,
                    })
                except Exception:
                    pass
                self._coach_hint_remaining = 0
                # End burst immediately on real breakthrough
                if self._burst_active:
                    self._end_burst(reason="breakthrough")
            # Tick down burst window if active
            if self._burst_active:
                if self._burst_iters_left > 0:
                    self._burst_iters_left -= 1
                if self._burst_iters_left <= 0 and self._burst_active:
                    self._end_burst(reason="window_complete")
        # Cooldown tick
        if self._coach_cooldown > 0:
            self._coach_cooldown -= 1
        # If hint already active, skip trigger (but allow ping regardless of cooldown)
        if self._coach_hint_remaining > 0:
            return
        # Evaluate best-plateau stagnancy
        win = self.config.coach.window
        best_plateau = bool(self._stable_best_count >= win)
        # Simplified stagnancy trigger: best-id plateau only
        trigger = best_plateau
        # Track whether a ping escalation will force a probe (default False)
        escalate_from_ping = False
        # Log the evaluation snapshot regardless of trigger outcome
        try:
            self._append_coach_event({
                "event": "stagnancy_eval",
                "status": "evaluated",
                "window_size": win,
                "best_stable_iters": self._stable_best_count,
                "best_plateau": best_plateau,
                "trigger": bool(trigger),
                "hint_active": self._coach_hint_remaining > 0,
                "cooldown": self._coach_cooldown,
            })
        except Exception:
            pass
        if not trigger:
            # Consider periodic stagnancy ping if configured
            try:
                iter_num = getattr(child_program, "iteration_found", None)
            except Exception:
                iter_num = None
            # Always update running score plot on each iteration
            try:
                self._update_score_plot(iter_num, child_program)
            except Exception:
                pass
            ping_after = getattr(self.config.coach, "ping_after", 0) or 0
            ping_every = getattr(self.config.coach, "ping_every", 0) or 0
            escalate_from_ping = False
            if (
                iter_num is not None
                and ping_after > 0
                and ping_every > 0
                and iter_num >= ping_after
                and ((iter_num - ping_after) % ping_every == 0)
            ):
                try:
                    self._append_coach_event({"event": "ping_call_start", "iteration": iter_num})
                except Exception:
                    pass
                try:
                    stagnant = self._run_stagnancy_ping_once()
                except Exception as _pe:
                    stagnant = False
                    try:
                        self._append_coach_event({"event": "ping_call_end", "status": f"failed: {_pe}"})
                    except Exception:
                        pass
                else:
                    try:
                        self._append_coach_event({"event": "ping_call_end", "iteration": iter_num, "stagnant": bool(stagnant)})
                    except Exception:
                        pass
                    if stagnant:
                        escalate_from_ping = True
            if not escalate_from_ping:
                return
        # If we are cooling down, skip probing for plateau-triggered checks only (ping escalation bypasses cooldown)
        if self._coach_cooldown > 0 and not escalate_from_ping:
            try:
                self._update_score_plot(getattr(child_program, "iteration_found", None), child_program)
            except Exception:
                pass
            return
        # Produce a fresh diagnosis by invoking coach_probe.py once, then read and set hint text
        try:
            self._append_coach_event({"event": "probe_call_start"})
            self._run_coach_probe_once()
            self._append_coach_event({"event": "probe_call_end", "status": "ok"})
        except Exception as _pe:
            self._append_coach_event({"event": "probe_call_end", "status": f"failed: {_pe}"})
        probe_reason = "ping" if escalate_from_ping else "best_plateau"
        hint_text = self._read_latest_coach_hint()
        if hint_text:
            plateau_note = ""
            if best_plateau:
                plateau_note = f" Plateau signal: best program unchanged for last {self._stable_best_count} iterations."
            # Strong exploration burst header to steer the next iterations
            burst_hdr = (
                "EXPLORE_BURST: diversify ideas; avoid repeating recent themes; propose materially different approaches aimed at large gains. "
                "Full-file diff permitted if needed (you may replace solve() and necessary imports)."
            )
            # Add near-best map when plateauing to surface rabbit holes explicitly
            near_best_block = ""
            try:
                # Show when plateauing or on ping-escalated stagnancy
                if best_plateau or (probe_reason == "ping"):
                    nb = self._build_near_best_summary()
                    if nb:
                        # Log near-best map for visibility
                        try:
                            self._append_coach_event({
                                "event": "near_best_map",
                                "lines": nb.split("\n"),
                            })
                        except Exception:
                            pass
                        near_best_block = f"\nNEAR_BEST_MAP:\n{nb}\n"
            except Exception:
                near_best_block = ""
            self._coach_hint_text = f"{burst_hdr}{near_best_block}ALERT: STAGNANCY DETECTED. {hint_text}{plateau_note}"
            # Enforce a minimum burst duration of 10 iterations for the injected hint
            self._coach_hint_remaining = max(int(getattr(self.config.coach, "hint_iters", 2) or 2), 10)
            # Start a burst window for macro exploration (10 iters), with reversible knob tweaks
            try:
                self._burst_active = True
                self._burst_iters_left = 10
                # Snapshot current knobs
                db_cfg = getattr(self.database, "config", None)
                self._pre_burst = {
                    "diff_based_evolution": bool(getattr(self.config, "diff_based_evolution", True)),
                    "exploration_ratio": getattr(db_cfg, "exploration_ratio", None) if db_cfg else None,
                    "exploitation_ratio": getattr(db_cfg, "exploitation_ratio", None) if db_cfg else None,
                    "num_diverse_programs": getattr(self.config.prompt, "num_diverse_programs", None),
                }
                # Apply temporary exploration re-balance for the burst
                if db_cfg is not None:
                    try:
                        db_cfg.exploration_ratio = 0.5
                        db_cfg.exploitation_ratio = 0.5
                    except Exception:
                        pass
                try:
                    if getattr(self.config, "prompt", None) and hasattr(self.config.prompt, "num_diverse_programs") and isinstance(self.config.prompt.num_diverse_programs, int):
                        self.config.prompt.num_diverse_programs = int(self.config.prompt.num_diverse_programs) + 3
                except Exception:
                    pass
                # Log burst start
                self._append_coach_event({
                    "event": "burst_start",
                    "burst_iters": self._burst_iters_left,
                    "pre_burst": self._pre_burst,
                })
            except Exception:
                # Fail-closed: if burst init fails, continue without burst
                self._burst_active = False
                self._burst_iters_left = 0
            # Apply cooldown only for plateau-triggered probes; ping-triggered probes do not set cooldown
            if probe_reason == "best_plateau":
                self._coach_cooldown = self.config.coach.cooldown
            logger.info("Coach diagnosis activated for next prompts.")
            # Log activation with simple stats
            try:
                self._append_coach_event({
                    "event": "trigger_stagnancy",
                    "window_size": win,
                    "best_stable_iters": self._stable_best_count,
                    "best_plateau": best_plateau,
                    "reason": probe_reason,
                    "hint": self._coach_hint_text,
                })
            except Exception:
                pass
        else:
            # No actionable diagnosis; set a short cooldown only for plateau-triggered probes
            if probe_reason == "best_plateau":
                self._coach_cooldown = self.config.coach.cooldown

    def _read_latest_coach_hint(self) -> str:
        """Read the last coach diagnosis and build a neutral one-line hint."""
        try:
            log_dir = self.config.log_dir or os.path.join(self.database.config.db_path or self.database.config.db_path or "", "logs")
            path = os.path.join(log_dir, "coach_probe.jsonl")
            if not os.path.isfile(path):
                return ""
            last_line = ""
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last_line = line.strip()
            if not last_line:
                return ""
            obj = json.loads(last_line)
            coach = obj.get("coach_reply", {})
            if not isinstance(coach, dict):
                return ""
            if coach.get("stagnant") is not True:
                return ""
            stuck_summary = coach.get("stuck_summary") or ""
            call_to_action = coach.get("call_to_action") or ""
            shoots = coach.get("shoots") if isinstance(coach.get("shoots"), list) else []
            avoid = coach.get("avoid") if isinstance(coach.get("avoid"), list) else []
            rabbit_holes = coach.get("rabbit_holes") if isinstance(coach.get("rabbit_holes"), list) else []
            big_bets = coach.get("big_bets") if isinstance(coach.get("big_bets"), list) else []
            success_criteria = coach.get("success_criteria") if isinstance(coach.get("success_criteria"), list) else []
            guardrails = coach.get("guardrails") if isinstance(coach.get("guardrails"), list) else []
            if stuck_summary:
                msg = f"Coach diagnosis: {stuck_summary}"
                if call_to_action:
                    msg = f"{msg} {call_to_action}"
                # Append concise shoots/avoid guidance if available
                if shoots:
                    msg = f"{msg} Shoots: " + "; ".join([str(s) for s in shoots[:3]])
                if avoid:
                    msg = f"{msg} Avoid: " + "; ".join([str(a) for a in avoid[:2]])
                # Append exploration-focused fields if available
                if rabbit_holes:
                    msg = f"{msg} Rabbit holes: " + "; ".join([str(rh) for rh in rabbit_holes[:3]])
                if big_bets:
                    msg = f"{msg} Big bets: " + "; ".join([str(bb) for bb in big_bets[:3]])
                if success_criteria:
                    msg = f"{msg} Success criteria: " + "; ".join([str(sc) for sc in success_criteria[:2]])
                if guardrails:
                    msg = f"{msg} Guardrails: " + "; ".join([str(gr) for gr in guardrails[:2]])
                return msg
            return ""
        except Exception as e:
            logger.debug(f"Reading coach diagnosis failed: {e}")
            return ""

    def _append_coach_event(self, record: Dict[str, Any]) -> None:
        """Append a coach event record to logs/coach_events.jsonl."""
        try:
            log_dir = self.config.log_dir
            if not log_dir:
                return
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(log_dir, "coach_events.jsonl")
            payload = {
                "ts": time.time(),
                **record,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Coach event log append failed: {e}")

    def _run_coach_probe_once(self) -> None:
        """
        Inline diagnosis: read last N intent rows, call LLM, append to logs/coach_probe.jsonl.
        Fail-closed on any error.
        """
        log_dir = self.config.log_dir
        if not log_dir:
            return
        output_dir = os.path.dirname(log_dir.rstrip(os.sep))
        intent_path = os.path.join(log_dir, "intent_log.jsonl")
        # Read last N rows
        if probe_read_last_n_jsonl:
            rows = probe_read_last_n_jsonl(intent_path, 10)
            context_rows = probe_build_minimal_context(rows)
        else:
            raise RuntimeError("coach_probe helpers unavailable; cannot build diagnosis context.")
        # Build minimal prompt (diagnosis-only)
        task_name = os.path.basename(os.path.dirname(output_dir.rstrip(os.sep)))
        if not (probe_read_best_metrics and probe_read_task_context and probe_make_prompt):
            raise RuntimeError("coach_probe helpers unavailable; cannot build diagnosis prompt.")
        best_metrics = probe_read_best_metrics(output_dir)
        task_context = probe_read_task_context(output_dir)
        # Include plateau metadata to guide the diagnosis toward decisive outcomes on plateaus
        win = int(getattr(self.config.coach, "window", 10) or 10)
        plateau = bool(self._stable_best_count >= win)
        prompt = probe_make_prompt(
            context_rows,
            task_name=task_name,
            best_metrics=best_metrics,
            task_context=task_context,
            best_plateau=plateau,
            best_stable_iters=int(self._stable_best_count or 0),
            window_size=win,
        )
        # Call LLM (diagnosis) with no explicit token cap
        model = "gpt-5"
        reply_text, meta = self._llm_call_json(model, prompt, max_tokens=None)
        try:
            reply_json = json.loads(reply_text) if reply_text else {"stagnant": False, "no_action": True, "raw": reply_text}
        except Exception:
            reply_json = {"stagnant": False, "no_action": True, "raw": reply_text}
        # Append a probe log line
        probe_path = os.path.join(log_dir, "coach_probe.jsonl")
        record = {
            "ts": time.time(),
            "intent_log_path": intent_path,
            "window_size": 10,
            "model": model,
            "context_rows": context_rows,
            "coach_reply": reply_json,
            "raw_reply": reply_text,
            "llm_meta": meta,
        }
        with open(probe_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Also log the exact LLM call (prompt + raw reply) for auditing
        probe_calls_path = os.path.join(log_dir, "coach_probe_calls.jsonl")
        call_rec = {
            "ts": time.time(),
            "model": model,
            "prompt": prompt,
            "raw_reply": reply_text,
            "llm_meta": meta,
        }
        with open(probe_calls_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(call_rec, ensure_ascii=False) + "\n")

    def _update_score_plot(self, iteration: Optional[int], child_program: "Program") -> None:
        """
        Append current iteration score to scores.csv and refresh score_plot.png (iters vs best score).
        Designed to be lightweight; silently no-op if matplotlib is unavailable.
        """
        log_dir = getattr(self.config, "log_dir", None)
        if not log_dir:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            return
        # Extract scores
        latest_score = None
        try:
            if isinstance(child_program.metrics, dict) and "combined_score" in child_program.metrics:
                latest_score = float(child_program.metrics["combined_score"])
        except Exception:
            latest_score = None
        if iteration is None or latest_score is None:
            return
        # CSV path
        csv_path = os.path.join(log_dir, "scores.csv")
        rows: List[Dict[str, Any]] = []
        # Load existing rows
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("iteration,"):
                            continue
                        parts = line.split(",")
                        if len(parts) >= 3:
                            try:
                                rows.append({
                                    "iteration": int(parts[0]),
                                    "latest_score": float(parts[1]),
                                    "best_score": float(parts[2]),
                                })
                            except Exception:
                                continue
            except Exception:
                rows = []
        # Compute best so far
        prev_best = max((r["best_score"] for r in rows), default=float("-inf"))
        best_score = max(prev_best, latest_score if latest_score is not None else float("-inf"))
        # Append current row
        rows.append({"iteration": int(iteration), "latest_score": latest_score, "best_score": best_score})
        # Rewrite CSV
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("iteration,latest_score,best_score\n")
                for r in rows:
                    f.write(f'{r["iteration"]},{r["latest_score"]},{r["best_score"]}\n')
        except Exception:
            pass
        # Try plotting (matplotlib optional)
        try:
            import matplotlib
            matplotlib.use("Agg")  # non-interactive backend
            import matplotlib.pyplot as plt  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return
        try:
            xs = np.array([r["iteration"] for r in rows], dtype=float)
            ys_best = np.array([r["best_score"] for r in rows], dtype=float)
            ys_latest = np.array([r["latest_score"] for r in rows], dtype=float)
            fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
            # Scatter-only, clearly distinct dots for best vs latest
            ax.scatter(xs, ys_best, label="best_score", s=22, c="#1f77b4", edgecolors="#1f77b4", zorder=3)
            ax.scatter(xs, ys_latest, label="latest_score", s=18, facecolors="none", edgecolors="#ff7f0e", zorder=2)
            # Annotate simple slope over last window points (best curve)
            window = max(3, int(getattr(self.config.coach, "window", 10) or 10))
            if len(xs) >= window:
                xw = xs[-window:]
                yw = ys_best[-window:]
                try:
                    # linear fit slope
                    coeffs = np.polyfit(xw, yw, 1)
                    slope = coeffs[0]
                    ax.set_title(f"Score vs Iteration (best)  |  last-{window} slope: {slope:+.4f}", fontsize=9)
                except Exception:
                    ax.set_title("Score vs Iteration (best)", fontsize=9)
            else:
                ax.set_title("Score vs Iteration (best)", fontsize=9)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Combined Score")
            ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            plot_path = os.path.join(log_dir, "score_plot.png")
            fig.savefig(plot_path)
            plt.close(fig)
        except Exception:
            # ignore plotting errors
            pass

    def _run_stagnancy_ping_once(self) -> bool:
        """
        Cheap yes/no stagnancy ping using a lightweight model.
        Returns True if stagnant, False otherwise.
        """
        log_dir = self.config.log_dir
        if not log_dir:
            return False
        output_dir = os.path.dirname(log_dir.rstrip(os.sep))
        intent_path = os.path.join(log_dir, "intent_log.jsonl")
        # Build minimal context rows
        context_rows: List[Dict[str, Any]] = []
        try:
            if probe_read_last_n_jsonl and probe_build_minimal_context:
                rows = probe_read_last_n_jsonl(intent_path, self.config.coach.window)
                context_rows = probe_build_minimal_context(rows)
            else:
                if os.path.isfile(intent_path):
                    with open(intent_path, "r", encoding="utf-8") as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    tail = lines[-self.config.coach.window :] if self.config.coach.window > 0 else lines
                    for ln in tail:
                        try:
                            obj = json.loads(ln)
                            context_rows.append({
                                "iteration": obj.get("iteration"),
                                "intent": (obj.get("llm_intent") or obj.get("intent") or ""),
                                "validator_output": obj.get("validator_output"),
                                "primary_metric": obj.get("primary_metric"),
                                "primary_metric_delta": obj.get("primary_metric_delta"),
                            })
                        except Exception:
                            continue
        except Exception:
            context_rows = []
        # Minimal ping prompt (short and strict)
        prompt = (
            "You are a concise monitor. Determine if recent attempts indicate stagnancy (local minimum or oscillation).\n"
            'Reply with STRICT JSON only: {"stagnant": true|false}\n'
            "Context rows (newest last): JSON array of {iteration,intent,validator_output,primary_metric,primary_metric_delta}.\n\n"
            f"{json.dumps(context_rows, ensure_ascii=False, separators=(',',':'))}"
        )
        # Call LLM
        model = getattr(self.config.coach, "ping_model", "gpt-5-mini") or "gpt-5-mini"
        text, meta = self._llm_call_json(model, prompt, max_tokens=None)
        # Log the ping call
        try:
            ping_calls_path = os.path.join(log_dir, "coach_ping_calls.jsonl")
            call_rec = {
                "ts": time.time(),
                "model": model,
                "prompt": prompt,
                "raw_reply": text,
                "llm_meta": meta,
            }
            with open(ping_calls_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(call_rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # Parse and append outcome
        stagnant = False
        try:
            obj = json.loads(text) if text else {}
            stagnant = bool(obj.get("stagnant")) if isinstance(obj, dict) else False
        except Exception:
            stagnant = False
        try:
            ping_path = os.path.join(log_dir, "coach_ping.jsonl")
            with open(ping_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.time(),
                    "stagnant": bool(stagnant),
                    "model": model,
                    "rows": context_rows,
                    "raw_reply": text,
                    "llm_meta": meta,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return bool(stagnant)

    # ---------------- inline coach helpers (LLM call only) ----------------

    def _llm_call_json(self, model: str, prompt: str, max_tokens: Optional[int] = None) -> tuple:
        """Call OpenAI Chat Completions with minimal JSON-style reply. Returns (text, meta)."""
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("OpenAI SDK not installed. pip install openai>=1.0.0") from e
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_APIKEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        client = OpenAI(api_key=api_key)
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful, concise JSON-only assistant. Reply with valid JSON."},
                {"role": "user", "content": prompt},
            ],
        }
        # Intentionally do not set any max token limits; only request JSON for non gpt-5 models
        if not model.startswith("gpt-5"):
            kwargs["response_format"] = {"type": "json_object"}
        completion = client.chat.completions.create(**kwargs)
        text = (completion.choices[0].message.content or "").strip()
        finish_reason = getattr(completion.choices[0], "finish_reason", None)
        usage = getattr(completion, "usage", None)
        meta = {
            "model": getattr(completion, "model", model),
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            },
        }
        return  text, meta

    # ---------------- burst helpers ----------------

    def _end_burst(self, reason: str = "") -> None:
        """Restore pre-burst knobs and end the exploration burst."""
        try:
            # Restore selection ratios
            db_cfg = getattr(self.database, "config", None)
            if db_cfg is not None:
                if isinstance(self._pre_burst.get("exploration_ratio", None), (int, float)):
                    db_cfg.exploration_ratio = self._pre_burst["exploration_ratio"]
                if isinstance(self._pre_burst.get("exploitation_ratio", None), (int, float)):
                    db_cfg.exploitation_ratio = self._pre_burst["exploitation_ratio"]
            # Restore prompt diversity (controller-side; workers may not reflect immediately)
            pre_ndp = self._pre_burst.get("num_diverse_programs", None)
            if pre_ndp is not None and getattr(self.config, "prompt", None) and hasattr(self.config.prompt, "num_diverse_programs"):
                try:
                    self.config.prompt.num_diverse_programs = int(pre_ndp)
                except Exception:
                    pass
            # Log
            try:
                self._append_coach_event({
                    "event": "burst_end",
                    "reason": reason or "complete",
                })
            except Exception:
                pass
        finally:
            self._burst_active = False
            self._burst_iters_left = 0
            self._pre_burst = {}

    # ---------------- near-best summary helpers ----------------

    def _build_near_best_summary(self, max_rows: int = 150, top_k: int = 5, near_ratio: float = 0.98) -> str:
        """
        Build a concise 'near-best map' from recent intent_log.jsonl:
        - Select last max_rows entries
        - Compute best combined_score in window
        - Keep up to top_k entries with score >= near_ratio * best
        - Return short lines: iter | score | delta | intent snippet
        """
        try:
            log_dir = getattr(self.config, "log_dir", None)
            if not log_dir:
                return ""
            path = os.path.join(log_dir, "intent_log.jsonl")
            if not os.path.isfile(path):
                return ""
            rows: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        rows.append(json.loads(s))
                    except Exception:
                        continue
            if not rows:
                return ""
            rows = rows[-max_rows:]
            # Extract scores
            scored: List[Dict[str, Any]] = []
            best_score = float("-inf")
            for r in rows:
                vo = r.get("validator_output") or {}
                score = None
                try:
                    if isinstance(vo, dict) and "combined_score" in vo:
                        score = float(vo["combined_score"])
                except Exception:
                    score = None
                if score is None:
                    continue
                best_score = max(best_score, score)
                scored.append({"row": r, "score": score})
            if not scored or not (best_score == best_score):
                return ""
            # Filter near-best
            cutoff = best_score * float(near_ratio)
            near = [x for x in scored if x["score"] >= cutoff]
            near.sort(key=lambda x: x["score"], reverse=True)
            near = near[:top_k]
            if not near:
                return ""
            # Render concise lines
            lines: List[str] = []
            header = f"Near-best (top {len(near)} within {int(near_ratio*100)}% of best={best_score:.4f}):"
            lines.append(header)
            for x in near:
                r = x["row"]
                it = r.get("iteration")
                sc = x["score"]
                d = r.get("primary_metric_delta")
                try:
                    delta = f"{float(d):+0.4f}" if isinstance(d, (int, float)) else str(d)
                except Exception:
                    delta = str(d)
                intent = r.get("intent")
                if isinstance(intent, dict):
                    snippet = str(intent.get("action") or intent.get("plan") or json.dumps(intent, ensure_ascii=False, separators=(",", ":")))
                else:
                    snippet = str(intent) if intent is not None else ""
                snippet = snippet.replace("\n", " ").strip()
                if len(snippet) > 180:
                    snippet = snippet[:180] + "..."
                lines.append(f"- iter {it} | score {sc:0.4f} | Δ {delta} | {snippet}")
            return "\n".join(lines)
        except Exception:
            return ""
