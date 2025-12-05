"""
Process-based parallel controller for true parallelism
"""

import asyncio
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
from openevolve.utils.metrics_utils import safe_numeric_average, get_fitness_score

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


@dataclass
class MultiChildSerializableResult:
    """Result for multi-child stagnation generation"""

    children: List[Dict[str, Any]] = None  # List of child dicts with program, prompt, etc.
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    iteration: int = 0
    error: Optional[str] = None
    num_generated: int = 0
    num_selected: int = 0


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

        # Get stagnation info (clean, optional)
        is_stagnant = db_snapshot.get("is_stagnant", False)
        best_program_code = db_snapshot.get("best_program_code")
        best_score = db_snapshot.get("best_score", 0.0)

        # Get generated paradigms if available (clean, optional)
        current_paradigms = db_snapshot.get("current_paradigms", [])
        paradigm_index = db_snapshot.get("paradigm_index", 0)

        # Get sibling programs (previous children of this parent) for context
        sibling_program_ids = db_snapshot.get("sibling_program_ids", [])
        sibling_programs = [programs[pid].to_dict() for pid in sibling_program_ids if pid in programs]

        # Calculate parent fitness for sibling context
        feature_dimensions = db_snapshot.get("feature_dimensions", [])
        parent_fitness = get_fitness_score(parent.metrics, feature_dimensions)

        # Get sampling mode for mode-aware prompts
        sampling_mode = db_snapshot.get("sampling_mode")
        
        iteration_start = time.time()
        # Only retry if error retry is enabled in config
        enable_retry = getattr(_worker_config.database, 'enable_error_retry', False) if hasattr(_worker_config, 'database') else False
        max_retries = getattr(_worker_config, 'max_error_retries', 2) if enable_retry else 0
        error_context = None
        
        # Get stagnation logger for retry logging (same logger instance)
        stagnation_logger = logging.getLogger(f"{__name__}.stagnation")
        
        # Retry loop for error recovery
        for retry_attempt in range(max_retries + 1):
            # Log retry trigger
            if retry_attempt > 0:
                paradigm_info = ""
                if current_paradigms and len(current_paradigms) > 0:
                    paradigm = current_paradigms[paradigm_index % len(current_paradigms)]
                    paradigm_info = f" | Paradigm: {paradigm.get('idea', 'N/A')[:60]}"
                
                error_msg_short = error_context[:200] if error_context else "Unknown error"
                stagnation_logger.info(
                    f"RETRY_TRIGGERED | Iteration {iteration} | "
                    f"Attempt {retry_attempt + 1}/{max_retries + 1} | "
                    f"Error: {error_msg_short}{paradigm_info}"
                )
            
            # Build prompt (with error context if retrying)
            prompt = _worker_prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in best_programs_only],
                top_programs=[p.to_dict() for p in programs_for_prompt],
                inspirations=[p.to_dict() for p in inspirations],
                language=_worker_config.language,
                evolution_round=iteration,
                diff_based_evolution=_worker_config.diff_based_evolution,
                program_artifacts=parent_artifacts,
                feature_dimensions=feature_dimensions,
                is_stagnant=is_stagnant,
                best_program_code=best_program_code,
                best_score=best_score,
                current_paradigms=current_paradigms,
                paradigm_index=paradigm_index,
                error_context=error_context,
                # Sibling context and mode-aware prompts
                sibling_programs=sibling_programs,
                parent_fitness=parent_fitness,
                sampling_mode=sampling_mode,
            )

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
                if retry_attempt < max_retries:
                    error_context = f"LLM generation failed: {str(e)}"
                    continue
                # Log final failure
                stagnation_logger.warning(
                    f"RETRY_EXHAUSTED | Iteration {iteration} | "
                    f"Attempts: {max_retries + 1} | "
                    f"Error: LLM generation failed: {str(e)[:200]}"
                )
                return SerializableResult(error=f"LLM generation failed: {str(e)}", iteration=iteration)

            # Check for None response
            if llm_response is None:
                if retry_attempt < max_retries:
                    error_context = "LLM returned None response"
                    continue
                # Log final failure
                stagnation_logger.warning(
                    f"RETRY_EXHAUSTED | Iteration {iteration} | "
                    f"Attempts: {max_retries + 1} | "
                    f"Error: LLM returned None response"
                )
                return SerializableResult(error="LLM returned None response", iteration=iteration)

            # Parse response based on evolution mode
            if _worker_config.diff_based_evolution:
                from openevolve.utils.code_utils import extract_diffs, apply_diff, format_diff_summary

                diff_blocks = extract_diffs(llm_response)
                if not diff_blocks:
                    if retry_attempt < max_retries:
                        error_context = "No valid diffs found in response"
                        continue
                    # Log final failure
                    stagnation_logger.warning(
                        f"RETRY_EXHAUSTED | Iteration {iteration} | "
                        f"Attempts: {max_retries + 1} | "
                        f"Error: No valid diffs found in response"
                    )
                    return SerializableResult(
                        error=f"No valid diffs found in response", iteration=iteration
                    )

                child_code = apply_diff(parent.code, llm_response)
                changes_summary = format_diff_summary(diff_blocks)
            else:
                from openevolve.utils.code_utils import parse_full_rewrite

                new_code = parse_full_rewrite(llm_response, _worker_config.language)
                if not new_code:
                    if retry_attempt < max_retries:
                        error_context = "No valid code found in response"
                        continue
                    # Log final failure
                    stagnation_logger.warning(
                        f"RETRY_EXHAUSTED | Iteration {iteration} | "
                        f"Attempts: {max_retries + 1} | "
                        f"Error: No valid code found in response"
                    )
                    return SerializableResult(
                        error=f"No valid code found in response", iteration=iteration
                    )

                child_code = new_code
                changes_summary = "Full rewrite"

            # Check code length
            if len(child_code) > _worker_config.max_code_length:
                if retry_attempt < max_retries:
                    error_context = f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})"
                    continue
                # Log final failure
                stagnation_logger.warning(
                    f"RETRY_EXHAUSTED | Iteration {iteration} | "
                    f"Attempts: {max_retries + 1} | "
                    f"Error: Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})"
                )
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

            # Check for evaluation errors (clean detection - no regex)
            score = child_metrics.get("combined_score", safe_numeric_average(child_metrics))
            has_error = (
                child_metrics.get("error") is not None or
                child_metrics.get("timeout") is True or
                (score == 0.0 and 
                 artifacts and 
                 (artifacts.get("stderr") or artifacts.get("traceback"))) or
                (score == 0.0 and 
                 child_metrics.get("success_rate", 1.0) == 0.0)  # Score 0 with 0 success rate = correctness issue, retry
            )

            # If no error, success - return result
            if not has_error:
                # Log retry success if this was a retry
                if retry_attempt > 0:
                    score = child_metrics.get("combined_score", safe_numeric_average(child_metrics))
                    stagnation_logger.info(
                        f"RETRY_SUCCESS | Iteration {iteration} | "
                        f"Attempt {retry_attempt + 1}/{max_retries + 1} | "
                        f"Score: {score:.6f}"
                    )
                
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
                        "retry_attempt": retry_attempt if retry_attempt > 0 else None,
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

            # Error detected - prepare error context for retry
            if retry_attempt < max_retries:
                # Extract error message cleanly (no regex)
                error_msg = None
                if artifacts:
                    error_msg = artifacts.get("stderr") or artifacts.get("traceback")
                if not error_msg:
                    error_msg = str(child_metrics.get("error", "")) or "Evaluation failed"
                
                if child_metrics.get("timeout"):
                    error_context = f"Evaluation timed out after {_worker_config.evaluator.timeout}s. {error_msg if error_msg != 'Evaluation failed' else ''}"
                else:
                    error_context = f"Evaluation error: {error_msg}"
                
                logger.debug(f"Iteration {iteration} attempt {retry_attempt + 1} failed with error, retrying...")
                continue

            # All retries exhausted - return error result
            # Log retry exhaustion
            error_msg = None
            if artifacts:
                error_msg = artifacts.get("stderr") or artifacts.get("traceback")
            if not error_msg:
                error_msg = str(child_metrics.get("error", "")) or "Evaluation failed"
            
            error_msg_short = error_msg[:200] if error_msg else "Unknown error"
            stagnation_logger.warning(
                f"RETRY_EXHAUSTED | Iteration {iteration} | "
                f"Attempts: {max_retries + 1} | "
                f"Final error: {error_msg_short}"
            )
            
            # Create child program with error metrics for tracking
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
                    "retry_attempt": retry_attempt,
                    "error_retries_exhausted": True,
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


def _run_stagnation_multi_child_worker(
    iteration: int,
    db_snapshot: Dict[str, Any],
    parent_id: str,
    inspiration_ids: List[str],
    num_children: int = 3,
) -> MultiChildSerializableResult:
    """Run multi-child generation for a stagnating island.

    Generates multiple children SEQUENTIALLY, with each child seeing
    the previous ones as sibling context. This helps break out of
    local optima by exploring multiple directions from the same parent.
    """
    try:
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

        programs_for_prompt = island_programs[
            : _worker_config.prompt.num_top_programs + _worker_config.prompt.num_diverse_programs
        ]
        best_programs_only = island_programs[: _worker_config.prompt.num_top_programs]

        # Feature dimensions for fitness calculation
        feature_dimensions = db_snapshot.get("feature_dimensions", [])
        parent_fitness = get_fitness_score(parent.metrics, feature_dimensions)

        iteration_start = time.time()
        children_results = []
        generated_siblings = []  # Track children as we generate them for sibling context

        for child_idx in range(num_children):
            try:
                # Build prompt with sibling context (previous children in this batch)
                sibling_programs_for_prompt = [
                    {
                        "code": c["code"],
                        "metrics": c["metrics"],
                        "iteration_found": c["iteration_found"],
                    }
                    for c in generated_siblings
                ]

                prompt = _worker_prompt_sampler.build_prompt(
                    current_program=parent.code,
                    parent_program=parent.code,
                    program_metrics=parent.metrics,
                    previous_programs=[p.to_dict() for p in best_programs_only],
                    top_programs=[p.to_dict() for p in programs_for_prompt],
                    inspirations=[p.to_dict() for p in inspirations],
                    language=_worker_config.language,
                    evolution_round=iteration,
                    diff_based_evolution=_worker_config.diff_based_evolution,
                    program_artifacts=parent_artifacts,
                    feature_dimensions=feature_dimensions,
                    # Sibling context for multi-child
                    sibling_programs=sibling_programs_for_prompt,
                    parent_fitness=parent_fitness,
                    sampling_mode="exploration",  # Force exploration for stagnation
                )

                # Generate code
                llm_response = asyncio.run(
                    _worker_llm_ensemble.generate_with_context(
                        system_message=prompt["system"],
                        messages=[{"role": "user", "content": prompt["user"]}],
                    )
                )

                if llm_response is None:
                    logger.warning(f"Multi-child {child_idx + 1}: LLM returned None")
                    continue

                # Parse response
                if _worker_config.diff_based_evolution:
                    from openevolve.utils.code_utils import extract_diffs, apply_diff, format_diff_summary

                    diff_blocks = extract_diffs(llm_response)
                    if not diff_blocks:
                        logger.warning(f"Multi-child {child_idx + 1}: No valid diffs")
                        continue
                    child_code = apply_diff(parent.code, llm_response)
                    changes_summary = format_diff_summary(diff_blocks)
                else:
                    from openevolve.utils.code_utils import parse_full_rewrite

                    new_code = parse_full_rewrite(llm_response, _worker_config.language)
                    if not new_code:
                        logger.warning(f"Multi-child {child_idx + 1}: No valid code")
                        continue
                    child_code = new_code
                    changes_summary = "Full rewrite"

                # Check code length
                if len(child_code) > _worker_config.max_code_length:
                    logger.warning(f"Multi-child {child_idx + 1}: Code too long")
                    continue

                # Evaluate
                import uuid
                child_id = str(uuid.uuid4())
                child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))
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
                        "stagnation_child_index": child_idx,
                        "stagnation_batch_size": num_children,
                    },
                )

                # Add to results
                children_results.append({
                    "child_program_dict": child_program.to_dict(),
                    "prompt": prompt,
                    "llm_response": llm_response,
                    "artifacts": artifacts,
                })

                # Add to sibling context for next child
                generated_siblings.append({
                    "code": child_code,
                    "metrics": child_metrics,
                    "iteration_found": iteration,
                })

                logger.info(
                    f"Multi-child {child_idx + 1}/{num_children}: "
                    f"Generated {child_id} with score "
                    f"{child_metrics.get('combined_score', safe_numeric_average(child_metrics)):.4f}"
                )

            except Exception as e:
                logger.warning(f"Multi-child {child_idx + 1} failed: {e}")
                continue

        iteration_time = time.time() - iteration_start

        # Select best child
        if children_results:
            children_results.sort(
                key=lambda c: c["child_program_dict"]["metrics"].get(
                    "combined_score",
                    safe_numeric_average(c["child_program_dict"]["metrics"])
                ),
                reverse=True,
            )

        return MultiChildSerializableResult(
            children=children_results,
            parent_id=parent.id,
            iteration_time=iteration_time,
            iteration=iteration,
            num_generated=len(children_results),
            num_selected=1 if children_results else 0,
        )

    except Exception as e:
        logger.exception(f"Error in multi-child worker iteration {iteration}")
        return MultiChildSerializableResult(
            error=str(e),
            iteration=iteration,
            children=[],
        )


class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase, evolution_tracer=None, file_suffix: str = ".py", output_dir: Optional[str] = None):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.evolution_tracer = evolution_tracer
        self.file_suffix = file_suffix
        self.output_dir = output_dir  # Store output_dir for logging

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations
        self.num_islands = config.database.num_islands

        # Stagnation tracking (clean, optional feature)
        self.best_score_history: List[float] = []
        self.is_stagnant: bool = False
        self.stagnation_logger: Optional[logging.Logger] = None
        self._setup_stagnation_logging()
        
        # Async paradigm generation (clean, simple)
        self.paradigm_generation_task: Optional[asyncio.Task] = None
        self.current_paradigms: List[Dict[str, Any]] = []
        self.paradigms_used_count: int = 0
        self.paradigms_uses_per_paradigm: int = 2  # Use each paradigm 2 times
        self.iteration_paradigm_map: Dict[int, Dict[str, Any]] = {}  # Track which iteration used which paradigm
        
        # Track previously tried paradigms and their outcomes
        self.previously_tried_paradigms: List[Dict[str, Any]] = []  # List of tried ideas with outcomes

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")
    
    def _setup_stagnation_logging(self) -> None:
        """Setup separate logger for stagnation monitoring (clean, optional feature)"""
        # Only setup if paradigm breakthrough is enabled
        if not getattr(self.config.database, 'enable_paradigm_breakthrough', False):
            self.stagnation_logger = None
            return
        
        try:
            # Get log directory - use same as main logger (output_dir/logs)
            if self.config.log_dir:
                log_dir = Path(self.config.log_dir)
            elif self.output_dir:
                # Use output_dir/logs (same as main logger)
                log_dir = Path(self.output_dir) / "logs"
            else:
                # Fallback to evaluation_file parent / logs
                log_dir = Path(self.evaluation_file).parent / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create stagnation logger
            stagnation_logger = logging.getLogger(f"{__name__}.stagnation")
            stagnation_logger.setLevel(logging.INFO)
            
            # Remove existing handlers to avoid duplicates
            stagnation_logger.handlers.clear()
            
            # Create file handler for stagnation log with immediate flushing
            log_file = log_dir / "stagnation_monitor.log"
            
            # Create formatter with timestamp
            formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            
            # Create custom handler that flushes immediately after each log
            class FlushingFileHandler(logging.FileHandler):
                def emit(self, record):
                    super().emit(record)
                    self.flush()
            
            flushing_handler = FlushingFileHandler(log_file, mode='a')
            flushing_handler.setLevel(logging.INFO)
            flushing_handler.setFormatter(formatter)
            
            stagnation_logger.addHandler(flushing_handler)
            stagnation_logger.propagate = False  # Don't propagate to root logger
            
            self.stagnation_logger = stagnation_logger
            logger.info(f"✅ Stagnation logging initialized: {log_file}")
            
            # Log initial setup
            if self.stagnation_logger:
                self.stagnation_logger.info(
                    f"INIT | Stagnation monitoring initialized | "
                    f"Window: {getattr(self.config, 'stagnation_window', 5)} | "
                    f"Threshold: {getattr(self.config, 'stagnation_improvement_threshold', 0.01)*100:.2f}%"
                )
            
        except Exception as e:
            # Never break initialization - just log and continue without stagnation logger
            logger.warning(f"Failed to setup stagnation logging (non-fatal): {e}")
            self.stagnation_logger = None

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
    
    def _check_stagnation(self, iteration: int, iteration_score: Optional[float] = None) -> None:
        """Check for stagnation and update state (clean, non-intrusive)"""
        # Skip if paradigm breakthrough is disabled
        if not getattr(self.config.database, 'enable_paradigm_breakthrough', False):
            return
        
        try:
            best_program = self.database.get_best_program()
            if not best_program or not best_program.metrics:
                return
            
            # Get best score
            best_score = best_program.metrics.get("combined_score")
            if best_score is None:
                best_score = safe_numeric_average(best_program.metrics)
            
            # Update history
            self.best_score_history.append(best_score)
            window = getattr(self.config, 'stagnation_window', 5)
            if len(self.best_score_history) > window:
                self.best_score_history.pop(0)
            
            # Log every iteration (even before window is reached)
            if self.stagnation_logger:
                iteration_score_str = f" | Iteration score: {iteration_score:.6f}" if iteration_score is not None else ""
                if len(self.best_score_history) < window:
                    # Building history
                    self.stagnation_logger.info(
                        f"CHECK | Iteration {iteration} | "
                        f"Best score: {best_score:.6f}{iteration_score_str} | "
                        f"History: [{', '.join(f'{s:.6f}' for s in self.best_score_history)}] | "
                        f"Building history ({len(self.best_score_history)}/{window})"
                    )
            
            # Check stagnation (need at least window size to check)
            if len(self.best_score_history) >= window:
                old_best = self.best_score_history[0]
                new_best = self.best_score_history[-1]
                threshold = getattr(self.config, 'stagnation_improvement_threshold', 0.01)
                
                # Calculate improvement rate (total improvement over the window period)
                if old_best > 0:
                    improvement_rate = (new_best - old_best) / old_best
                else:
                    improvement_rate = 0.0
                
                was_stagnant = self.is_stagnant
                self.is_stagnant = improvement_rate < threshold
                
                # Log stagnation check (simple)
                if self.stagnation_logger:
                    iteration_score_str = f" | Iteration score: {iteration_score:.6f}" if iteration_score is not None else ""
                    self.stagnation_logger.info(
                        f"CHECK | Iteration {iteration} | "
                        f"Best score: {best_score:.6f}{iteration_score_str} | "
                        f"History: [{', '.join(f'{s:.6f}' for s in self.best_score_history)}] | "
                        f"Improvement rate: {improvement_rate*100:.2f}% | "
                        f"Stagnant: {self.is_stagnant}"
                    )
                
                # Log when stagnation state changes
                if was_stagnant != self.is_stagnant:
                    if self.is_stagnant:
                        logger.info(f"⚠️  Stagnation detected at iteration {iteration}")
                        if self.stagnation_logger:
                            self.stagnation_logger.info(
                                f"STAGNATION DETECTED | Iteration {iteration} | "
                                f"Best score: {best_score:.6f}"
                            )
                        
                        # Start async paradigm generation if not already running
                        if self.paradigm_generation_task is None or self.paradigm_generation_task.done():
                            self.paradigm_generation_task = asyncio.create_task(
                                self._generate_paradigms_async(best_program.code, best_score)
                            )
                            logger.info("Started async paradigm generation")
                            if self.stagnation_logger:
                                self.stagnation_logger.info(
                                    f"PARADIGMS_GENERATION_STARTED | Iteration: {iteration} | "
                                    f"Best score: {best_score:.6f}"
                                )
                    else:
                        # Stagnation cleared - cancel paradigm generation if still running
                        if self.paradigm_generation_task and not self.paradigm_generation_task.done():
                            self.paradigm_generation_task.cancel()
                            logger.info(f"⚠️ Iteration {iteration}: Stagnation cleared - cancelled paradigm generation")
                            if self.stagnation_logger:
                                self.stagnation_logger.info(
                                    f"PARADIGM_GENERATION_CANCELLED | Iteration: {iteration} | "
                                    f"Reason: Stagnation cleared"
                                )
                        # Clear paradigms if any
                        if self.current_paradigms:
                            cleared_count = len(self.current_paradigms)
                            self.current_paradigms = []
                            self.paradigms_used_count = 0
                            logger.info(f"⚠️ Iteration {iteration}: Stagnation cleared - cleared {cleared_count} paradigms")
                            if self.stagnation_logger:
                                self.stagnation_logger.info(
                                    f"PARADIGMS_CLEARED | Iteration: {iteration} | "
                                    f"Reason: Stagnation cleared | "
                                    f"Paradigms cleared: {cleared_count}"
                                )
                        logger.info(f"✅ Stagnation cleared at iteration {iteration}")
                        if self.stagnation_logger:
                            self.stagnation_logger.info(
                                f"STAGNATION CLEARED | Iteration {iteration} | "
                                f"Best score: {best_score:.6f}"
                            )
        except Exception as e:
            # Never break evolution - just log and continue
            logger.debug(f"Stagnation check error (non-fatal): {e}")
    
    async def _generate_paradigms_async(self, best_program_code: str, best_score: float) -> None:
        """Generate paradigms asynchronously (robust, well-coded)"""
        try:
            # Get paths
            example_dir = Path(self.evaluation_file).parent
            config_path = example_dir / "config.yaml"
            evaluator_path = Path(self.evaluation_file)
            
            if not config_path.exists():
                if self.stagnation_logger:
                    self.stagnation_logger.warning(f"PARADIGMS_GENERATION_FAILED | Config not found: {config_path}")
                logger.debug(f"Paradigm generation skipped: config not found at {config_path}")
                return
            
            if not evaluator_path.exists():
                if self.stagnation_logger:
                    self.stagnation_logger.warning(f"PARADIGMS_GENERATION_FAILED | Evaluator not found: {evaluator_path}")
                logger.debug(f"Paradigm generation skipped: evaluator not found at {evaluator_path}")
                return
            
            # Get API key from config or environment (config takes priority)
            api_key = getattr(self.config.database, 'paradigm_api_key', None) or os.getenv("OPENAI_API_KEY")
            if not api_key:
                if self.stagnation_logger:
                    self.stagnation_logger.warning("PARADIGMS_GENERATION_FAILED | No API key configured (set paradigm_api_key in config or OPENAI_API_KEY env var)")
                logger.warning("Paradigm generation skipped: No API key configured")
                return
            
            # Import generate_paradigms function
            import sys
            import importlib.util
            module_path = Path(__file__).parent.parent / "generate_breakthrough_paradigms.py"
            if not module_path.exists():
                if self.stagnation_logger:
                    self.stagnation_logger.warning(f"PARADIGMS_GENERATION_FAILED | Module not found: {module_path}")
                logger.error(f"Paradigm generation module not found: {module_path}")
                return
            
            spec = importlib.util.spec_from_file_location("generate_breakthrough_paradigms", module_path)
            if spec is None or spec.loader is None:
                if self.stagnation_logger:
                    self.stagnation_logger.warning("PARADIGMS_GENERATION_FAILED | Failed to load module")
                logger.error("Failed to load paradigm generation module")
                return
            
            module = importlib.util.module_from_spec(spec)
            sys.modules["generate_breakthrough_paradigms"] = module
            spec.loader.exec_module(module)
            generate_paradigms = module.generate_paradigms
            
            # Get model from config (default: gpt-5-mini)
            model = getattr(self.config.database, 'paradigm_model', 'gpt-5-mini')
            temperature = 1.0  # Default temperature for paradigm generation
            
            if self.stagnation_logger:
                self.stagnation_logger.info(f"PARADIGMS_GENERATION_STARTED | Model: {model} | Temperature: {temperature} | Best score: {best_score:.6f}")
            
            # Build previously tried ideas list grouped by approach type
            # Group by approach_type to help LLM recognize patterns
            approach_groups = {}
            for tried in self.previously_tried_paradigms:
                idea = tried.get('idea', '')
                approach_type = tried.get('approach_type', '')
                # Use LLM-provided approach_type, default to 'Other' if missing
                if not approach_type or approach_type is None:
                    approach_type = 'Other'
                
                status = tried.get('status', 'unclear')
                reason = tried.get('reason', '')
                score = tried.get('score', 0.0)
                best_before = tried.get('best_score_before', 0.0)
                
                if approach_type not in approach_groups:
                    approach_groups[approach_type] = {'success': [], 'failed': [], 'unclear': []}
                
                # Calculate improvement percentage
                if best_before > 0:
                    pct_change = ((score - best_before) / best_before) * 100
                    pct_str = f"{pct_change:+.2f}%"
                else:
                    pct_str = "N/A"
                
                # Keep idea description full (not truncated) so LLM can see what was tried
                idea_desc = idea[:100] if len(idea) > 100 else idea
                # Add complexity indicator if idea suggests multi-step approach (robust detection)
                complexity_hint = ""
                idea_lower = idea_desc.lower()
                # Check for clear multi-step patterns (avoid false positives)
                multi_step_patterns = [
                    ' then ',  # "X then Y" pattern
                    ' then apply',  # "X then apply Y"
                    ' then use',  # "X then use Y"
                    ' after ',  # "X after Y" pattern
                    ' combine ',  # "combine X and Y"
                    ' multi-step',  # explicit multi-step
                    'multi-stage',  # explicit multi-stage
                    ' first ',  # "first X then Y" pattern
                    ' second ',  # "first X second Y" pattern
                    ' followed by',  # "X followed by Y"
                    ' and then',  # "X and then Y"
                ]
                # Only flag if we see clear multi-step patterns (not just single words that might be part of normal text)
                if any(pattern in idea_lower for pattern in multi_step_patterns):
                    complexity_hint = " [COMPLEX MULTI-STEP - likely failed due to complexity]"
                
                if status == 'success':
                    entry = f"{idea_desc} | Score: {best_before:.6f} → {score:.6f} ({pct_str}){complexity_hint}"
                    approach_groups[approach_type]['success'].append(entry)
                elif status == 'failed' or status == 'correctness':
                    if status == 'correctness':
                        entry = f"{idea_desc} | Score: {best_before:.6f} → 0.000000 (correctness issue){complexity_hint}"
                    else:
                        entry = f"{idea_desc} | Score: {best_before:.6f} → {score:.6f} ({pct_str}){complexity_hint}"
                    approach_groups[approach_type]['failed'].append(entry)
                else:
                    # Unclear results also get complexity hint if applicable
                    entry = f"{idea_desc} | Score: {best_before:.6f} → {score:.6f} ({pct_str}){complexity_hint}"
                    approach_groups[approach_type]['unclear'].append(entry)
            
            # Format grouped feedback
            previously_tried_ideas = []
            if approach_groups:
                previously_tried_ideas.append("## CRITICAL: Previously Tried Ideas - ANALYZE THIS FIRST")
                previously_tried_ideas.append("")
                previously_tried_ideas.append("**IMPORTANT:** Ideas are grouped by approach type. Review what was tried and what happened.")
                previously_tried_ideas.append("")
                
                # Show successful approaches first
                successful_types = [at for at, group in approach_groups.items() if group['success']]
                if successful_types:
                    previously_tried_ideas.append("### Successful Approach Types (LEARN FROM THESE):")
                    previously_tried_ideas.append("**When an approach succeeds, identify the underlying principle that made it work, don't just add complexity.**")
                    for approach_type in successful_types:
                        group = approach_groups[approach_type]
                        previously_tried_ideas.append(f"**{approach_type}:**")
                        for entry in group['success']:
                            previously_tried_ideas.append(f"  ✅ {entry}")
                    previously_tried_ideas.append("")
                
                # Show failed approaches (present as facts, let LLM decide)
                failed_types = [at for at, group in approach_groups.items() if group['failed']]
                if failed_types:
                    previously_tried_ideas.append("### Previously Tried Approaches (What Happened):")
                    for approach_type in failed_types:
                        group = approach_groups[approach_type]
                        previously_tried_ideas.append(f"**{approach_type}:**")
                        for entry in group['failed']:
                            previously_tried_ideas.append(f"  {entry}")
                    previously_tried_ideas.append("")
                
                # Show unclear results
                unclear_types = [at for at, group in approach_groups.items() if group['unclear']]
                if unclear_types:
                    previously_tried_ideas.append("### Unclear Results:")
                    for approach_type in unclear_types:
                        group = approach_groups[approach_type]
                        for entry in group['unclear']:
                            previously_tried_ideas.append(f"  ➡️  [{approach_type}] {entry}")
                    previously_tried_ideas.append("")
                
                # Guidance on how to interpret feedback is now in the main prompt - no duplication needed here
                previously_tried_ideas.append("")
            
            # Log feedback being passed to LLM (for debugging)
            if self.stagnation_logger and previously_tried_ideas:
                self.stagnation_logger.info("PREVIOUSLY_TRIED_FEEDBACK | Feedback being passed to LLM:")
                for line in previously_tried_ideas[:30]:  # Log first 30 lines to avoid huge logs
                    self.stagnation_logger.info(f"  {line}")
                if len(previously_tried_ideas) > 30:
                    self.stagnation_logger.info(f"  ... ({len(previously_tried_ideas) - 30} more lines)")
            
            # Call LLM to generate paradigms with timeout
            # Increased timeout for GPT-5 models with thinking tokens (can take 10+ minutes)
            try:
                # Get API base from config (default: OpenAI)
                api_base = getattr(self.config.database, 'paradigm_api_base', 'https://api.openai.com/v1')
                
                paradigms = await asyncio.wait_for(
                    generate_paradigms(
                        config_path=str(config_path),
                        evaluator_path=str(evaluator_path),
                        api_key=api_key,
                        api_base=api_base,
                        model=model,
                        temperature=temperature,
                        output_path=None,
                        best_program_code=best_program_code,
                        best_score=best_score,
                        previously_tried_ideas=previously_tried_ideas if previously_tried_ideas else None,
                    ),
                    timeout=900.0  # 15 minutes - GPT-5 with thinking tokens needs more time
                )
            except asyncio.TimeoutError:
                if self.stagnation_logger:
                    self.stagnation_logger.warning("PARADIGMS_GENERATION_FAILED | Timeout after 15 minutes")
                logger.warning("Paradigm generation timed out after 15 minutes")
                return
            except Exception as e:
                if self.stagnation_logger:
                    self.stagnation_logger.warning(f"PARADIGMS_GENERATION_FAILED | LLM call error: {e}")
                logger.warning(f"Paradigm generation LLM call failed: {e}")
                return
            
            # Validate paradigms
            if not paradigms:
                if self.stagnation_logger:
                    self.stagnation_logger.warning("PARADIGMS_GENERATION_FAILED | Empty list returned")
                logger.warning("Paradigm generation returned empty list")
                return
            
            if not isinstance(paradigms, list):
                if self.stagnation_logger:
                    self.stagnation_logger.warning(f"PARADIGMS_GENERATION_FAILED | Invalid type: {type(paradigms)}")
                logger.warning(f"Paradigm generation returned invalid type: {type(paradigms)}")
                return
            
            # Use paradigms (simple - no effectiveness check)
            self.current_paradigms = paradigms
            self.paradigms_used_count = 0
            
            if self.stagnation_logger:
                self.stagnation_logger.info(
                    f"PARADIGMS_GENERATED | Count: {len(paradigms)} | "
                    f"Best score: {best_score:.6f}"
                )
                for i, p in enumerate(paradigms, 1):
                    if not isinstance(p, dict):
                        continue
                    self.stagnation_logger.info("")
                    self.stagnation_logger.info(f"PARADIGM_{i}_FULL:")
                    self.stagnation_logger.info(f"  Idea: {p.get('idea', 'N/A')}")
                    self.stagnation_logger.info(f"  Description: {p.get('description', 'N/A')}")
                    self.stagnation_logger.info(f"  What to optimize: {p.get('what_to_optimize', 'N/A')}")
                    self.stagnation_logger.info(f"  Cautions: {p.get('cautions', 'N/A')}")
                    self.stagnation_logger.info("")
            
            logger.info(f"✅ Generated {len(paradigms)} breakthrough paradigms")
                
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_str = traceback.format_exc()
            
            if self.stagnation_logger:
                self.stagnation_logger.warning(f"PARADIGMS_GENERATION_FAILED | Error: {error_msg}")
                self.stagnation_logger.debug(f"Traceback: {traceback_str}")
            
            logger.error(f"Paradigm generation error: {error_msg}")
            logger.debug(f"Paradigm generation traceback: {traceback_str}")

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

        # Include stagnation info (clean, optional)
        snapshot["is_stagnant"] = self.is_stagnant
        
        # Include generated paradigms if available (clean, optional)
        # Note: Actual paradigm selection happens in submission loop, not here
        # This is just for reference - the real paradigm is passed to _submit_iteration
        snapshot["current_paradigms"] = self.current_paradigms
        snapshot["paradigm_index"] = 0  # Will be set in _submit_iteration based on passed paradigm
        
        # Get best program info for stagnation prompt
        best_program = self.database.get_best_program()
        if best_program:
            snapshot["best_program_code"] = best_program.code
            snapshot["best_score"] = best_program.metrics.get("combined_score", safe_numeric_average(best_program.metrics))
        else:
            snapshot["best_program_code"] = None
            snapshot["best_score"] = 0.0

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

                # Handle MultiChildSerializableResult (from multi-child stagnation worker)
                if isinstance(result, MultiChildSerializableResult):
                    if result.error:
                        logger.warning(f"Iteration {completed_iteration} multi-child error: {result.error}")
                    elif result.children:
                        # Add all children to database (sorted by score, best first)
                        logger.info(
                            f"Iteration {completed_iteration}: Multi-child stagnation generated "
                            f"{result.num_generated} children"
                        )
                        for child_data in result.children:
                            child_program = Program(**child_data["child_program_dict"])
                            self.database.add(child_program, iteration=completed_iteration)

                            if child_data.get("artifacts"):
                                self.database.store_artifacts(child_program.id, child_data["artifacts"])

                            # Log evolution trace for each child
                            if self.evolution_tracer:
                                parent_program = self.database.get(result.parent_id) if result.parent_id else None
                                if parent_program:
                                    island_id = child_program.metadata.get("island", self.database.current_island)
                                    self.evolution_tracer.log_trace(
                                        iteration=completed_iteration,
                                        parent_program=parent_program,
                                        child_program=child_program,
                                        prompt=child_data.get("prompt"),
                                        llm_response=child_data.get("llm_response"),
                                        artifacts=child_data.get("artifacts"),
                                        island_id=island_id,
                                        metadata={
                                            "iteration_time": result.iteration_time,
                                            "changes": child_program.metadata.get("changes", ""),
                                            "multi_child": True,
                                            "stagnation_child_index": child_program.metadata.get("stagnation_child_index"),
                                        }
                                    )

                            # Log prompt for each child
                            if child_data.get("prompt"):
                                self.database.log_prompt(
                                    template_key="multi_child_user",
                                    program_id=child_program.id,
                                    prompt=child_data["prompt"],
                                    responses=[child_data.get("llm_response")] if child_data.get("llm_response") else [],
                                )

                            score = child_program.metrics.get("combined_score", safe_numeric_average(child_program.metrics))
                            child_idx = child_program.metadata.get("stagnation_child_index", "?")
                            logger.info(
                                f"  Child {child_idx}: {child_program.id} score={score:.4f}"
                            )

                        # Check for new best after adding all children
                        best_program = self.database.get_best_program()
                        if best_program and result.children:
                            best_child = result.children[0]  # Already sorted, best first
                            if best_program.id == best_child["child_program_dict"]["id"]:
                                logger.info(
                                    f"🌟 New best from multi-child at iteration {completed_iteration}: "
                                    f"{best_program.id}"
                                )

                        # Check stagnation after multi-child
                        self._check_stagnation(completed_iteration)

                        # Island management
                        current_island_counter += result.num_generated
                        self.database.increment_island_generation()

                        if self.database.should_migrate():
                            logger.info(f"Performing migration at iteration {completed_iteration}")
                            self.database.migrate_programs()
                            self.database.log_island_status()

                # Handle regular SerializableResult
                elif result.error:
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

                    # Log progress (mark if used paradigms - check iteration_paradigm_map for accurate tracking)
                    paradigm_marker = ""
                    paradigm_info = ""
                    # Check if this iteration used a paradigm (even if paradigms were cleared)
                    paradigm_used = self.iteration_paradigm_map.get(completed_iteration)
                    if paradigm_used:
                        paradigm_marker = " [BREAKTHROUGH PARADIGM]"
                        paradigm_info = f" | Paradigm: {paradigm_used.get('idea', 'N/A')[:40]}"
                        
                        # Track paradigm result for future learning
                        iteration_score = child_program.metrics.get('combined_score', safe_numeric_average(child_program.metrics))
                        best_score_before = paradigm_used.get('best_score_before', 0.0)
                        idea = paradigm_used.get('idea', '')
                        
                        # Determine status: success = any increase, failure = any decrease
                        if iteration_score == 0.0:
                            status = 'correctness'
                            reason = 'Score 0 - correctness issue, hard to implement'
                        elif iteration_score > best_score_before:
                            status = 'success'
                            reason = f'Improved from {best_score_before:.6f} to {iteration_score:.6f}'
                        elif iteration_score < best_score_before:
                            status = 'failed'
                            reason = f'Regressed from {best_score_before:.6f} to {iteration_score:.6f}'
                        else:
                            status = 'unclear'
                            reason = 'No change'
                        
                        # Extract approach_type from paradigm (if available)
                        approach_type = paradigm_used.get('approach_type', '')
                        # Use LLM-provided approach_type, default to 'Other' if missing
                        if not approach_type or approach_type is None:
                            approach_type = 'Other'
                        
                        # Store result for future paradigm generation
                        self.previously_tried_paradigms.append({
                            'idea': idea,
                            'approach_type': approach_type,
                            'score': iteration_score,
                            'best_score_before': best_score_before,
                            'improvement': iteration_score - best_score_before,
                            'status': status,
                            'reason': reason
                        })
                        
                        # Keep only last 10 entries (very recent attempts only)
                        if len(self.previously_tried_paradigms) > 10:
                            self.previously_tried_paradigms = self.previously_tried_paradigms[-10:]
                        
                        # Log to stagnation logger
                        if self.stagnation_logger:
                            best_score = self.database.get_best_program().metrics.get('combined_score', safe_numeric_average(self.database.get_best_program().metrics)) if self.database.get_best_program() else 0.0
                            self.stagnation_logger.info(
                                f"PARADIGM_COMPLETED | Iteration {completed_iteration} | "
                                f"Program {child_program.id} | "
                                f"Iteration score: {iteration_score:.6f} | "
                                f"Best score: {best_score:.6f} | "
                                f"Paradigm idea: {paradigm_used.get('idea', 'N/A')[:80]} | "
                                f"Approach type: {approach_type} | "
                                f"Status: {status}"
                            )
                        # Clean up mapping after logging (safe delete)
                        if completed_iteration in self.iteration_paradigm_map:
                            del self.iteration_paradigm_map[completed_iteration]
                    
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s{paradigm_marker}{paradigm_info}"
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
                    
                    # Check stagnation (clean, non-intrusive)
                    iteration_score = child_program.metrics.get('combined_score', safe_numeric_average(child_program.metrics))
                    was_stagnant_before_check = self.is_stagnant
                    self._check_stagnation(completed_iteration, iteration_score=iteration_score)
                    
                    # Log if stagnation state changed
                    if was_stagnant_before_check != self.is_stagnant:
                        if self.stagnation_logger:
                            self.stagnation_logger.info(
                                f"STAGNATION_STATE_CHANGED | Iteration {completed_iteration} | "
                                f"Was stagnant: {was_stagnant_before_check} → Now stagnant: {self.is_stagnant}"
                            )
                    
                    # Check if paradigms need to be cleared (usage limit)
                    if self.current_paradigms:
                        max_usage = len(self.current_paradigms) * self.paradigms_uses_per_paradigm
                        if self.paradigms_used_count >= max_usage:
                            logger.info(f"Paradigms usage limit reached ({self.paradigms_used_count}/{max_usage}). Clearing paradigms.")
                            if self.stagnation_logger:
                                self.stagnation_logger.info(
                                    f"PARADIGMS_CLEARED | Iteration: {completed_iteration} | "
                                    f"Used {self.paradigms_used_count} times | "
                                    f"Max usage: {max_usage} | "
                                    f"Paradigms cleared: {len(self.current_paradigms)}"
                                )
                            # Log which paradigms were cleared
                            for i, p in enumerate(self.current_paradigms, 1):
                                if self.stagnation_logger:
                                    self.stagnation_logger.info(
                                        f"PARADIGM_CLEARED | Iteration: {completed_iteration} | "
                                        f"Paradigm {i}/{len(self.current_paradigms)} | "
                                        f"Idea: {p.get('idea', 'N/A')[:80]}"
                                    )
                            self.current_paradigms = []
                            self.paradigms_used_count = 0
                            # Reset stagnation state and history for fresh check
                            # After using paradigms for max iterations, reset stagnation to False
                            # This allows fresh stagnation detection to trigger again when needed
                            was_stagnant = self.is_stagnant
                            self.is_stagnant = False
                            self.best_score_history = []
                            logger.info("Paradigms cleared - resetting stagnation state and history for fresh check")
                            if self.stagnation_logger:
                                self.stagnation_logger.info(
                                    f"STAGNATION_HISTORY_RESET | Iteration: {completed_iteration} | "
                                    f"History cleared for fresh stagnation check"
                                )
                                if was_stagnant:
                                    self.stagnation_logger.info(
                                        f"STAGNATION_STATE_CHANGED | Iteration: {completed_iteration} | "
                                        f"Was stagnant: {was_stagnant} → Now stagnant: {self.is_stagnant} | "
                                        f"Reason: Paradigms usage limit reached"
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
                    # Multi-sampling when stagnant (clean, simple)
                    num_samples = 1
                    if self.is_stagnant and self.current_paradigms:
                        num_samples = getattr(self.config, 'stagnation_paradigm_samples', 3)
                    
                    for _ in range(num_samples):
                        if next_iteration < total_iterations and len(island_pending[island_id]) < batch_per_island:
                            # Get paradigm for this iteration (simple sequential use)
                            # Check: only use if still stagnant
                            paradigm_for_iteration = None
                            if self.is_stagnant and self.current_paradigms:
                                # Use next paradigm in list (simple rotation)
                                paradigm_index = self.paradigms_used_count % len(self.current_paradigms)
                                paradigm_for_iteration = self.current_paradigms[paradigm_index]
                                self.paradigms_used_count += 1
                                
                                # Log submission with paradigm (clear and accurate)
                                logger.info(
                                    f"🔄 Submitting iteration {next_iteration} with paradigm "
                                    f"({paradigm_index + 1}/{len(self.current_paradigms)}, "
                                    f"usage {self.paradigms_used_count}/{len(self.current_paradigms) * self.paradigms_uses_per_paradigm}): "
                                    f"{paradigm_for_iteration.get('idea', 'N/A')[:60]}"
                                )
                                if self.stagnation_logger:
                                    self.stagnation_logger.info(
                                        f"PARADIGM_SUBMITTED | Iteration {next_iteration} | "
                                        f"Paradigm {paradigm_index + 1}/{len(self.current_paradigms)} | "
                                        f"Usage: {self.paradigms_used_count}/{len(self.current_paradigms) * self.paradigms_uses_per_paradigm} | "
                                        f"Idea: {paradigm_for_iteration.get('idea', 'N/A')[:80]}"
                                    )
                            else:
                                # Not using paradigm - log why (only if paradigms available but not stagnant)
                                if self.current_paradigms and not self.is_stagnant:
                                    if self.stagnation_logger:
                                        self.stagnation_logger.info(
                                            f"PARADIGM_NOT_USED | Iteration {next_iteration} | "
                                            f"Reason: Not stagnant (stagnant={self.is_stagnant}, paradigms_available={len(self.current_paradigms)})"
                                        )
                            
                            future = self._submit_iteration(next_iteration, island_id, paradigm_for_iteration)
                            if future:
                                pending_futures[next_iteration] = future
                                island_pending[island_id].append(next_iteration)
                                next_iteration += 1
                        else:
                            break
                    break  # Only submit batch per completion to maintain balance

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
        self, iteration: int, island_id: Optional[int] = None, paradigm: Optional[Dict[str, Any]] = None
    ) -> Optional[Future]:
        """Submit an iteration to the process pool, optionally pinned to a specific island"""
        try:
            # Use specified island or current island
            target_island = island_id if island_id is not None else self.database.current_island

            # Use thread-safe sampling that doesn't modify shared state
            # This fixes the race condition from GitHub issue #246
            # Check: if paradigm provided but NOT stagnant, skip using paradigm
            if paradigm and not self.is_stagnant:
                # Stagnation cleared while paradigms were generating - skip using paradigm
                logger.info(f"⚠️ Iteration {iteration}: Stagnation cleared - skipping paradigm (was: {paradigm.get('idea', 'N/A')[:60]})")
                if self.stagnation_logger:
                    self.stagnation_logger.info(
                        f"PARADIGM_SKIPPED | Iteration {iteration} | "
                        f"Reason: Stagnation cleared | "
                        f"Paradigm idea: {paradigm.get('idea', 'N/A')[:80]}"
                    )
                paradigm = None
            
            # When stagnant with paradigms, use best program as parent (more direct)
            if paradigm and self.is_stagnant:
                # Use best program as parent when applying paradigms
                best_program = self.database.get_best_program()
                if best_program:
                    parent = best_program
                    # Still sample inspirations normally
                    _, inspirations, sampling_mode = self.database.sample_from_island(
                        island_id=target_island,
                        num_inspirations=self.config.prompt.num_top_programs
                    )
                    sampling_mode = "exploration"  # Force exploration for stagnation
                else:
                    # Fallback to normal sampling if no best program
                    parent, inspirations, sampling_mode = self.database.sample_from_island(
                        island_id=target_island,
                        num_inspirations=self.config.prompt.num_top_programs
                    )
            else:
                # Normal sampling when not stagnant or no paradigm
                parent, inspirations, sampling_mode = self.database.sample_from_island(
                    island_id=target_island,
                    num_inspirations=self.config.prompt.num_top_programs
                )

            # Get sibling programs for context (previous children of this parent)
            sibling_programs = self.database.get_children(parent.id)

            # Create database snapshot
            db_snapshot = self._create_database_snapshot()
            db_snapshot["sampling_island"] = target_island  # Mark which island this is for
            db_snapshot["iteration"] = iteration  # Add iteration number for logging
            db_snapshot["sibling_program_ids"] = [p.id for p in sibling_programs]  # Sibling context
            db_snapshot["sampling_mode"] = sampling_mode  # For mode-aware prompts
            db_snapshot["sibling_context_limit"] = self.database.sibling_context_limit

            # Simple: use provided paradigm (passed from submission)
            if paradigm:
                # Find index of this paradigm in list
                paradigm_index = 0
                if self.current_paradigms:
                    try:
                        paradigm_index = self.current_paradigms.index(paradigm)
                    except ValueError:
                        paradigm_index = 0
                db_snapshot["paradigm_index"] = paradigm_index
                db_snapshot["current_paradigms"] = [paradigm]  # Only pass the one being used
                # CRITICAL: Ensure is_stagnant is True when using paradigms
                db_snapshot["is_stagnant"] = True
                # Store paradigm info for this iteration (for logging even after paradigms cleared)
                # This ensures we can log PARADIGM_COMPLETED even if paradigms are cleared before completion
                best_program = self.database.get_best_program()
                best_score_before = best_program.metrics.get('combined_score', safe_numeric_average(best_program.metrics)) if best_program else 0.0
                self.iteration_paradigm_map[iteration] = {
                    **paradigm.copy(),
                    'best_score_before': best_score_before  # Track best score before trying this paradigm
                }
            else:
                db_snapshot["paradigm_index"] = 0
                db_snapshot["current_paradigms"] = []

            # Check for iteration-based stagnation (separate from improvement-rate stagnation)
            # This triggers multi-child generation when an island hasn't improved for N iterations
            island_stagnating = self.database.is_island_stagnating(target_island)

            # Use multi-child worker if island is stagnating AND we're not already using paradigms
            # (paradigms use improvement-rate stagnation, multi-child uses iteration-based)
            if island_stagnating and not paradigm:
                num_children = getattr(self.config, 'stagnation_multi_child_count', 3)
                logger.info(
                    f"🔄 Island {target_island} stagnating - using multi-child generation "
                    f"({num_children} children) for iteration {iteration}"
                )
                future = self.executor.submit(
                    _run_stagnation_multi_child_worker,
                    iteration,
                    db_snapshot,
                    parent.id,
                    [insp.id for insp in inspirations],
                    num_children,
                )
            else:
                # Normal single-child iteration
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
