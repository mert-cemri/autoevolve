import asyncio
import os
import uuid
import logging
import time
from dataclasses import dataclass

from openevolve.database import Program, ProgramDatabase
from openevolve.config import Config
from openevolve.evaluator import Evaluator
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.prompt.sampler import PromptSampler
from openevolve.utils.code_utils import (
    apply_diff,
    extract_diffs,
    format_diff_summary,
    parse_full_rewrite,
)


@dataclass
class Result:
    """Resulting program and metrics from an iteration of OpenEvolve"""

    child_program: str = None
    parent: str = None
    child_metrics: str = None
    iteration_time: float = None
    prompt: str = None
    llm_response: str = None
    artifacts: dict = None
    iteration: int = 0


@dataclass
class MultiChildResult:
    """Resulting programs from multi-child generation on stagnation"""

    children: list = None  # List of Result objects
    parent: str = None
    iteration_time: float = None
    iteration: int = 0


async def run_iteration_with_shared_db(
    iteration: int,
    config: Config,
    database: ProgramDatabase,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
):
    """
    Run a single iteration using shared memory database

    This is optimized for use with persistent worker processes.
    """
    logger = logging.getLogger(__name__)

    try:
        # Sample parent and inspirations from database (with sampling mode for mode-aware prompts)
        parent, inspirations, sampling_mode = database.sample_from_island(
            island_id=database.current_island,
            num_inspirations=config.prompt.num_top_programs
        )

        # Get artifacts for the parent program if available
        parent_artifacts = database.get_artifacts(parent.id)

        # Get island-specific top programs for prompt context (maintain island isolation)
        parent_island = parent.metadata.get("island", database.current_island)
        island_top_programs = database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = database.get_top_programs(3, island_idx=parent_island)

        # Get sibling programs (previous children of this parent) for context
        sibling_programs = database.get_children(parent.id)
        sibling_programs_dicts = [p.to_dict() for p in sibling_programs]

        # Calculate parent fitness for sibling context
        from openevolve.utils.metrics_utils import get_fitness_score
        parent_fitness = get_fitness_score(parent.metrics, database.config.feature_dimensions)

        # Build prompt with sampling_mode for mode-aware prompts
        prompt = prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in island_previous_programs],
            top_programs=[p.to_dict() for p in island_top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=config.language,
            evolution_round=iteration,
            diff_based_evolution=config.diff_based_evolution,
            program_artifacts=parent_artifacts if parent_artifacts else None,
            feature_dimensions=database.config.feature_dimensions,
            sibling_programs=sibling_programs_dicts,
            parent_fitness=parent_fitness,
            sampling_mode=sampling_mode,  # Pass sampling mode for different prompts
        )

        result = Result(parent=parent)
        iteration_start = time.time()

        # Generate code modification
        llm_response = await llm_ensemble.generate_with_context(
            system_message=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )

        # Parse the response
        if config.diff_based_evolution:
            diff_blocks = extract_diffs(llm_response)

            if not diff_blocks:
                logger.warning(f"Iteration {iteration+1}: No valid diffs found in response")
                return None

            # Apply the diffs
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            # Parse full rewrite
            new_code = parse_full_rewrite(llm_response, config.language)

            if not new_code:
                logger.warning(f"Iteration {iteration+1}: No valid code found in response")
                return None

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > config.max_code_length:
            logger.warning(
                f"Iteration {iteration+1}: Generated code exceeds maximum length "
                f"({len(child_code)} > {config.max_code_length})"
            )
            return None

        # Evaluate the child program
        child_id = str(uuid.uuid4())
        result.child_metrics = await evaluator.evaluate_program(child_code, child_id)

        # Handle artifacts if they exist
        artifacts = evaluator.get_pending_artifacts(child_id)

        # Set template_key of Prompts
        template_key = (
            "full_rewrite_user" if not config.diff_based_evolution else "diff_user"
        )

        # Create a child program
        result.child_program = Program(
            id=child_id,
            code=child_code,
            language=config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=result.child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
            },
            prompts={
                template_key: {
                    "system": prompt["system"],
                    "user": prompt["user"],
                    "responses": [llm_response] if llm_response is not None else [],
                }
            } if database.config.log_prompts else None,
        )

        result.prompt = prompt
        result.llm_response = llm_response
        result.artifacts = artifacts
        result.iteration_time = time.time() - iteration_start
        result.iteration = iteration

        return result

    except Exception as e:
        logger.exception(f"Error in iteration {iteration}: {e}")
        return None


async def run_iteration_multi_child_stagnation(
    iteration: int,
    config: Config,
    database: ProgramDatabase,
    evaluator: Evaluator,
    llm_ensemble: LLMEnsemble,
    prompt_sampler: PromptSampler,
    num_children: int = 3,
):
    """
    Run a single iteration with multi-child generation for breaking stagnation.

    When an island is stagnating (no improvement for stagnation_threshold iterations),
    we generate multiple diverse children from the same parent to try different approaches.

    Args:
        iteration: Current iteration number
        config: Configuration object
        database: Program database
        evaluator: Program evaluator
        llm_ensemble: LLM ensemble for code generation
        prompt_sampler: Prompt sampler for building prompts
        num_children: Number of children to generate (default: 3)

    Returns:
        MultiChildResult with all generated children
    """
    logger = logging.getLogger(__name__)
    iteration_start = time.time()

    try:
        # Sample parent and inspirations from database
        # For stagnation breaking, we always use exploration mode
        parent, inspirations, _ = database.sample_from_island(
            island_id=database.current_island,
            num_inspirations=config.prompt.num_top_programs
        )
        sampling_mode = "exploration"  # Force exploration mode for stagnation breaking

        logger.info(
            f"Iteration {iteration+1}: STAGNATION DETECTED - Generating {num_children} diverse children"
        )

        # Get parent artifacts and island context
        parent_artifacts = database.get_artifacts(parent.id)
        parent_island = parent.metadata.get("island", database.current_island)
        island_top_programs = database.get_top_programs(5, island_idx=parent_island)
        island_previous_programs = database.get_top_programs(3, island_idx=parent_island)

        # Get sibling programs for context
        sibling_programs = database.get_children(parent.id)
        sibling_programs_dicts = [p.to_dict() for p in sibling_programs]

        # Calculate parent fitness for sibling context
        from openevolve.utils.metrics_utils import get_fitness_score
        parent_fitness = get_fitness_score(parent.metrics, database.config.feature_dimensions)

        # Generate children sequentially with sibling awareness
        children_results = []
        generated_siblings_this_iteration = []  # Track siblings generated in this iteration

        for child_idx in range(num_children):
            logger.debug(f"Generating stagnation-breaking child {child_idx + 1}/{num_children}")

            # Build prompt with sibling context (both historical and current iteration siblings)
            all_siblings = sibling_programs_dicts + generated_siblings_this_iteration

            # Add stagnation-aware context to the prompt
            stagnation_context = (
                f"\n\n## STAGNATION ALERT\n"
                f"This parent's island has not improved for {database.get_island_stagnation(parent_island)} iterations. "
                f"Previous refinement approaches haven't broken through. "
                f"Consider trying a significantly different approach or algorithm.\n"
                f"This is child {child_idx + 1} of {num_children} being generated to break stagnation."
            )

            prompt = prompt_sampler.build_prompt(
                current_program=parent.code,
                parent_program=parent.code,
                program_metrics=parent.metrics,
                previous_programs=[p.to_dict() for p in island_previous_programs],
                top_programs=[p.to_dict() for p in island_top_programs],
                inspirations=[p.to_dict() for p in inspirations],
                language=config.language,
                evolution_round=iteration,
                diff_based_evolution=config.diff_based_evolution,
                program_artifacts=parent_artifacts if parent_artifacts else None,
                feature_dimensions=database.config.feature_dimensions,
                sibling_programs=all_siblings[-database.sibling_context_limit:],  # Limit siblings
                parent_fitness=parent_fitness,
                sampling_mode=sampling_mode,  # Use exploration mode for stagnation breaking
                stagnation_context=stagnation_context,  # Add stagnation context
            )

            # Generate code modification
            llm_response = await llm_ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )

            # Parse the response
            if config.diff_based_evolution:
                diff_blocks = extract_diffs(llm_response)

                if not diff_blocks:
                    logger.warning(f"Child {child_idx + 1}: No valid diffs found in response")
                    continue

                child_code = apply_diff(parent.code, llm_response)
                changes_summary = format_diff_summary(diff_blocks)
            else:
                new_code = parse_full_rewrite(llm_response, config.language)

                if not new_code:
                    logger.warning(f"Child {child_idx + 1}: No valid code found in response")
                    continue

                child_code = new_code
                changes_summary = "Full rewrite"

            # Check code length
            if len(child_code) > config.max_code_length:
                logger.warning(
                    f"Child {child_idx + 1}: Generated code exceeds maximum length "
                    f"({len(child_code)} > {config.max_code_length})"
                )
                continue

            # Evaluate the child program
            child_id = str(uuid.uuid4())
            child_metrics = await evaluator.evaluate_program(child_code, child_id)

            # Handle artifacts
            artifacts = evaluator.get_pending_artifacts(child_id)

            # Set template_key
            template_key = (
                "full_rewrite_user" if not config.diff_based_evolution else "diff_user"
            )

            # Create child program
            child_program = Program(
                id=child_id,
                code=child_code,
                language=config.language,
                parent_id=parent.id,
                generation=parent.generation + 1,
                metrics=child_metrics,
                iteration_found=iteration,
                metadata={
                    "changes": changes_summary,
                    "parent_metrics": parent.metrics,
                    "child_index": child_idx,
                    "stagnation_child": True,  # Mark as stagnation-breaking child
                    "island": parent_island,
                },
                prompts={
                    template_key: {
                        "system": prompt["system"],
                        "user": prompt["user"],
                        "responses": [llm_response] if llm_response is not None else [],
                    }
                } if database.config.log_prompts else None,
            )

            # Create result for this child
            result = Result(
                child_program=child_program,
                parent=parent,
                child_metrics=child_metrics,
                prompt=prompt,
                llm_response=llm_response,
                artifacts=artifacts,
                iteration=iteration,
            )

            children_results.append(result)

            # Add to current iteration siblings for diversity
            generated_siblings_this_iteration.append({
                "metrics": child_metrics,
                "metadata": {"changes": changes_summary},
                "iteration_found": iteration,
            })

        logger.info(
            f"Iteration {iteration+1}: Generated {len(children_results)} stagnation-breaking children"
        )

        # Return multi-child result
        return MultiChildResult(
            children=children_results,
            parent=parent,
            iteration_time=time.time() - iteration_start,
            iteration=iteration,
        )

    except Exception as e:
        logger.exception(f"Error in multi-child stagnation iteration {iteration}: {e}")
        return None
