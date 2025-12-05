"""
Prompt sampling for OpenEvolve
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import (
    safe_numeric_average,
    get_fitness_score,
    format_feature_coordinates,
)

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for code evolution"""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.template_manager = TemplateManager(custom_template_dir=config.template_dir)

        # Store custom template mappings
        self.system_template_override = None
        self.user_template_override = None

        # Only log once to reduce duplication
        if not hasattr(logger, "_prompt_sampler_logged"):
            logger.info("Initialized prompt sampler")
            logger._prompt_sampler_logged = True

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """
        Set custom templates to use for this sampler

        Args:
            system_template: Template name for system message
            user_template: Template name for user message
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Set custom templates: system={system_template}, user={user_template}")

    def build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],  # Add inspirations parameter
        language: str = "python",
        evolution_round: int = 0,
        diff_based_evolution: bool = True,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        feature_dimensions: Optional[List[str]] = None,
        sibling_programs: Optional[List[Dict[str, Any]]] = None,  # Previous children of this parent
        parent_fitness: Optional[float] = None,  # Parent's fitness for sibling context
        sampling_mode: Optional[str] = None,  # Sampling mode for mode-aware prompts
        # Paradigm-based stagnation parameters
        is_stagnant: bool = False,
        best_program_code: Optional[str] = None,
        best_score: float = 0.0,
        current_paradigms: Optional[List[Dict[str, Any]]] = None,
        paradigm_index: int = 0,
        error_context: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM

        Args:
            current_program: Current program code
            parent_program: Parent program from which current was derived
            program_metrics: Dictionary of metric names to values
            previous_programs: List of previous program attempts
            top_programs: List of top-performing programs (best by fitness)
            inspirations: List of inspiration programs (diverse/creative examples)
            language: Programming language
            evolution_round: Current evolution round
            diff_based_evolution: Whether to use diff-based evolution (True) or full rewrites (False)
            template_key: Optional override for template key
            program_artifacts: Optional artifacts from program evaluation
            sibling_programs: Optional list of previous children of the same parent
            parent_fitness: Optional parent's fitness score (for sibling context delta calculation)
            sampling_mode: Sampling mode ("exploration", "exploitation", or "weighted") for mode-aware prompts
            **kwargs: Additional keys to replace in the user prompt

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Select template based on evolution mode (with overrides)
        if template_key:
            # Use explicitly provided template key
            user_template_key = template_key
        elif self.user_template_override:
            # Use the override set with set_templates
            user_template_key = self.user_template_override
        else:
            # Default behavior: diff-based vs full rewrite
            user_template_key = "diff_user" if diff_based_evolution else "full_rewrite_user"

        # Get the template
        user_template = self.template_manager.get_template(user_template_key)

        # Use system template override if set
        if self.system_template_override:
            system_message = self.template_manager.get_template(self.system_template_override)
        else:
            system_message = self.config.system_message
            # If system_message is a template name rather than content, get the template
            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)

        # Format metrics
        metrics_str = self._format_metrics(program_metrics)

        # Identify areas for improvement
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs, feature_dimensions
        )

        # Format evolution history
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, inspirations, language, feature_dimensions
        )

        # Format artifacts section if enabled and available
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        # Format sibling context if available
        sibling_context = ""
        if sibling_programs and parent_fitness is not None:
            sibling_context = self._format_sibling_context(
                sibling_programs, parent_fitness, feature_dimensions or []
            )

        # Format selection context based on sampling mode (mode-aware prompts)
        selection_context = self._get_selection_context(sampling_mode, program_metrics)

        # Handle stagnation context if provided via kwargs
        stagnation_context = kwargs.pop("stagnation_context", "")

        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Calculate fitness and feature coordinates for the new template format
        feature_dimensions = feature_dimensions or []
        fitness_score = get_fitness_score(program_metrics, feature_dimensions)
        feature_coords = format_feature_coordinates(program_metrics, feature_dimensions)

        # Build paradigm-based stagnation section if stagnant
        paradigm_stagnation_section = ""
        if is_stagnant:
            # Use LLM-generated paradigms only (no hardcoded fallback)
            if current_paradigms and len(current_paradigms) > 0:
                # Use generated paradigm (rotate through them)
                paradigm = current_paradigms[paradigm_index % len(current_paradigms)]
                paradigm_stagnation_section = f"""
## ⚠️ STAGNATION DETECTED: BREAKTHROUGH IDEA REQUIRED

The best program has stalled at score {best_score:.6f}. You MUST implement this breakthrough idea to improve it:

**BREAKTHROUGH IDEA TO IMPLEMENT:**
{paradigm.get('idea', 'N/A')}

**HOW TO IMPLEMENT IT:**
{paradigm.get('description', 'N/A')}

**WHAT TO OPTIMIZE:**
{paradigm.get('what_to_optimize', 'N/A')}

**IMPORTANT CONSIDERATIONS:**
{paradigm.get('cautions', 'N/A')}

**YOUR TASK:**
Modify the program below to implement the breakthrough idea above. Your goal is to achieve a score better than {best_score:.6f}.

**CRITICAL:**
- You MUST implement the breakthrough idea
- Ensure the paradigm is actually used in your implementation (not just mentioned in comments)
- Correctness is essential - your implementation must be correct and functional
- Verify output format matches evaluator requirements
- Make purposeful changes that implement the idea
- Test your implementation logic carefully

---
"""
            else:
                # No generated paradigms available yet - just log stagnation
                # Will use paradigms once LLM generates them
                paradigm_stagnation_section = f"""
## ⚠️ STAGNATION DETECTED

The current best program has stalled at score {best_score:.6f}. Breakthrough ideas are being generated.

**OBJECTIVE:** Make your best attempt to develop a different approach that achieves a score better than {best_score:.6f}.

**IMPORTANT:**
- Correctness is critical - ensure your implementation is correct and functional
- Make purposeful, well-thought-out changes
- Avoid random or excessive changes - focus on meaningful improvements
- Test your logic carefully to ensure it works correctly

---

## Current Best Program (for reference):
```{language}
{best_program_code if best_program_code else ''}
```

---
"""

        # Add error context section if retrying after error
        error_section = ""
        if error_context:
            error_section = f"""
## ⚠️ RETRY: Previous Attempt Failed

The previous attempt to generate/modify this program failed with an error. Please fix the issue and generate a corrected version.

**Error Details:**
{error_context}

**Your Task:**
- Fix the error in the previous attempt
- Ensure the code is correct and will execute without errors
- Maintain the same approach/goal but correct the implementation
- Verify output format matches evaluator requirements

---

"""

        # Format the final user message
        # Note: sibling_context is passed as empty to template, then appended below
        user_message = error_section + paradigm_stagnation_section + user_template.format(
            metrics=metrics_str,
            fitness_score=f"{fitness_score:.4f}",
            feature_coords=feature_coords,
            feature_dimensions=", ".join(feature_dimensions) if feature_dimensions else "None",
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            sibling_context="",  # Handled separately below
            **kwargs,
        )

        # Append selection context, sibling context, and stagnation context after the main message
        if selection_context:
            user_message = user_message + "\n\n" + selection_context
        if sibling_context:
            user_message = user_message + "\n\n" + sibling_context
        if stagnation_context:
            user_message = user_message + "\n" + stagnation_context

        return {
            "system": system_message,
            "user": user_message,
        }

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for the prompt using safe formatting"""
        # Use safe formatting to handle mixed numeric and string values
        formatted_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                try:
                    formatted_parts.append(f"- {name}: {value:.4f}")
                except (ValueError, TypeError):
                    formatted_parts.append(f"- {name}: {value}")
            else:
                formatted_parts.append(f"- {name}: {value}")
        return "\n".join(formatted_parts)

    def _get_selection_context(self, sampling_mode: Optional[str], metrics: Dict[str, float]) -> str:
        """
        Generate context-aware guidance based on parent selection mode.

        This provides different prompting strategies for exploration vs exploitation:
        - Exploitation: Focus on refining and improving the existing approach
        - Exploration: Encourage trying fundamentally different approaches

        Args:
            sampling_mode: The sampling mode ("exploration", "exploitation", or "weighted")
            metrics: The parent program's metrics

        Returns:
            String with mode-specific guidance to append to the prompt
        """
        if not sampling_mode:
            return ""

        metrics_text = self._format_metrics(metrics)

        if sampling_mode == "exploitation":
            return f"""## PARENT SELECTION CONTEXT
This parent was selected from the archive of top-performing programs. It has demonstrated strong performance, but there is likely still room for significant improvement.

### OPTIMIZATION GUIDANCE
- This solution works well, but don't assume it's optimal - meaningful improvements are still possible
- You may refine the existing approach OR introduce better algorithms if you identify a clear opportunity
- Consider: algorithmic improvements, better data structures, more efficient libraries, parallelization
- Optimizations like vectorization, caching, and numerical stability improvements are valuable
- If you see a fundamentally better approach, pursue it - but ensure correctness is maintained
- Think critically: what assumptions does this solution make? Can they be relaxed or improved?

### PARENT METRICS
{metrics_text}

Your goal: Improve upon this solution - whether through refinement or strategic redesign.
"""
        else:  # exploration or weighted
            return f"""## PARENT SELECTION CONTEXT
This parent was selected through diversity-driven sampling to explore different
regions of the solution space. It may or may not represent optimal performance.

### EXPLORATION GUIDANCE
- Consider alternative algorithmic approaches or techniques
- Experiment with different methods, or approaches
- Don't be constrained by the parent's approach - it's a starting point, not a template
- Look for opportunities to try fundamentally different algorithms or novel techniques
- Balance creativity with correctness - new ideas should still produce valid solutions

### PARENT METRICS
{metrics_text}

Your goal: Discover new approaches that might outperform current solutions.
"""

    def _format_sibling_context(
        self,
        sibling_programs: List[Dict[str, Any]],
        parent_fitness: float,
        feature_dimensions: List[str],
    ) -> str:
        """
        Format context about previous mutations of the same parent.

        Shows the actual code that was tried so the LLM can avoid repeating
        the same approaches.

        Args:
            sibling_programs: List of previous children of this parent
            parent_fitness: Parent's fitness score for delta calculation
            feature_dimensions: Feature dimensions to exclude from fitness

        Returns:
            Formatted string with sibling context, or empty string if no siblings
        """
        if not sibling_programs:
            return ""

        # Track statistics
        improved_count = 0
        regressed_count = 0
        unchanged_count = 0

        sibling_entries = []

        for i, sibling in enumerate(sibling_programs, 1):
            sibling_metrics = sibling.get("metrics", {})
            sibling_fitness = get_fitness_score(sibling_metrics, feature_dimensions)
            delta = sibling_fitness - parent_fitness

            # Track statistics
            if delta > 0.001:
                improved_count += 1
                status = "IMPROVED"
            elif delta < -0.001:
                regressed_count += 1
                status = "REGRESSED"
            else:
                unchanged_count += 1
                status = "NO IMPROVEMENT"

            # Extract the evolved code block from the child
            child_code = sibling.get("code", "")
            evolved_block = self._extract_evolve_block(child_code)

            # Use evolved block if available, otherwise use truncated full code
            if evolved_block:
                code_to_show = evolved_block.strip()
            else:
                code_to_show = child_code[:800] + "..." if len(child_code) > 800 else child_code

            # Truncate if still too long (max ~40 lines)
            code_lines = code_to_show.split('\n')
            if len(code_lines) > 40:
                code_to_show = '\n'.join(code_lines[:40]) + '\n... (truncated)'

            # Build entry
            entry = f"### Attempt {i}: {parent_fitness:.4f} → {sibling_fitness:.4f} ({delta:+.4f}) [{status}]\n"
            entry += f"```python\n{code_to_show}\n```"
            sibling_entries.append(entry)

        # Build the full context
        lines = [
            "## PREVIOUS ATTEMPTS ON THIS PARENT - DO NOT REPEAT THESE",
            f"**{len(sibling_programs)} previous attempt(s)**: {improved_count} improved, {unchanged_count} no change, {regressed_count} regressed",
            "",
            "The code below was ALREADY TRIED. Try something DIFFERENT.",
            "",
        ]

        # Add the actual code entries
        lines.extend(sibling_entries)
        lines.append("")

        # Add guidance based on patterns
        if unchanged_count == len(sibling_programs):
            lines.append("**All attempts failed to improve.** You MUST try a fundamentally different approach.")
        elif improved_count > 0:
            lines.append("**Some attempts improved.** Analyze what worked and build on it.")

        return "\n".join(lines)

    def _extract_evolve_block(self, code: str) -> str:
        """
        Extract the EVOLVE-BLOCK from code if present.

        Args:
            code: Full program code

        Returns:
            The content between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END, or empty string
        """
        start_marker = "# EVOLVE-BLOCK-START"
        end_marker = "# EVOLVE-BLOCK-END"

        start_idx = code.find(start_marker)
        end_idx = code.find(end_marker)

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return code[start_idx:end_idx + len(end_marker)]

        return ""

    def _identify_improvement_areas(
        self,
        current_program: str,
        parent_program: str,
        metrics: Dict[str, float],
        previous_programs: List[Dict[str, Any]],
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """Identify improvement areas with proper fitness/feature separation"""

        improvement_areas = []
        feature_dimensions = feature_dimensions or []

        # Calculate fitness (excluding feature dimensions)
        current_fitness = get_fitness_score(metrics, feature_dimensions)

        # Track fitness changes (not individual metrics)
        if previous_programs:
            prev_metrics = previous_programs[-1].get("metrics", {})
            prev_fitness = get_fitness_score(prev_metrics, feature_dimensions)

            if current_fitness > prev_fitness:
                msg = self.template_manager.get_fragment(
                    "fitness_improved", prev=prev_fitness, current=current_fitness
                )
                improvement_areas.append(msg)
            elif current_fitness < prev_fitness:
                msg = self.template_manager.get_fragment(
                    "fitness_declined", prev=prev_fitness, current=current_fitness
                )
                improvement_areas.append(msg)
            elif abs(current_fitness - prev_fitness) < 1e-6:  # Essentially unchanged
                msg = self.template_manager.get_fragment("fitness_stable", current=current_fitness)
                improvement_areas.append(msg)

        # Note feature exploration (not good/bad, just informational)
        if feature_dimensions:
            feature_coords = format_feature_coordinates(metrics, feature_dimensions)
            if feature_coords != "No feature coordinates":
                msg = self.template_manager.get_fragment(
                    "exploring_region", features=feature_coords
                )
                improvement_areas.append(msg)

        # Code length check (configurable threshold)
        threshold = (
            self.config.suggest_simplification_after_chars or self.config.code_length_threshold
        )
        if threshold and len(current_program) > threshold:
            msg = self.template_manager.get_fragment("code_too_long", threshold=threshold)
            improvement_areas.append(msg)

        # Default guidance if nothing specific
        if not improvement_areas:
            improvement_areas.append(self.template_manager.get_fragment("no_specific_guidance"))

        return "\n".join(f"- {area}" for area in improvement_areas)

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")

        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("metadata", {}).get("changes", "Unknown changes")

            # Format performance metrics using safe formatting
            performance_parts = []
            for name, value in program.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            # Determine outcome based on comparison with parent (only numeric metrics)
            parent_metrics = program.get("metadata", {}).get("parent_metrics", {})
            outcome = "Mixed results"

            # Safely compare only numeric metrics
            program_metrics = program.get("metrics", {})

            # Check if all numeric metrics improved
            numeric_comparisons_improved = []
            numeric_comparisons_regressed = []

            for m in program_metrics:
                prog_value = program_metrics.get(m, 0)
                parent_value = parent_metrics.get(m, 0)

                # Only compare if both values are numeric
                if isinstance(prog_value, (int, float)) and isinstance(parent_value, (int, float)):
                    if prog_value > parent_value:
                        numeric_comparisons_improved.append(True)
                    else:
                        numeric_comparisons_improved.append(False)

                    if prog_value < parent_value:
                        numeric_comparisons_regressed.append(True)
                    else:
                        numeric_comparisons_regressed.append(False)

            # Determine outcome based on numeric comparisons
            if numeric_comparisons_improved and all(numeric_comparisons_improved):
                outcome = "Improvement in all metrics"
            elif numeric_comparisons_regressed and all(numeric_comparisons_regressed):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]

        for i, program in enumerate(selected_top):
            # Use the full program code
            program_code = program.get("code", "")

            # Calculate fitness score (prefers combined_score, excludes feature dimensions)
            score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

            # Extract key features (this could be more sophisticated)
            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        try:
                            key_features.append(f"Performs well on {name} ({value:.4f})")
                        except (ValueError, TypeError):
                            key_features.append(f"Performs well on {name} ({value})")
                    else:
                        key_features.append(f"Performs well on {name} ({value})")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_code,
                    key_features=key_features_str,
                )
                + "\n\n"
            )

        # Format diverse programs using num_diverse_programs config
        diverse_programs_str = ""
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):
            # Skip the top programs we already included
            remaining_programs = top_programs[self.config.num_top_programs :]

            # Sample diverse programs from the remaining
            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            if num_diverse > 0:
                # Use random sampling to get diverse programs
                diverse_programs = random.sample(remaining_programs, num_diverse)

                diverse_programs_str += "\n\n## Diverse Programs\n\n"

                for i, program in enumerate(diverse_programs):
                    # Use the full program code
                    program_code = program.get("code", "")

                    # Calculate fitness score (prefers combined_score, excludes feature dimensions)
                    score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

                    # Extract key features
                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            f"Alternative approach to {name}"
                            for name in list(program.get("metrics", {}).keys())[
                                :2
                            ]  # Just first 2 metrics
                        ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=language,
                            program_snippet=program_code,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )

        # Combine top and diverse programs
        combined_programs_str = top_programs_str + diverse_programs_str

        # Format inspirations section
        inspirations_section_str = self._format_inspirations_section(
            inspirations, language, feature_dimensions
        )

        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
            inspirations_section=inspirations_section_str,
        )

    def _format_inspirations_section(
        self,
        inspirations: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """
        Format the inspirations section for the prompt

        Args:
            inspirations: List of inspiration programs
            language: Programming language

        Returns:
            Formatted inspirations section string
        """
        if not inspirations:
            return ""

        # Get templates
        inspirations_section_template = self.template_manager.get_template("inspirations_section")
        inspiration_program_template = self.template_manager.get_template("inspiration_program")

        inspiration_programs_str = ""

        for i, program in enumerate(inspirations):
            # Use the full program code
            program_code = program.get("code", "")

            # Calculate fitness score (prefers combined_score, excludes feature dimensions)
            score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

            # Determine program type based on metadata and score
            program_type = self._determine_program_type(program, feature_dimensions or [])

            # Extract unique features (emphasizing diversity rather than just performance)
            unique_features = self._extract_unique_features(program)

            inspiration_programs_str += (
                inspiration_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    program_type=program_type,
                    language=language,
                    program_snippet=program_code,
                    unique_features=unique_features,
                )
                + "\n\n"
            )

        return inspirations_section_template.format(
            inspiration_programs=inspiration_programs_str.strip()
        )

    def _determine_program_type(
        self, program: Dict[str, Any], feature_dimensions: Optional[List[str]] = None
    ) -> str:
        """
        Determine the type/category of an inspiration program

        Args:
            program: Program dictionary

        Returns:
            String describing the program type
        """
        metadata = program.get("metadata", {})
        score = get_fitness_score(program.get("metrics", {}), feature_dimensions or [])

        # Check metadata for explicit type markers
        if metadata.get("diverse", False):
            return "Diverse"
        if metadata.get("migrant", False):
            return "Migrant"
        if metadata.get("random", False):
            return "Random"

        # Classify based on score ranges
        if score >= 0.8:
            return "High-Performer"
        elif score >= 0.6:
            return "Alternative"
        elif score >= 0.4:
            return "Experimental"
        else:
            return "Exploratory"

    def _extract_unique_features(self, program: Dict[str, Any]) -> str:
        """
        Extract unique features of an inspiration program

        Args:
            program: Program dictionary

        Returns:
            String describing unique aspects of the program
        """
        features = []

        # Extract from metadata if available
        metadata = program.get("metadata", {})
        if "changes" in metadata:
            changes = metadata["changes"]
            if (
                isinstance(changes, str)
                and self.config.include_changes_under_chars
                and len(changes) < self.config.include_changes_under_chars
            ):
                features.append(f"Modification: {changes}")

        # Analyze metrics for standout characteristics
        metrics = program.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    features.append(f"Excellent {metric_name} ({value:.3f})")
                elif value <= 0.3:
                    features.append(f"Alternative {metric_name} approach")

        # Code-based features (simple heuristics)
        code = program.get("code", "")
        if code:
            code_lower = code.lower()
            if "class" in code_lower and "def __init__" in code_lower:
                features.append("Object-oriented approach")
            if "numpy" in code_lower or "np." in code_lower:
                features.append("NumPy-based implementation")
            if "for" in code_lower and "while" in code_lower:
                features.append("Mixed iteration strategies")
            if (
                self.config.concise_implementation_max_lines
                and len(code.split("\n")) <= self.config.concise_implementation_max_lines
            ):
                features.append("Concise implementation")
            elif (
                self.config.comprehensive_implementation_min_lines
                and len(code.split("\n")) >= self.config.comprehensive_implementation_min_lines
            ):
                features.append("Comprehensive implementation")

        # Default if no specific features found
        if not features:
            program_type = self._determine_program_type(program)
            features.append(f"{program_type} approach to the problem")

        # Use num_top_programs as limit for features (similar to how we limit programs)
        feature_limit = self.config.num_top_programs
        return ", ".join(features[:feature_limit])

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template

        # Apply variations defined in the config
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)

        return result

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        """
        Render artifacts for prompt inclusion

        Args:
            artifacts: Dictionary of artifact name to content

        Returns:
            Formatted string for prompt inclusion (empty string if no artifacts)
        """
        if not artifacts:
            return ""

        sections = []

        # Process all artifacts using .items()
        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            # Truncate if too long
            if len(content) > self.config.max_artifact_bytes:
                content = content[: self.config.max_artifact_bytes] + "\n... (truncated)"

            sections.append(f"### {key}\n```\n{content}\n```")

        if sections:
            return "## Last Execution Output\n\n" + "\n\n".join(sections)
        else:
            return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """
        Safely decode an artifact value to string

        Args:
            value: Artifact value (string or bytes)

        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            # Apply security filter if enabled
            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        else:
            return str(value)

    def _apply_security_filter(self, text: str) -> str:
        """
        Apply security filtering to artifact text

        Args:
            text: Input text

        Returns:
            Filtered text with potential secrets/sensitive info removed
        """
        import re

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)

        # Basic patterns for common secrets (can be expanded)
        secret_patterns = [
            (r"[A-Za-z0-9]{32,}", "<REDACTED_TOKEN>"),  # Long alphanumeric tokens
            (r"sk-[A-Za-z0-9]{48}", "<REDACTED_API_KEY>"),  # OpenAI-style API keys
            (r"password[=:]\s*[^\s]+", "password=<REDACTED>"),  # Password assignments
            (r"token[=:]\s*[^\s]+", "token=<REDACTED>"),  # Token assignments
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
