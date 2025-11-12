"""
Prompt sampling for OpenEvolve
"""

import logging
import random
import json
import os
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
            **kwargs: Additional keys to replace in the user prompt

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Lightweight structured logging of inputs for debugging/inspection
        summary = {
            "evolution_round": evolution_round,
            "language": language,
            "diff_based_evolution": diff_based_evolution,
            "feature_dimensions": list(feature_dimensions or []),
            "program_metrics_keys": list(program_metrics.keys()) if isinstance(program_metrics, dict) else [],
            "previous_programs_count": len(previous_programs or []),
            "top_programs_count": len(top_programs or []),
            "inspirations_count": len(inspirations or []),
            "artifacts_keys": list((program_artifacts or {}).keys()) if isinstance(program_artifacts, dict) else [],
            "current_program_chars": len(current_program or ""),
            "parent_program_chars": len(parent_program or ""),
            "previous_ids": [p.get("id") for p in (previous_programs or []) if isinstance(p, dict) and p.get("id")],
            "top_ids": [p.get("id") for p in (top_programs or []) if isinstance(p, dict) and p.get("id")],
            "inspiration_ids": [p.get("id") for p in (inspirations or []) if isinstance(p, dict) and p.get("id")],
        }
        # Hardcoded JSONL append for prompt input summaries
        try:
            _log_path = "/Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl"
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps({"type": "inputs", **summary}, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("PromptSampler.build_prompt: failed to append inputs to prompt_builder_logs.txt", exc_info=True)
        # Select template based on evolution mode (with overrides)
        # import pdb; pdb.set_trace()
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

        # Append memory-based similar parent changes if available
        similar_parent_changes = kwargs.get("similar_parent_changes", [])
        similar_section = self._format_similar_parent_changes(
            similar_parent_changes or [], language, feature_dimensions
        )
        if similar_section:
            evolution_history = evolution_history + "\n\n" + similar_section

        # Format artifacts section if enabled and available
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Calculate fitness and feature coordinates for the new template format
        feature_dimensions = feature_dimensions or []
        fitness_score = get_fitness_score(program_metrics, feature_dimensions)
        feature_coords = format_feature_coordinates(program_metrics, feature_dimensions)

        # Log intermediate computed context (details)
        try:
            _log_path = "/Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl"
            details = {
                "type": "details",
                "evolution_round": evolution_round,
                "template_key": user_template_key,
                "fitness_score": fitness_score,
                "feature_coords": feature_coords,
                "metrics_preview": metrics_str.split("\n")[:8],
                "improvement_areas_preview": (improvement_areas.split("\n") if isinstance(improvement_areas, str) else []),
                "previous_programs_count": len(previous_programs or []),
                "top_programs_count": len(top_programs or []),
                "inspirations_count": len(inspirations or []),
                "artifacts_present": bool(program_artifacts),
            }
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(details, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("PromptSampler.build_prompt: failed to append details to prompt_builder_logs.txt", exc_info=True)

        # Format the final user message
        user_message = user_template.format(
            metrics=metrics_str,
            fitness_score=f"{fitness_score:.4f}",
            feature_coords=feature_coords,
            feature_dimensions=", ".join(feature_dimensions) if feature_dimensions else "None",
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            **kwargs,
        )

        # Hardcoded JSONL append for final prompt text
        try:
            _log_path = "/Users/cusgadmin/Documents/AutoEvolve_Math_MAS/autoevolve/prompt_builder_logs.jsonl"
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(
                    json.dumps(
                        {
                            "type": "prompt",
                            "evolution_round": evolution_round,
                            "language": language,
                            "diff_based_evolution": diff_based_evolution,
                            "system": system_message,
                            "user": user_message,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            logger.debug("PromptSampler.build_prompt: failed to append prompt to prompt_builder_logs.txt", exc_info=True)

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

    def _format_similar_parent_changes(
        self,
        similar_parent_changes: List[Dict[str, Any]],
        language: str,
        feature_dimensions: Optional[List[str]] = None,
    ) -> str:
        """Format similar-parent best/worst changes section without truncation.

        Selection is bounded by counts only (no code truncation), mirroring top/diverse behavior.
        Counts are controlled via PromptConfig:
          - num_similar_parent_best (default 3)
          - num_similar_parent_worst (default 3)
          - include_similar_parent_worst (default true)
        """
        if not similar_parent_changes:
            return ""

        # Use PromptConfig knobs
        topn = max(0, int(self.config.num_similar_parent_best))
        worstn = max(0, int(self.config.num_similar_parent_worst))
        include_worst = bool(self.config.include_similar_parent_worst)

        # Separate with available deltas
        with_deltas = [c for c in similar_parent_changes if isinstance(c.get("delta_combined_score"), (int, float))]
        if not with_deltas:
            return ""

        # Best (positive delta) and worst (negative delta)
        best = [c for c in with_deltas if c["delta_combined_score"] > 0]
        worst = [c for c in with_deltas if c["delta_combined_score"] < 0]
        best.sort(key=lambda x: x["delta_combined_score"], reverse=True)
        worst.sort(key=lambda x: x["delta_combined_score"])  # most negative first
        best = best[: min(topn, len(best))]
        worst = worst[: min(worstn, len(worst))] if include_worst else []

        if not best and not worst:
            return ""

        lines: List[str] = []
        lines.append("## Similar Parents: Prior Changes from Similar Starting Points")
        lines.append("")
        lines.append(
            "Below are examples of what happened when we evolved programs that STARTED from similar code "
            "to your current program. These examples show similar PROBLEM STRUCTURES, not solutions to copy."
        )
        lines.append("")
        lines.append("**GRADIENT INTERPRETATION:** Examples are ranked by their **gradient** = improvement / code distance. "
                     "High gradient means large improvement from small, focused changes (very efficient). "
                     "Low gradient means small improvement from large changes (inefficient). "
                     "Focus on learning from high-gradient examples.")
        lines.append("")
        lines.append("**INTENTION:** Use these examples to:")
        lines.append("- **LEARN STRATEGIC PATTERNS** from best examples (e.g., 'hexagonal layouts worked well', "
                     "'iterative refinement helped') - extract the STRATEGY, not the specific code")
        lines.append("- **AVOID FAILURE MODES** from worst examples (e.g., 'over-constraining led to poor results') "
                     "- understand WHY they failed, not what the code was")
        lines.append("- **INNOVATE BEYOND** these examples - your goal is to CREATE A BETTER SOLUTION that "
                     "OUTPERFORMS these examples, not to match or copy them")
        lines.append("")
        lines.append("**⚠️ CRITICAL PITFALLS TO AVOID:**")
        lines.append("- ❌ DON'T copy these solutions exactly - similarity means starting point, not solution")
        lines.append("- ❌ DON'T converge to local patterns shown here - explore NEW directions")
        lines.append("- ❌ DON'T treat best examples as recipes - use them to understand WHAT worked, not HOW to replicate")
        lines.append("- ❌ DON'T study worst examples in detail - use them to know WHAT to avoid, not as curiosities")
        lines.append("- ❌ DON'T match these examples - your solution should BEAT them, achieving superior performance")
        lines.append("")
        lines.append("**HOW TO USE THESE EXAMPLES:**")
        lines.append("- **Best examples:** Extract HIGH-LEVEL STRATEGIES (patterns, approaches, principles) "
                     "that led to improvements. Focus on WHY they worked, not WHAT the code was.")
        lines.append("- **Worst examples:** Identify the FAILURE MODES and ensure you avoid similar mistakes. "
                     "Understand WHAT went wrong at a strategic level.")
        lines.append("- **Your task:** Create a UNIQUE, INNOVATIVE solution that builds on successful patterns, "
                     "avoids failure modes, and SURPASSES all these examples in performance.")
        lines.append("")
        lines.append("Remember: These examples show similar STARTING POINTS, not templates to follow. "
                     "Your solution should be BETTER than all of them.")
        lines.append("")

        def fmt_item(idx: int, rec: Dict[str, Any]) -> str:
            pid = rec.get("parent_id") or rec.get("parent")
            cid = rec.get("child_id") or rec.get("child")
            p_comb = rec.get("parent_combined_score")
            c_comb = rec.get("child_combined_score")
            delta = rec.get("delta_combined_score")
            p_code = rec.get("parent_code") or ""
            c_code = rec.get("child_code") or ""
            change_summary = rec.get("change_summary")
            parent_metrics = rec.get("parent_metrics") if isinstance(rec.get("parent_metrics"), dict) else {}
            child_metrics = rec.get("child_metrics") if isinstance(rec.get("child_metrics"), dict) else {}

            # Extract gradient and distance for gradient-based evolution
            gradient = rec.get("gradient")
            distance = rec.get("distance")

            # Build a concise header; include IDs if available
            header = f"### Example {idx}: Δ={delta:+.4f} (parent={p_comb} → child={c_comb})"

            # Add gradient information if available
            if gradient is not None:
                # Determine efficiency label based on gradient magnitude
                if abs(gradient) > 1.0:
                    efficiency = "very efficient"
                elif abs(gradient) > 0.3:
                    efficiency = "moderately efficient"
                else:
                    efficiency = "inefficient"

                header += f" | Gradient={gradient:+.3f} ({efficiency})"
                if distance is not None:
                    header += f" | Distance={distance:.3f}"

            if pid or cid:
                header += f" | parent={pid} → child={cid}"

            # Optional change summary
            summary_block = ""
            if isinstance(change_summary, str) and change_summary.strip():
                summary_block = f"\nChanges: {change_summary.strip()}\n"

            # Optional metrics blocks
            metrics_block = ""
            try:
                p_metrics_str = self._format_metrics(parent_metrics) if parent_metrics else ""
                c_metrics_str = self._format_metrics(child_metrics) if child_metrics else ""
                if p_metrics_str or c_metrics_str:
                    metrics_parts = []
                    if p_metrics_str:
                        metrics_parts.append(f"Parent metrics:\n{p_metrics_str}")
                    if c_metrics_str:
                        metrics_parts.append(f"Child metrics:\n{c_metrics_str}")
                    metrics_block = "\n" + "\n\n".join(metrics_parts) + "\n"
            except Exception:
                metrics_block = ""

            body = (
                (summary_block + metrics_block if (summary_block or metrics_block) else "")
                + f"```{language}\n# Parent\n{p_code}\n```\n\n"
                + f"```{language}\n# Child\n{c_code}\n```"
            )
            return header + "\n" + body

        if best:
            lines.append("\n### Best prior changes (positive delta)")
            for i, rec in enumerate(best, 1):
                try:
                    lines.append(fmt_item(i, rec))
                except Exception:
                    continue

        if worst:
            lines.append("\n### Regressions to avoid (negative delta)")
            for i, rec in enumerate(worst, 1):
                try:
                    lines.append(fmt_item(i, rec))
                except Exception:
                    continue

        return "\n\n".join(lines)

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
