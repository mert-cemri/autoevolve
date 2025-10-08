"""
Best of N Search Strategy

Generates N independent lineages from the initial program and evolves each
linearly for T iterations. Returns the best program across all lineages.
"""

import logging
import os
from typing import List

from .base_search import BaseSearch, Program

logger = logging.getLogger(__name__)


class BestOfNSearch(BaseSearch):
    """
    Best of N Search Strategy

    Algorithm:
    1. Generate N variants from initial program
    2. For T iterations:
       - For each variant, generate improved version
       - Evaluate and update if better
    3. Return best program across all lineages
    """

    def search(self, n: int = 4, iterations: int = 10) -> Program:
        """
        Run Best of N search

        Args:
            n: Number of parallel lineages
            iterations: Number of iterations per lineage

        Returns:
            Best program found
        """
        logger.info(
            f"Starting Best of N search with n={n}, iterations={iterations}, "
            f"num_eval_problems={self.num_eval_problems}"
        )
        logger.info(
            f"Evolution model: {self.model}, "
            f"Agent model: {os.environ.get('OPENEVOLVE_MODEL', 'unknown')}"
        )

        # Initialize N lineages from initial program
        lineages: List[Program] = []
        initial = Program(self.initial_program, parent_id=None, generation=0)
        self.evaluate_program(initial)

        # Track best program globally
        global_best = initial
        iteration_bests = []  # Track best per iteration

        logger.info(f"Initial program: {initial}")

        # Save initial program
        self.save_program(initial, "iteration_0000_best.py")
        iteration_bests.append({
            "iteration": 0,
            "best_score": initial.score,
            "best_program_id": initial.id,
            "metrics": initial.metrics
        })

        # Create N variants
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE: Creating {n} initial lineages")
        logger.info(f"{'='*60}")
        for i in range(n):
            logger.info(f"\n[Lineage {i+1}/{n}] Generating variant from initial program...")
            code = self.mutate_program(
                initial,
                prompt_context=f"This is variant {i+1}/{n}. Create a unique improvement approach."
            )
            logger.info(f"[Lineage {i+1}/{n}] Generated code ({len(code)} chars), evaluating...")
            program = Program(code, parent_id=initial.id, generation=1)
            self.evaluate_program(program)
            logger.info(f"[Lineage {i+1}/{n}] Score: {program.score:.4f}, Metrics: {program.metrics}")
            lineages.append(program)

            self.history.append({
                "iteration": 0,
                "lineage": i,
                "program_id": program.id,
                "score": program.score,
                "metrics": program.metrics
            })

        # Evolve each lineage independently
        for t in range(1, iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {t}/{iterations}")
            logger.info(f"{'='*60}")

            for i, current in enumerate(lineages):
                logger.info(f"\n[Iteration {t}, Lineage {i+1}/{n}] Current score: {current.score:.4f}")
                logger.info(f"[Iteration {t}, Lineage {i+1}/{n}] Generating improved version...")

                # Generate improved version
                code = self.mutate_program(
                    current,
                    prompt_context=f"Iteration {t}/{iterations}. Current score: {current.score:.4f}"
                )
                logger.info(f"[Iteration {t}, Lineage {i+1}/{n}] Generated code ({len(code)} chars), evaluating...")
                new_program = Program(code, parent_id=current.id, generation=t + 1)
                self.evaluate_program(new_program)
                logger.info(f"[Iteration {t}, Lineage {i+1}/{n}] New score: {new_program.score:.4f}")

                # Update lineage if improved
                if new_program.score > current.score:
                    logger.info(
                        f"[Iteration {t}, Lineage {i+1}/{n}] ✓ IMPROVED: {current.score:.4f} → {new_program.score:.4f}"
                    )
                    lineages[i] = new_program
                else:
                    logger.info(f"[Iteration {t}, Lineage {i+1}/{n}] ✗ No improvement, keeping current")

                # Track global best
                if new_program.score > global_best.score:
                    global_best = new_program

                self.history.append({
                    "iteration": t,
                    "lineage": i,
                    "program_id": new_program.id,
                    "score": new_program.score,
                    "metrics": new_program.metrics,
                    "improved": new_program.score > current.score
                })

            # Save best program for this iteration
            self.save_program(global_best, f"iteration_{t:04d}_best.py")
            iteration_bests.append({
                "iteration": t,
                "best_score": global_best.score,
                "best_program_id": global_best.id,
                "metrics": global_best.metrics
            })

        # Find best program across all lineages
        best = max(lineages, key=lambda p: p.score)
        logger.info(f"\n=== Search Complete ===")
        logger.info(f"Best program: {best}")
        logger.info(f"Best score: {best.score:.4f}")
        logger.info(f"Best metrics: {best.metrics}")

        # Save results
        self.save_program(best, "best_program.py")
        self.save_history()

        # Save iteration summary
        import json
        summary_file = self.output_dir / "iteration_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(iteration_bests, f, indent=2)

        # Save all final lineages
        for i, program in enumerate(lineages):
            self.save_program(program, f"lineage_{i}_final.py")

        return best
