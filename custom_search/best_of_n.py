"""
Best of N Search Strategy

Generates N independent lineages from the initial program and evolves each
linearly for T iterations. Returns the best program across all lineages.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from .base_search import BaseSearch, Program

logger = logging.getLogger(__name__)


class BestOfNSearch(BaseSearch):
    """
    Best of N Search Strategy

    Algorithm:
    1. Generate N variants from initial program
    2. For T iterations:
       - For each variant, generate improved version (in parallel)
       - Evaluate and update if better
    3. Return best program across all lineages
    """

    def _evolve_lineage(self, parent: Program, lineage_idx: int, total: int,
                       iteration: int, total_iterations: int) -> Tuple[Program, int]:
        """
        Evolve a single lineage (mutation + evaluation).

        Returns:
            Tuple of (new_program, lineage_idx)
        """
        try:
            logger.info(
                f"  [Lineage {lineage_idx+1}/{total}] Starting evolution "
                f"(parent_score={parent.score:.4f})"
            )

            # Generate improved version
            code = self.mutate_program(
                parent,
                prompt_context=f"Iteration {iteration}/{total_iterations}. Current score: {parent.score:.4f}"
            )
            logger.info(
                f"  [Lineage {lineage_idx+1}/{total}] Code generated ({len(code)} chars), evaluating..."
            )

            # Evaluate
            new_program = Program(code, parent_id=parent.id, generation=iteration + 1)
            self.evaluate_program(new_program)

            logger.info(
                f"  [Lineage {lineage_idx+1}/{total}] ✓ Evaluation complete: "
                f"score={new_program.score:.4f}, metrics={new_program.metrics}"
            )

            return new_program, lineage_idx

        except Exception as e:
            logger.error(f"  [Lineage {lineage_idx+1}/{total}] ✗ Failed: {e}")
            # Return parent program if evolution fails
            return parent, lineage_idx

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

        # Create N variants (in parallel)
        logger.info(f"\n{'='*60}")
        logger.info(f"STAGE: Creating {n} initial lineages (PARALLEL)")
        logger.info(f"{'='*60}\n")

        def create_initial_variant(idx: int) -> Tuple[Program, int]:
            """Create a single initial variant."""
            try:
                logger.info(f"  [Lineage {idx+1}/{n}] Generating variant from initial program...")
                code = self.mutate_program(
                    initial,
                    prompt_context=f"This is variant {idx+1}/{n}. Create a unique improvement approach."
                )
                logger.info(f"  [Lineage {idx+1}/{n}] Generated code ({len(code)} chars), evaluating...")
                program = Program(code, parent_id=initial.id, generation=1)
                self.evaluate_program(program)
                logger.info(
                    f"  [Lineage {idx+1}/{n}] ✓ Complete: score={program.score:.4f}, "
                    f"metrics={program.metrics}"
                )
                return program, idx
            except Exception as e:
                logger.error(f"  [Lineage {idx+1}/{n}] ✗ Failed: {e}")
                return initial, idx

        # Execute in parallel
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [executor.submit(create_initial_variant, i) for i in range(n)]
            results = []
            for future in as_completed(futures):
                program, idx = future.result()
                results.append((idx, program))

        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])
        lineages = [prog for _, prog in results]

        # Log results
        logger.info(f"\n{'='*60}")
        logger.info(f"Initial lineages created:")
        for i, program in enumerate(lineages):
            logger.info(f"  Lineage {i+1}: score={program.score:.4f}")
            self.history.append({
                "iteration": 0,
                "lineage": i,
                "program_id": program.id,
                "score": program.score,
                "metrics": program.metrics
            })
        logger.info(f"{'='*60}")

        # Evolve each lineage independently (in parallel)
        for t in range(1, iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {t}/{iterations} (PARALLEL)")
            logger.info(f"{'='*60}")
            logger.info(f"Current lineage scores: {[f'{p.score:.4f}' for p in lineages]}")
            logger.info(f"Global best: {global_best.score:.4f}\n")

            # Evolve all lineages in parallel
            with ThreadPoolExecutor(max_workers=n) as executor:
                futures = [
                    executor.submit(self._evolve_lineage, lineages[i], i, n, t, iterations)
                    for i in range(n)
                ]
                results = []
                for future in as_completed(futures):
                    new_program, idx = future.result()
                    results.append((idx, new_program))

            # Sort by index to maintain lineage order
            results.sort(key=lambda x: x[0])

            # Update lineages and track improvements
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {t} Results:")
            logger.info(f"{'='*60}")
            for idx, new_program in results:
                current = lineages[idx]

                # Update lineage if improved
                if new_program.score > current.score:
                    improvement = new_program.score - current.score
                    logger.info(
                        f"  Lineage {idx+1}: {current.score:.4f} → {new_program.score:.4f} "
                        f"(+{improvement:.4f}) ✓ IMPROVED"
                    )
                    lineages[idx] = new_program
                else:
                    logger.info(
                        f"  Lineage {idx+1}: {current.score:.4f} → {new_program.score:.4f} "
                        f"✗ No improvement"
                    )

                # Track global best
                if new_program.score > global_best.score:
                    logger.info(
                        f"    ★★★ NEW GLOBAL BEST: {global_best.score:.4f} → "
                        f"{new_program.score:.4f} ★★★"
                    )
                    global_best = new_program

                self.history.append({
                    "iteration": t,
                    "lineage": idx,
                    "program_id": new_program.id,
                    "score": new_program.score,
                    "metrics": new_program.metrics,
                    "improved": new_program.score > current.score
                })

            logger.info(f"{'='*60}")
            logger.info(f"End of iteration {t}: Global best = {global_best.score:.4f}")
            logger.info(f"{'='*60}")

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
