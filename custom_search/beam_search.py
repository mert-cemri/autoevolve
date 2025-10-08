"""
Beam Search Strategy

Starts with initial program, branches N times, keeps top M, then branches each
into N/M programs. Repeats for T iterations and returns best program.
"""

import logging
import os
from typing import List

from .base_search import BaseSearch, Program

logger = logging.getLogger(__name__)


class BeamSearch(BaseSearch):
    """
    Beam Search Strategy

    Algorithm:
    1. Start with initial program
    2. For T iterations:
       a. Branch current beam (size M) into N total candidates
       b. Evaluate all candidates
       c. Keep top M as new beam
    3. Return best program from final beam
    """

    def search(
        self,
        beam_width: int = 4,
        branch_factor: int = 8,
        iterations: int = 10
    ) -> Program:
        """
        Run Beam Search

        Args:
            beam_width: Number of programs to keep at each iteration (M)
            branch_factor: Total number of branches per iteration (N)
            iterations: Number of iterations (T)

        Returns:
            Best program found
        """
        logger.info(
            f"Starting Beam Search with beam_width={beam_width}, "
            f"branch_factor={branch_factor}, iterations={iterations}, "
            f"num_eval_problems={self.num_eval_problems}"
        )
        logger.info(
            f"Evolution model: {self.model}, "
            f"Agent model: {os.environ.get('OPENEVOLVE_MODEL', 'unknown')}"
        )

        # Initialize with initial program
        initial = Program(self.initial_program, parent_id=None, generation=0)
        self.evaluate_program(initial)
        logger.info(f"Initial program: {initial}")

        # Current beam
        beam: List[Program] = [initial]

        # Track all programs and best per iteration
        all_programs: List[Program] = [initial]
        global_best = initial
        iteration_bests = []

        # Save initial
        self.save_program(initial, "iteration_0000_best.py")
        iteration_bests.append({
            "iteration": 0,
            "best_score": initial.score,
            "best_program_id": initial.id,
            "beam_scores": [initial.score],
            "metrics": initial.metrics
        })

        # Iterate
        for t in range(1, iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {t}/{iterations}")
            logger.info(f"{'='*60}")
            logger.info(f"Current beam size: {len(beam)}")

            # Calculate how many branches per beam member
            branches_per_member = max(1, branch_factor // len(beam))
            logger.info(f"Generating {branches_per_member} branches per beam member\n")

            # Generate candidates
            candidates: List[Program] = []

            for i, parent in enumerate(beam):
                for j in range(branches_per_member):
                    candidate_num = len(candidates) + 1
                    logger.info(
                        f"[Candidate {candidate_num}/{branch_factor}] "
                        f"Beam member {i+1}/{len(beam)}, branch {j+1}/{branches_per_member}, "
                        f"parent_score={parent.score:.4f}"
                    )

                    code = self.mutate_program(
                        parent,
                        prompt_context=(
                            f"Iteration {t}/{iterations}. "
                            f"Parent score: {parent.score:.4f}. "
                            f"Branch {j+1}/{branches_per_member} from this parent."
                        )
                    )

                    program = Program(code, parent_id=parent.id, generation=t)
                    self.evaluate_program(program)
                    candidates.append(program)
                    all_programs.append(program)

                    # Track global best
                    if program.score > global_best.score:
                        logger.info(f"  ★ NEW GLOBAL BEST: {global_best.score:.4f} → {program.score:.4f}")
                        global_best = program

                    self.history.append({
                        "iteration": t,
                        "parent_id": parent.id,
                        "program_id": program.id,
                        "score": program.score,
                        "metrics": program.metrics
                    })

            # Select top M candidates for new beam
            candidates.sort(key=lambda p: p.score, reverse=True)
            beam = candidates[:beam_width]

            logger.info(f"New beam scores: {[f'{p.score:.4f}' for p in beam]}")
            logger.info(f"Best in beam: {beam[0].score:.4f}")

            # Save best program for this iteration
            self.save_program(global_best, f"iteration_{t:04d}_best.py")
            iteration_bests.append({
                "iteration": t,
                "best_score": global_best.score,
                "best_program_id": global_best.id,
                "beam_scores": [p.score for p in beam],
                "metrics": global_best.metrics
            })

        # Best program is top of final beam
        best = beam[0]
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

        # Save final beam
        for i, program in enumerate(beam):
            self.save_program(program, f"beam_{i}.py")

        return best
