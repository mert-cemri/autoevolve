"""
Beam Search strategy

Maintains a beam of M best programs and branches N times per iteration.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from openevolve.database import Program
from openevolve.search_strategies.base_strategy import SearchStrategy
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)


class BeamSearchStrategy(SearchStrategy):
    """
    Beam Search strategy.

    Algorithm:
    1. Start with initial program
    2. Each iteration:
       - Generate N candidates from current beam (M programs)
       - Keep top M candidates as new beam
    3. Return best program from final beam
    """

    def __init__(self, config):
        super().__init__(config)

        # Strategy-specific config
        self.beam_width = getattr(config, 'beam_width', 4)
        self.branch_factor = getattr(config, 'branch_factor', 8)

        # Current beam: list of program IDs
        self.beam: List[str] = []

        # Track which beam members need to generate children
        self.pending_parents: List[Tuple[str, int]] = []  # (parent_id, branch_index)

        # Candidates for next beam (accumulated during iteration)
        self.candidates: List[str] = []
        self.current_iteration: int = 0

        logger.info(
            f"Initialized Beam Search strategy with beam_width={self.beam_width}, "
            f"branch_factor={self.branch_factor}"
        )

    def add_program(self, program: Program, iteration: int) -> None:
        """Add program to beam or candidates."""
        self.programs[program.id] = program

        # First program initializes beam
        if not self.beam and not self.candidates:
            self.beam = [program.id]
            logger.info(f"Initialized beam with {program.id}")
        else:
            # Add to candidates for beam selection
            self.candidates.append(program.id)

        # Update global best
        self.update_best(program)

    def sample_parent(self, iteration: int) -> Optional[Program]:
        """
        Sample parent from current beam.

        For each iteration, we need to generate branch_factor total children.
        This distributes branches across beam members.
        """
        # New iteration - reset candidates and prepare parents
        if iteration != self.current_iteration:
            self._prepare_new_iteration(iteration)

        # Get next parent to generate child from
        if not self.pending_parents:
            return None

        parent_id, branch_idx = self.pending_parents.pop(0)

        if parent_id not in self.programs:
            logger.warning(f"Parent {parent_id} not found in programs")
            return None

        parent = self.programs[parent_id]

        # Store branch info for context
        parent._temp_branch_info = {
            "beam_position": self.beam.index(parent_id),
            "branch_index": branch_idx
        }

        logger.debug(
            f"Selected parent {parent_id} (beam position {parent._temp_branch_info['beam_position']}, "
            f"branch {branch_idx})"
        )

        return parent

    def _prepare_new_iteration(self, iteration: int) -> None:
        """Prepare for new iteration: update beam and schedule branches."""
        logger.info(f"Preparing iteration {iteration}")

        # If we have candidates from previous iteration, select new beam
        if self.candidates:
            self._select_new_beam()

        self.current_iteration = iteration
        self.candidates = []

        # Schedule branches: distribute branch_factor across beam members
        if not self.beam:
            logger.warning("No programs in beam!")
            return

        branches_per_member = max(1, self.branch_factor // len(self.beam))
        self.pending_parents = []

        for parent_id in self.beam:
            for branch_idx in range(branches_per_member):
                self.pending_parents.append((parent_id, branch_idx))

        logger.info(
            f"Scheduled {len(self.pending_parents)} branches "
            f"({branches_per_member} per beam member)"
        )

    def _select_new_beam(self) -> None:
        """Select top beam_width programs from candidates."""
        # Get all candidate programs
        candidate_programs = [self.programs[pid] for pid in self.candidates if pid in self.programs]

        if not candidate_programs:
            logger.warning("No valid candidates for beam selection")
            return

        # Sort by fitness
        candidate_programs.sort(
            key=lambda p: get_fitness_score(p.metrics, []),
            reverse=True
        )

        # Select top beam_width
        new_beam = [p.id for p in candidate_programs[:self.beam_width]]

        logger.info(
            f"Selected new beam: {len(new_beam)} programs "
            f"(scores: {[get_fitness_score(self.programs[pid].metrics, []) for pid in new_beam[:3]]})"
        )

        self.beam = new_beam

    def get_context_programs(
        self, parent: Program, iteration: int
    ) -> Tuple[List[Program], List[Program]]:
        """
        Get context programs for prompt.

        Returns:
            - best_programs: Top programs from current beam
            - inspiration_programs: Other beam members
        """
        # Get all beam programs
        beam_programs = [self.programs[pid] for pid in self.beam if pid in self.programs]

        if not beam_programs:
            return [], []

        # Sort by fitness
        beam_programs.sort(
            key=lambda p: get_fitness_score(p.metrics, []),
            reverse=True
        )

        # Best programs: top 3 from beam
        num_best = min(3, len(beam_programs))
        best_programs = beam_programs[:num_best]

        # Inspiration: other beam members
        inspiration_programs = beam_programs[num_best:num_best+3]

        return best_programs, inspiration_programs

    def get_best_program(self) -> Optional[Program]:
        """Return best program from current beam."""
        if self.best_program_id:
            return self.programs.get(self.best_program_id)

        # Fallback: best from beam
        if not self.beam:
            return None

        beam_programs = [self.programs[pid] for pid in self.beam if pid in self.programs]
        if not beam_programs:
            return None

        best = max(beam_programs, key=lambda p: get_fitness_score(p.metrics, []))
        return best

    def get_snapshot(self) -> Dict[str, Any]:
        """Create snapshot for worker processes."""
        return {
            "strategy": "beam_search",
            "beam_width": self.beam_width,
            "branch_factor": self.branch_factor,
            "programs": {pid: prog.to_dict() for pid, prog in self.programs.items()},
            "beam": list(self.beam),
            "current_iteration": self.current_iteration,
        }
