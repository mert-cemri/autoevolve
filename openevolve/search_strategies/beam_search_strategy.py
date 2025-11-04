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

        # Store config reference for logging
        self.config = config

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
        
        # Track branches in current cycle
        self.branches_expected_in_cycle: int = 0  # How many branches we scheduled
        self.branches_completed_in_cycle: int = 0  # How many have completed (success + failure)

        # Get memory setting
        memory_enabled = False
        if hasattr(config, 'memory') and config.memory:
            memory_enabled = getattr(config.memory, 'enabled', False)

        # Log configuration clearly
        logger.info("=" * 80)
        logger.info("BEAM SEARCH STRATEGY INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"  beam_width:     {self.beam_width} (config value: {getattr(config, 'beam_width', 'NOT SET - using default')})")
        logger.info(f"  branch_factor:   {self.branch_factor} (config value: {getattr(config, 'branch_factor', 'NOT SET - using default')})")
        logger.info(f"  memory_enabled:  {memory_enabled}")
        logger.info("=" * 80)

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
        
        # Track completion of this branch (success case)
        self.branches_completed_in_cycle += 1

        # Update global best
        self.update_best(program)
    
    def mark_branch_failed(self) -> None:
        """Mark a branch as failed (no valid code, timeout, etc)."""
        self.branches_completed_in_cycle += 1

    def sample_parent(self, iteration: int) -> Optional[Program]:
        """
        Sample parent from current beam.

        For each iteration, we need to generate branch_factor total children.
        This distributes branches across beam members.

        Beam selection happens when all pending_parents are exhausted,
        allowing search to continue even if some children fail.
        """
        # Check if we need to prepare a new beam cycle
        # This happens when:
        # 1. All branches from current cycle are sampled (pending_parents empty)
        # 2. AND we have candidates (even if not all branch_factor succeeded)
        # 3. OR it's the first iteration (no beam cycle started yet)
        if not self.pending_parents:
            # All branches from current cycle have been sampled
            # But we need to wait until all branches have COMPLETED (success or failure)
            # before selecting beam
            if self.branches_expected_in_cycle > 0:
                # Check if all expected branches have completed
                if self.branches_completed_in_cycle >= self.branches_expected_in_cycle:
                    # All branches completed - can select beam
                    if len(self.candidates) > 0:
                        logger.info("=" * 80)
                        logger.info("BEAM CYCLE COMPLETE - ALL BRANCHES FINISHED - SELECTING NEW BEAM")
                        logger.info("=" * 80)
                        logger.info(f"  Iteration:           {iteration}")
                        logger.info(f"  Branches completed:  {self.branches_completed_in_cycle}/{self.branches_expected_in_cycle}")
                        logger.info(f"  Candidates found:    {len(self.candidates)} (successful)")
                        logger.info(f"  Failed branches:     {self.branches_expected_in_cycle - len(self.candidates)}")
                        logger.info(f"  Success rate:        {len(self.candidates)}/{self.branches_expected_in_cycle} "
                                   f"({100.0 * len(self.candidates) / self.branches_expected_in_cycle:.1f}%)")
                        logger.info(f"  Current beam_width:   {self.beam_width}")
                        memory_enabled = False
                        if hasattr(self, 'config') and hasattr(self.config, 'memory') and self.config.memory:
                            memory_enabled = getattr(self.config.memory, 'enabled', False)
                        logger.info(f"  Memory enabled:       {memory_enabled}")
                        logger.info("=" * 80)
                        self._prepare_new_iteration(iteration)
                    elif len(self.candidates) == 0:
                        # All branches failed - still need to select something (keep current beam)
                        logger.warning("=" * 80)
                        logger.warning("BEAM CYCLE COMPLETE - ALL BRANCHES FAILED")
                        logger.warning("=" * 80)
                        logger.warning(f"  Iteration:           {iteration}")
                        logger.warning(f"  Branches completed:  {self.branches_completed_in_cycle}/{self.branches_expected_in_cycle}")
                        logger.warning(f"  All {self.branches_expected_in_cycle} branches failed - keeping current beam")
                        logger.warning("=" * 80)
                        self._prepare_new_iteration(iteration)
                else:
                    # Not all branches completed yet - wait
                    logger.debug(
                        f"Waiting for branches to complete: "
                        f"{self.branches_completed_in_cycle}/{self.branches_expected_in_cycle}, "
                        f"iteration {iteration}"
                    )
                    return None
            elif len(self.candidates) > 0:
                # No expected branches tracked (first cycle edge case) - proceed if we have candidates
                self._prepare_new_iteration(iteration)
            elif len(self.candidates) == 0 and self.beam:
                # Initial setup: beam exists (from initial program) but no candidates yet
                # This is the very first call (iteration 1) - prepare first cycle
                self._prepare_new_iteration(iteration)
            else:
                # No candidates yet and no beam - this shouldn't happen
                # Return None to wait
                logger.warning(
                    f"No candidates available and no beam. Waiting for initialization. "
                    f"iteration {iteration}"
                )
                return None

        # If still no pending parents after preparation, we're done
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
        """
        Prepare for new beam cycle: select new beam from candidates and schedule branches.
        
        This is called when all branches from the current cycle are complete
        (pending_parents is empty), allowing beam selection even if some children failed.
        """
        # If we have candidates from previous beam cycle, select new beam
        if self.candidates:
            self._select_new_beam()

        # Track iteration for logging/debugging
        self.current_iteration = iteration
        
        # Reset candidates for next cycle
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

        # Track how many branches we expect to complete in this cycle
        self.branches_expected_in_cycle = len(self.pending_parents)
        self.branches_completed_in_cycle = 0

        logger.info("=" * 80)
        logger.info("STARTING NEW BEAM CYCLE")
        logger.info("=" * 80)
        logger.info(f"  Iteration:         {iteration}")
        logger.info(f"  Beam programs:     {len(self.beam)}")
        logger.info(f"  branch_factor:     {self.branch_factor}")
        logger.info(f"  Branches scheduled: {len(self.pending_parents)}")
        logger.info(f"  Expected to complete: {self.branches_expected_in_cycle}")
        logger.info(f"  Per beam member:   {branches_per_member}")
        memory_enabled = False
        if hasattr(self, 'config') and hasattr(self.config, 'memory') and self.config.memory:
            memory_enabled = getattr(self.config.memory, 'enabled', False)
        logger.info(f"  Memory enabled:    {memory_enabled}")
        logger.info("=" * 80)

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
        beam_scores = [get_fitness_score(self.programs[pid].metrics, []) for pid in new_beam]

        logger.info("=" * 80)
        logger.info("NEW BEAM SELECTED")
        logger.info("=" * 80)
        logger.info(f"  beam_width:      {self.beam_width}")
        logger.info(f"  Candidates pool: {len(candidate_programs)}")
        logger.info(f"  Selected beam:   {len(new_beam)} programs")
        logger.info(f"  Beam scores:     {[f'{s:.4f}' for s in beam_scores[:5]]}")
        logger.info(f"  Best score:      {beam_scores[0]:.4f}" if beam_scores else "  (no scores)")
        logger.info("=" * 80)

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

