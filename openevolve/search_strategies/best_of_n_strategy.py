"""
Best-of-N search strategy

Maintains N independent lineages that evolve in parallel.
Each lineage is a simple linear chain of programs.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from openevolve.database import Program
from openevolve.search_strategies.base_strategy import SearchStrategy
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)


class BestOfNStrategy(SearchStrategy):
    """
    Best-of-N search strategy.

    Algorithm:
    1. Create N independent lineages from initial program
    2. Each iteration: evolve all N lineages in parallel
    3. Each lineage keeps only if child is better than parent
    4. Return best program across all lineages
    """

    def __init__(self, config):
        super().__init__(config)

        # Strategy-specific config
        self.n = getattr(config, 'n_lineages', 4)

        # Lineage tracking: lineage_id -> current program_id
        self.lineages: Dict[int, Optional[str]] = {i: None for i in range(self.n)}

        # Lineage assignment: program_id -> lineage_id
        self.program_to_lineage: Dict[str, int] = {}

        # Track which lineages need evolution this iteration
        self.pending_lineages: List[int] = []

        logger.info(f"Initialized Best-of-N strategy with {self.n} lineages")

    def add_program(self, program: Program, iteration: int) -> None:
        """Add program to its lineage."""
        self.programs[program.id] = program

        # Determine lineage
        if program.parent_id and program.parent_id in self.programs:
            # Get the parent to check if it has _temp_lineage (set during sampling)
            parent = self.programs[program.parent_id]
            
            # CRITICAL: Check _temp_lineage first - this is the lineage the parent was sampled from
            # This handles the case where same parent program (e.g., initial) is sampled from different lineages
            if hasattr(parent, '_temp_lineage') and parent._temp_lineage is not None:
                # Use the lineage that the parent was sampled from
                lineage_id = parent._temp_lineage
                # Clear the temp attribute after use
                delattr(parent, '_temp_lineage')
            elif program.parent_id in self.program_to_lineage:
                # Fall back to parent's stored lineage
                lineage_id = self.program_to_lineage[program.parent_id]
            else:
                # Parent has no lineage mapping - shouldn't happen, but handle gracefully
                logger.warning(f"Parent {program.parent_id[:8]}... has no lineage mapping, assigning to lineage 0")
                lineage_id = 0
            
            self.program_to_lineage[program.id] = lineage_id

            # Update lineage if this program is better than current
            current_id = self.lineages[lineage_id]
            if current_id is None:
                # First program in this lineage (shouldn't happen if parent exists)
                self.lineages[lineage_id] = program.id
                logger.info(f"Lineage {lineage_id} initialized with {program.id}")
            else:
                # Compare with current lineage head
                current_program = self.programs[current_id]
                if self._is_better(program, current_program):
                    self.lineages[lineage_id] = program.id
                    logger.info(
                        f"Lineage {lineage_id} updated: {current_id} → {program.id} "
                        f"(fitness: {get_fitness_score(current_program.metrics, [])} → "
                        f"{get_fitness_score(program.metrics, [])})"
                    )
        elif "lineage" in program.metadata:
            # Explicit lineage assignment
            lineage_id = program.metadata["lineage"]
            self.program_to_lineage[program.id] = lineage_id

            # Update lineage if this program is better than current
            current_id = self.lineages[lineage_id]
            if current_id is None:
                self.lineages[lineage_id] = program.id
                logger.info(f"Lineage {lineage_id} initialized with {program.id}")
            else:
                current_program = self.programs[current_id]
                if self._is_better(program, current_program):
                    self.lineages[lineage_id] = program.id
                    logger.info(
                        f"Lineage {lineage_id} updated: {current_id} → {program.id} "
                        f"(fitness: {get_fitness_score(current_program.metrics, [])} → "
                        f"{get_fitness_score(program.metrics, [])})"
                    )
        else:
            # Initial program: initialize ALL lineages with the same program
            # This ensures all N lineages start independently from the same base
            all_empty = all(pid is None for pid in self.lineages.values())
            if all_empty:
                # Initialize all lineages with initial program
                for lineage_id in range(self.n):
                    self.lineages[lineage_id] = program.id
                    self.program_to_lineage[program.id] = 0  # All point to same program, but we track lineage assignment
                # For lineage tracking, we need to handle this specially
                # The initial program exists once but all lineages reference it
                # We'll assign it to lineage 0 for tracking purposes
                self.program_to_lineage[program.id] = 0
                logger.info(f"Initial program {program.id} initialized ALL {self.n} lineages")
            else:
                # Assign to first available lineage (shouldn't normally happen)
                lineage_id = next((i for i, pid in self.lineages.items() if pid is None), 0)
                self.program_to_lineage[program.id] = lineage_id
                if self.lineages[lineage_id] is None:
                    self.lineages[lineage_id] = program.id
                    logger.info(f"Lineage {lineage_id} initialized with {program.id}")

        # Update global best
        self.update_best(program)

    def sample_parent(self, iteration: int) -> Optional[Program]:
        """
        Select parent from lineages in round-robin fashion.

        Returns:
            Current head of next lineage to evolve
        """
        # Initialize pending lineages if empty
        if not self.pending_lineages:
            self.pending_lineages = list(range(self.n))

        # Get next lineage
        if not self.pending_lineages:
            return None

        lineage_id = self.pending_lineages.pop(0)
        parent_id = self.lineages[lineage_id]

        if parent_id is None:
            logger.warning(f"Lineage {lineage_id} has no parent yet")
            return None

        parent = self.programs[parent_id]

        # Store lineage info in metadata for child
        if not hasattr(parent, '_temp_lineage'):
            parent._temp_lineage = lineage_id

        logger.debug(f"Selected parent from lineage {lineage_id}: {parent_id}")
        return parent

    def get_context_programs(
        self, parent: Program, iteration: int
    ) -> Tuple[List[Program], List[Program]]:
        """
        Get context programs for prompt.

        Returns:
            - best_programs: Top 3 programs across all lineages (for "previous attempts")
            - inspiration_programs: Current heads of other lineages (for diversity)
        """
        # Get all lineage heads
        lineage_heads = [
            self.programs[pid] for pid in self.lineages.values() if pid is not None
        ]

        if not lineage_heads:
            return [], []

        # Sort by fitness
        lineage_heads.sort(
            key=lambda p: get_fitness_score(p.metrics, []),
            reverse=True
        )

        # Best programs: top 3
        num_best = min(3, len(lineage_heads))
        best_programs = lineage_heads[:num_best]

        # Inspiration: other lineage heads (exclude parent's lineage)
        parent_lineage = getattr(parent, '_temp_lineage', None)
        if parent_lineage is not None:
            inspiration_programs = [
                p for i, p in enumerate(lineage_heads)
                if self.program_to_lineage.get(p.id) != parent_lineage
            ][:3]
        else:
            inspiration_programs = lineage_heads[num_best:num_best+3]

        return best_programs, inspiration_programs

    def get_best_program(self) -> Optional[Program]:
        """Return best program across all lineages."""
        if self.best_program_id:
            return self.programs.get(self.best_program_id)

        # Fallback: get best from lineage heads
        lineage_heads = [
            self.programs[pid] for pid in self.lineages.values() if pid is not None
        ]

        if not lineage_heads:
            return None

        best = max(lineage_heads, key=lambda p: get_fitness_score(p.metrics, []))
        return best

    def get_snapshot(self) -> Dict[str, Any]:
        """Create snapshot for worker processes."""
        return {
            "strategy": "best_of_n",
            "n_lineages": self.n,
            "programs": {pid: prog.to_dict() for pid, prog in self.programs.items()},
            "lineages": dict(self.lineages),
            "program_to_lineage": dict(self.program_to_lineage),
        }

    def _is_better(self, prog1: Program, prog2: Program) -> bool:
        """Check if prog1 is better than prog2."""
        fitness1 = get_fitness_score(prog1.metrics, [])
        fitness2 = get_fitness_score(prog2.metrics, [])
        return fitness1 > fitness2
