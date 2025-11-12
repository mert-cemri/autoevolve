"""
Base class for search strategies
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from openevolve.database import Program

logger = logging.getLogger(__name__)


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.

    Each strategy implements its own:
    - Parent selection logic
    - Program storage/organization
    - Context program selection (for prompts)
    - Best program tracking
    """

    def __init__(self, config):
        """
        Initialize the search strategy.

        Args:
            config: Configuration object
        """
        self.config = config
        self.programs: Dict[str, Program] = {}
        self.best_program_id: Optional[str] = None

    @abstractmethod
    def add_program(self, program: Program, iteration: int) -> None:
        """
        Add a program to the strategy's storage.

        Args:
            program: Program to add
            iteration: Current iteration number
        """
        pass

    @abstractmethod
    def sample_parent(self, iteration: int) -> Optional[Program]:
        """
        Select a parent program to evolve.

        Args:
            iteration: Current iteration number

        Returns:
            Selected parent program, or None if no programs available
        """
        pass

    @abstractmethod
    def get_context_programs(
        self, parent: Program, iteration: int
    ) -> Tuple[List[Program], List[Program]]:
        """
        Get programs to show in the evolution prompt.

        Args:
            parent: The parent program being evolved
            iteration: Current iteration number

        Returns:
            Tuple of (best_programs, inspiration_programs)
            - best_programs: Top performers (for "previous attempts" section)
            - inspiration_programs: Diverse/exploratory programs (for inspiration)
        """
        pass

    @abstractmethod
    def get_best_program(self) -> Optional[Program]:
        """
        Get the best program found so far.

        Returns:
            Best program, or None if no programs
        """
        pass

    @abstractmethod
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Create a serializable snapshot of the strategy's state.
        Used for passing to worker processes.

        Returns:
            Dictionary containing strategy state
        """
        pass

    def update_best(self, program: Program) -> bool:
        """
        Update best program tracking if this program is better.

        Args:
            program: Candidate program

        Returns:
            True if this is a new best program
        """
        from openevolve.utils.metrics_utils import get_fitness_score

        if not program.metrics:
            return False

        # Get new program fitness
        new_fitness = get_fitness_score(
            program.metrics,
            getattr(self.config, 'feature_dimensions', [])
        )

        # Get current best fitness
        current_best_fitness = None
        if self.best_program_id and self.best_program_id in self.programs:
            current_best = self.programs[self.best_program_id]
            current_best_fitness = get_fitness_score(
                current_best.metrics,
                getattr(self.config, 'feature_dimensions', [])
            )
        elif self.best_program_id:
            # Best program is missing - find actual best from remaining programs
            logger.warning(f"Best program {self.best_program_id} no longer in programs")
            if self.programs:
                all_programs = list(self.programs.values())
                actual_best = max(
                    all_programs,
                    key=lambda p: get_fitness_score(
                        p.metrics,
                        getattr(self.config, 'feature_dimensions', [])
                    ) if p.metrics else -float('inf')
                )
                current_best_fitness = get_fitness_score(
                    actual_best.metrics,
                    getattr(self.config, 'feature_dimensions', [])
                )
                # Update best_program_id to point to actual best if new program isn't better
                if new_fitness <= current_best_fitness:
                    self.best_program_id = actual_best.id
                    logger.info(f"Restored best program to {actual_best.id} (actual best in population)")

        # Update if better or first program
        if current_best_fitness is None or new_fitness > current_best_fitness:
            self.best_program_id = program.id
            logger.info(f"🌟 New best program: {program.id} (fitness: {new_fitness:.4f})")
            return True

        return False

    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        return self.__class__.__name__.replace("Strategy", "")
