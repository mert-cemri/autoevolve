"""
MAP-Elites strategy wrapper

Wraps the existing ProgramDatabase to work with the SearchStrategy interface.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from openevolve.database import Program, ProgramDatabase
from openevolve.search_strategies.base_strategy import SearchStrategy

logger = logging.getLogger(__name__)


class MAPElitesStrategy(SearchStrategy):
    """
    MAP-Elites strategy with island-based evolution.

    This wraps the existing ProgramDatabase to provide the SearchStrategy interface.
    """

    def __init__(self, config):
        # Don't call super().__init__ - we'll use database's storage
        self.config = config

        # Use existing ProgramDatabase
        self.database = ProgramDatabase(config.database)

        # For compatibility with SearchStrategy interface
        self.programs = self.database.programs  # Reference to database programs
        self.best_program_id = None  # Will track separately

        logger.info(
            f"Initialized MAP-Elites strategy with {config.database.num_islands} islands"
        )

    def add_program(self, program: Program, iteration: int) -> None:
        """Add program using database's MAP-Elites logic."""
        self.database.add(program, iteration=iteration)

        # Update our tracked best
        if self.database.best_program_id:
            self.best_program_id = self.database.best_program_id

    def sample_parent(self, iteration: int) -> Optional[Program]:
        """Sample parent from current island using database logic."""
        num_inspirations = self.config.prompt.num_top_programs

        try:
            parent, inspirations = self.database.sample_from_island(
                island_id=self.database.current_island,
                num_inspirations=num_inspirations
            )

            # Store inspirations for use in get_context_programs
            parent._temp_inspirations = inspirations

            return parent

        except Exception as e:
            logger.error(f"Error sampling parent: {e}")
            return None

    def get_context_programs(
        self, parent: Program, iteration: int
    ) -> Tuple[List[Program], List[Program]]:
        """
        Get context programs using database's island-based logic.

        Returns:
            - best_programs: Top programs from island (for "previous attempts")
            - inspiration_programs: Diverse programs (stored during sampling)
        """
        # Get inspirations stored during sampling
        inspirations = getattr(parent, '_temp_inspirations', [])

        # Get island programs
        parent_island = parent.metadata.get("island", self.database.current_island)
        island_program_ids = list(self.database.islands[parent_island])

        if not island_program_ids:
            return [], inspirations

        # Get island programs and sort by fitness
        island_programs = [
            self.database.programs[pid]
            for pid in island_program_ids
            if pid in self.database.programs
        ]

        from openevolve.utils.metrics_utils import get_fitness_score

        island_programs.sort(
            key=lambda p: get_fitness_score(
                p.metrics,
                self.config.database.feature_dimensions
            ),
            reverse=True
        )

        # Best programs: top 3 from island
        num_best = min(self.config.prompt.num_top_programs, len(island_programs))
        best_programs = island_programs[:num_best]

        return best_programs, inspirations

    def get_best_program(self) -> Optional[Program]:
        """Return best program tracked by database."""
        return self.database.get_best_program()

    def get_snapshot(self) -> Dict[str, Any]:
        """Create snapshot including full database state."""
        return {
            "strategy": "map_elites",
            "programs": {pid: prog.to_dict() for pid, prog in self.database.programs.items()},
            "islands": [list(island) for island in self.database.islands],
            "current_island": self.database.current_island,
            "feature_dimensions": self.database.config.feature_dimensions,
            "artifacts": {},  # Limited artifacts as in original
        }

    # Expose database methods for migration, island management, etc.

    def next_island(self) -> None:
        """Switch to next island."""
        self.database.next_island()

    def increment_island_generation(self) -> None:
        """Increment generation counter."""
        self.database.increment_island_generation()

    def should_migrate(self) -> bool:
        """Check if migration should occur."""
        return self.database.should_migrate()

    def migrate_programs(self) -> None:
        """Perform inter-island migration."""
        self.database.migrate_programs()

    def log_island_status(self) -> None:
        """Log island status."""
        self.database.log_island_status()

    def save(self, checkpoint_path: str, iteration: int) -> None:
        """Save database state."""
        self.database.save(checkpoint_path, iteration)

    def load(self, checkpoint_path: str) -> None:
        """Load database state."""
        self.database.load(checkpoint_path)

    @property
    def last_iteration(self) -> int:
        """Get last iteration number."""
        return self.database.last_iteration

    @last_iteration.setter
    def last_iteration(self, value: int) -> None:
        """Set last iteration number."""
        self.database.last_iteration = value
