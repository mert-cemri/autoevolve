"""
Monte Carlo Tree Search (MCTS) strategy

Uses UCT (Upper Confidence bounds for Trees) for exploration-exploitation balance.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from openevolve.database import Program
from openevolve.search_strategies.base_strategy import SearchStrategy
from openevolve.utils.metrics_utils import get_fitness_score

logger = logging.getLogger(__name__)


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, program_id: str, parent: Optional["MCTSNode"] = None):
        self.program_id = program_id
        self.parent = parent
        self.children: List["MCTSNode"] = []

        # MCTS statistics
        self.visits: int = 0
        self.total_reward: float = 0.0
        self.best_reward: float = 0.0

    def uct_value(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCT value for this node."""
        if self.visits == 0:
            return float('inf')

        if self.parent is None:
            # Root node: no exploration term needed
            return self.total_reward / self.visits

        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def best_child(self, exploration_constant: float) -> Optional["MCTSNode"]:
        """Select best child using UCT."""
        if not self.children:
            return None

        return max(self.children, key=lambda c: c.uct_value(exploration_constant))

    def update(self, reward: float) -> None:
        """Update node statistics."""
        self.visits += 1
        self.total_reward += reward
        self.best_reward = max(self.best_reward, reward)


class MCTSStrategy(SearchStrategy):
    """
    MCTS search strategy.

    Algorithm:
    1. Selection: Use UCT to traverse tree to a leaf
    2. Expansion: Generate children from the leaf
    3. Simulation: Evaluate new programs
    4. Backpropagation: Update statistics up to root
    """

    def __init__(self, config):
        super().__init__(config)

        # Strategy-specific config
        self.expansion_width = getattr(config, 'expansion_width', 3)
        self.exploration_constant = getattr(config, 'exploration_constant', 1.414)

        # MCTS tree
        self.root: Optional[MCTSNode] = None
        self.nodes: Dict[str, MCTSNode] = {}  # program_id -> node

        # Pending expansions: nodes that need children generated
        self.pending_expansions: List[Tuple[MCTSNode, int]] = []  # (node, child_index)

        # Log configuration clearly
        logger.info("=" * 80)
        logger.info("MCTS STRATEGY INITIALIZATION")
        logger.info("=" * 80)
        logger.info(f"  expansion_width:    {self.expansion_width} (config value: {getattr(config, 'expansion_width', 'NOT SET - using default 3')})")
        logger.info(f"  exploration_constant: {self.exploration_constant} (config value: {getattr(config, 'exploration_constant', 'NOT SET - using default 1.414')})")
        memory_enabled = False
        if hasattr(config, 'memory') and config.memory:
            memory_enabled = getattr(config.memory, 'enabled', False)
        logger.info(f"  memory_enabled:      {memory_enabled}")
        logger.info("=" * 80)

    def add_program(self, program: Program, iteration: int) -> None:
        """Add program to MCTS tree."""
        self.programs[program.id] = program

        # Initialize root
        if self.root is None:
            self.root = MCTSNode(program.id)
            self.nodes[program.id] = self.root
            logger.info(f"Initialized MCTS root with {program.id}")

            # Initial reward
            reward = get_fitness_score(program.metrics, [])
            self.root.update(reward)
        else:
            # Create node and link to parent
            node = MCTSNode(program.id)
            self.nodes[program.id] = node

            # Find parent node
            if program.parent_id and program.parent_id in self.nodes:
                parent_node = self.nodes[program.parent_id]
                parent_node.children.append(node)
                node.parent = parent_node

                # Backpropagate reward
                reward = get_fitness_score(program.metrics, [])
                self._backpropagate(node, reward)
            else:
                logger.warning(f"Parent {program.parent_id} not found for {program.id}")

        # Update global best
        self.update_best(program)

    def sample_parent(self, iteration: int) -> Optional[Program]:
        """
        Sample parent using MCTS selection.

        1. If pending expansions exist, generate their children
        2. Otherwise, select a leaf using UCT and schedule expansion
        """
        if self.root is None:
            return None

        # If we have pending expansions, generate their children
        if self.pending_expansions:
            node, child_idx = self.pending_expansions.pop(0)
            parent_id = node.program_id

            if parent_id not in self.programs:
                logger.warning(f"Parent {parent_id} not found")
                return None

            parent = self.programs[parent_id]
            parent._temp_mcts_info = {
                "node": node,
                "child_index": child_idx,
                "expansion_width": self.expansion_width
            }

            logger.debug(f"Expanding node {parent_id} (child {child_idx}/{self.expansion_width})")
            return parent

        # Select a leaf to expand using UCT
        leaf = self._select_leaf()

        if leaf is None:
            logger.warning("Could not select leaf for expansion")
            return None

        # Schedule expansion of this leaf
        self.pending_expansions = [(leaf, i) for i in range(self.expansion_width)]

        logger.info(
            f"Selected leaf {leaf.program_id} for expansion "
            f"(visits: {leaf.visits}, avg_reward: {leaf.total_reward / max(leaf.visits, 1):.4f})"
        )

        # Return first expansion
        if self.pending_expansions:
            return self.sample_parent(iteration)

        return None

    def _select_leaf(self) -> Optional[MCTSNode]:
        """
        Select a leaf node to expand using UCT.

        Traverses from root to a leaf, selecting best child at each level.
        """
        if self.root is None:
            return None

        current = self.root

        # Traverse to leaf
        while current.children:
            current = current.best_child(self.exploration_constant)
            if current is None:
                break

        return current

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagate reward up to root."""
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def get_context_programs(
        self, parent: Program, iteration: int
    ) -> Tuple[List[Program], List[Program]]:
        """
        Get context programs for prompt.

        Returns:
            - best_programs: Top programs by total reward (most promising)
            - inspiration_programs: High-reward leaf nodes (exploratory)
        """
        # Get all programs with nodes
        all_nodes = list(self.nodes.values())

        if not all_nodes:
            return [], []

        # Best programs: highest average reward
        visited_nodes = [n for n in all_nodes if n.visits > 0]
        if visited_nodes:
            visited_nodes.sort(
                key=lambda n: n.total_reward / n.visits,
                reverse=True
            )
            best_programs = [
                self.programs[n.program_id]
                for n in visited_nodes[:3]
                if n.program_id in self.programs
            ]
        else:
            best_programs = []

        # Inspiration: leaf nodes with good rewards (exploratory)
        leaf_nodes = [n for n in all_nodes if not n.children and n.visits > 0]
        if leaf_nodes:
            leaf_nodes.sort(key=lambda n: n.best_reward, reverse=True)
            inspiration_programs = [
                self.programs[n.program_id]
                for n in leaf_nodes[:3]
                if n.program_id in self.programs
            ]
        else:
            inspiration_programs = []

        return best_programs, inspiration_programs

    def get_best_program(self) -> Optional[Program]:
        """Return program with highest average reward."""
        if self.best_program_id:
            return self.programs.get(self.best_program_id)

        # Fallback: highest average reward
        if not self.nodes:
            return None

        visited_nodes = [n for n in self.nodes.values() if n.visits > 0]
        if not visited_nodes:
            return None

        best_node = max(visited_nodes, key=lambda n: n.total_reward / n.visits)
        return self.programs.get(best_node.program_id)

    def get_snapshot(self) -> Dict[str, Any]:
        """Create snapshot for worker processes."""
        # Serialize tree structure
        tree_data = {}
        for pid, node in self.nodes.items():
            tree_data[pid] = {
                "parent_id": node.parent.program_id if node.parent else None,
                "children_ids": [c.program_id for c in node.children],
                "visits": node.visits,
                "total_reward": node.total_reward,
                "best_reward": node.best_reward,
            }

        return {
            "strategy": "mcts",
            "expansion_width": self.expansion_width,
            "exploration_constant": self.exploration_constant,
            "programs": {pid: prog.to_dict() for pid, prog in self.programs.items()},
            "tree": tree_data,
            "root_id": self.root.program_id if self.root else None,
        }
