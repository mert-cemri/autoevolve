"""
Monte Carlo Tree Search (MCTS) for Code Evolution

Uses MCTS with UCT selection to explore the space of program improvements.
"""

import logging
import math
import os
from typing import List, Optional

from .base_search import BaseSearch, Program

logger = logging.getLogger(__name__)


class MCTSNode:
    """Node in the MCTS tree"""

    def __init__(self, program: Program, parent: Optional['MCTSNode'] = None):
        self.program = program
        self.parent = parent
        self.children: List[MCTSNode] = []

        # MCTS statistics
        self.visits = 0
        self.total_score = 0.0
        self.best_score = -float('inf')

    @property
    def avg_score(self) -> float:
        """Average score from simulations"""
        return self.total_score / self.visits if self.visits > 0 else 0.0

    def uct_score(self, exploration_constant: float = 1.414) -> float:
        """
        Calculate UCT (Upper Confidence Bound for Trees) score

        Args:
            exploration_constant: Exploration vs exploitation tradeoff

        Returns:
            UCT score
        """
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return self.avg_score

        exploitation = self.avg_score
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def best_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select best child using UCT"""
        return max(self.children, key=lambda c: c.uct_score(exploration_constant))

    def is_leaf(self) -> bool:
        """Check if node is a leaf"""
        return len(self.children) == 0

    def __repr__(self):
        return (
            f"MCTSNode(id={self.program.id}, visits={self.visits}, "
            f"avg={self.avg_score:.4f}, best={self.best_score:.4f})"
        )


class MCTSSearch(BaseSearch):
    """
    Monte Carlo Tree Search for Code Evolution

    Algorithm:
    1. Selection: Use UCT to select promising path from root to leaf
    2. Expansion: Generate children from selected leaf
    3. Simulation: Evaluate new program
    4. Backpropagation: Update statistics along path to root
    """

    def search(
        self,
        iterations: int = 50,
        expansion_width: int = 3,
        exploration_constant: float = 1.414
    ) -> Program:
        """
        Run MCTS search

        Args:
            iterations: Number of MCTS iterations
            expansion_width: Number of children to generate per expansion
            exploration_constant: UCT exploration parameter

        Returns:
            Best program found
        """
        logger.info(
            f"Starting MCTS with iterations={iterations}, "
            f"expansion_width={expansion_width}, "
            f"exploration_constant={exploration_constant}, "
            f"num_eval_problems={self.num_eval_problems}"
        )
        logger.info(
            f"Evolution model: {self.model}, "
            f"Agent model: {os.environ.get('OPENEVOLVE_MODEL', 'unknown')}"
        )

        # Initialize root node with initial program
        initial = Program(self.initial_program, parent_id=None, generation=0)
        self.evaluate_program(initial)
        root = MCTSNode(initial)
        root.visits = 1
        root.total_score = initial.score
        root.best_score = initial.score

        logger.info(f"Root program: {initial}")

        # Track best program globally and per iteration
        best_program = initial
        iteration_bests = []

        # Save initial
        self.save_program(initial, "iteration_0000_best.py")
        iteration_bests.append({
            "iteration": 0,
            "best_score": initial.score,
            "best_program_id": initial.id,
            "tree_visits": root.visits,
            "tree_avg_score": root.avg_score,
            "metrics": initial.metrics
        })

        # MCTS iterations
        for iteration in range(1, iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"MCTS ITERATION {iteration}/{iterations}")
            logger.info(f"{'='*60}")

            # 1. Selection: Traverse tree using UCT
            logger.info(f"Phase 1/4: Selection (UCT)")
            node = self._select(root, exploration_constant)
            logger.info(f"  Selected node: {node}")

            # 2. Expansion: Generate children
            if node.visits > 0 or node == root:  # Expand visited nodes
                logger.info(f"Phase 2/4: Expansion (generating {expansion_width} children)")
                children = self._expand(node, expansion_width)
                logger.info(f"  Expanded {len(children)} children")

                # 3. Simulation: Evaluate children
                logger.info(f"Phase 3/4: Simulation (evaluating {len(children)} children)")
                for idx, child in enumerate(children, 1):
                    logger.info(f"  Child {idx}/{len(children)}: {child.program.id}")
                    self.evaluate_program(child.program)
                    child.visits = 1
                    child.total_score = child.program.score
                    child.best_score = child.program.score

                    # Track best
                    if child.program.score > best_program.score:
                        logger.info(f"  ★ NEW BEST: {best_program.score:.4f} → {child.program.score:.4f}")
                        best_program = child.program

                    # 4. Backpropagation: Update path to root
                    self._backpropagate(child, child.program.score)

                    self.history.append({
                        "iteration": iteration,
                        "node_id": child.program.id,
                        "parent_id": child.program.parent_id,
                        "score": child.program.score,
                        "visits": child.visits,
                        "avg_score": child.avg_score,
                        "metrics": child.program.metrics
                    })

                logger.info(f"Phase 4/4: Backpropagation complete")

            # Log tree statistics
            logger.info(f"\nIteration Summary:")
            logger.info(f"  Root visits: {root.visits}, avg_score: {root.avg_score:.4f}")
            logger.info(f"  Best so far: {best_program.score:.4f}")

            # Save best program for this iteration
            self.save_program(best_program, f"iteration_{iteration:04d}_best.py")
            iteration_bests.append({
                "iteration": iteration,
                "best_score": best_program.score,
                "best_program_id": best_program.id,
                "tree_visits": root.visits,
                "tree_avg_score": root.avg_score,
                "metrics": best_program.metrics
            })

        logger.info(f"\n=== MCTS Complete ===")
        logger.info(f"Best program: {best_program}")
        logger.info(f"Best score: {best_program.score:.4f}")
        logger.info(f"Best metrics: {best_program.metrics}")

        # Save results
        self.save_program(best_program, "best_program.py")
        self.save_history()

        # Save iteration summary
        import json
        summary_file = self.output_dir / "iteration_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(iteration_bests, f, indent=2)

        # Save tree statistics
        self._save_tree_stats(root)

        return best_program

    def _select(self, root: MCTSNode, exploration_constant: float) -> MCTSNode:
        """
        Select a leaf node using UCT

        Args:
            root: Root node
            exploration_constant: UCT exploration parameter

        Returns:
            Selected leaf node
        """
        node = root

        while not node.is_leaf():
            node = node.best_child(exploration_constant)

        return node

    def _expand(self, node: MCTSNode, width: int) -> List[MCTSNode]:
        """
        Expand a node by generating children

        Args:
            node: Node to expand
            width: Number of children to generate

        Returns:
            List of new child nodes
        """
        children = []

        for i in range(width):
            # Generate improved version
            code = self.mutate_program(
                node.program,
                prompt_context=(
                    f"Current score: {node.program.score:.4f}. "
                    f"Visits: {node.visits}. "
                    f"Child {i+1}/{width}."
                )
            )

            program = Program(
                code,
                parent_id=node.program.id,
                generation=node.program.generation + 1
            )

            child = MCTSNode(program, parent=node)
            node.children.append(child)
            children.append(child)

        return children

    def _backpropagate(self, node: MCTSNode, score: float):
        """
        Backpropagate score up the tree

        Args:
            node: Starting node
            score: Score to backpropagate
        """
        current = node

        while current is not None:
            current.visits += 1
            current.total_score += score
            current.best_score = max(current.best_score, score)
            current = current.parent

    def _save_tree_stats(self, root: MCTSNode):
        """Save tree statistics"""
        stats = self._collect_tree_stats(root)

        stats_file = self.output_dir / "mcts_tree_stats.json"
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    def _collect_tree_stats(self, node: MCTSNode, depth: int = 0) -> dict:
        """Recursively collect tree statistics"""
        stats = {
            "program_id": node.program.id,
            "depth": depth,
            "visits": node.visits,
            "avg_score": node.avg_score,
            "best_score": node.best_score,
            "num_children": len(node.children),
            "children": [
                self._collect_tree_stats(child, depth + 1)
                for child in node.children
            ]
        }

        return stats
