"""
Custom Search Strategies for Code Evolution

This module implements alternative search strategies for evolving code:
- Best of N: N parallel lineages with independent evolution
- Beam Search: Beam search with branching and pruning
- MCTS: Monte Carlo Tree Search for code evolution
"""

from .best_of_n import BestOfNSearch
from .beam_search import BeamSearch
from .mcts_search import MCTSSearch

__all__ = ["BestOfNSearch", "BeamSearch", "MCTSSearch"]
