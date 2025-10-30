"""
Search strategies for OpenEvolve

This module provides different search/evolution strategies that can be used
with OpenEvolve's infrastructure (prompt building, parallel execution, etc.)
"""

from openevolve.search_strategies.base_strategy import SearchStrategy
from openevolve.search_strategies.map_elites_strategy import MAPElitesStrategy
from openevolve.search_strategies.best_of_n_strategy import BestOfNStrategy
from openevolve.search_strategies.beam_search_strategy import BeamSearchStrategy
from openevolve.search_strategies.mcts_strategy import MCTSStrategy

__all__ = [
    "SearchStrategy",
    "MAPElitesStrategy",
    "BestOfNStrategy",
    "BeamSearchStrategy",
    "MCTSStrategy",
]
