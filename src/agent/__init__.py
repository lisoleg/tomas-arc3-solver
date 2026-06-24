"""TOMAS Interactive Agent for ARC-AGI-3.

Provides the TomasAgent class that implements the ARC-AGI-3 Agent interface.
Also provides ExplorerAgent (graph-based), DopamineExplorer (RL-driven),
and PlannerAgent (hybrid planner-learner V5).

Grid perception and game profiles for non-oracle mode are also exported,
enabling the agent to work on all 25 games without env._game access.
"""

from .tomas_agent import TomasAgent
from .graph_explorer import ExplorerAgent
from .dopamine_explorer import DopamineExplorer
from .planner_agent import PlannerAgent
from .grid_perception import GridPerception
from .game_profiles import GameProfile, GameProfileRegistry
from .oracle_adapters import (
    GameEntity,
    OracleAdapter,
    LS20Adapter,
    TR87Adapter,
    FT09Adapter,
    ADAPTER_REGISTRY,
    get_oracle_adapter,
    auto_detect_adapter,
)

__all__ = [
    "TomasAgent",
    "ExplorerAgent",
    "DopamineExplorer",
    "PlannerAgent",
    "GridPerception",
    "GameProfile",
    "GameProfileRegistry",
    "GameEntity",
    "OracleAdapter",
    "LS20Adapter",
    "TR87Adapter",
    "FT09Adapter",
    "ADAPTER_REGISTRY",
    "get_oracle_adapter",
    "auto_detect_adapter",
]
