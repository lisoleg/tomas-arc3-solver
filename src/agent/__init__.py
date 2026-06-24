"""TOMAS Interactive Agent for ARC-AGI-3.

Provides the TomasAgent class that implements the ARC-AGI-3 Agent interface.
Also provides ExplorerAgent (graph-based), DopamineExplorer (RL-driven),
and PlannerAgent (hybrid planner-learner V5).

Grid perception and game profiles for non-oracle mode are also exported,
enabling the agent to work on all 25 games without env._game access.

Universal Oracle adapter enables Oracle mode for any game by
auto-discovering entities from env._game attributes.

Self-learning modules (AfterActionReview, CognitiveRecursiveDynamics,
OperatorAccumulator) provide episode analysis, cognitive reasoning,
and reusable operator accumulation for continuous improvement.
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
    get_universal_adapter,
)
from .universal_oracle_adapter import UniversalOracleAdapter
from .self_learning import (
    AfterActionReview,
    CognitiveRecursiveDynamics,
    OperatorAccumulator,
    Trajectory,
    Outcome,
    Pattern,
    AntiPattern,
    Lesson,
    ReviewResult,
    CognitiveResult,
    Operator,
)

__all__ = [
    # Core agents
    "TomasAgent",
    "ExplorerAgent",
    "DopamineExplorer",
    "PlannerAgent",
    # Perception and profiles
    "GridPerception",
    "GameProfile",
    "GameProfileRegistry",
    # Oracle adapters
    "GameEntity",
    "OracleAdapter",
    "LS20Adapter",
    "TR87Adapter",
    "FT09Adapter",
    "UniversalOracleAdapter",
    "ADAPTER_REGISTRY",
    "get_oracle_adapter",
    "get_universal_adapter",
    "auto_detect_adapter",
    # Self-learning modules
    "AfterActionReview",
    "CognitiveRecursiveDynamics",
    "OperatorAccumulator",
    "Trajectory",
    "Outcome",
    "Pattern",
    "AntiPattern",
    "Lesson",
    "ReviewResult",
    "CognitiveResult",
    "Operator",
]
