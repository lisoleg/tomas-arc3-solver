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
from .universal_solver_pipeline import UniversalSolverPipeline
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
    PsiAuditor,
    PsiAuditEntry,
    ConditionalDeltaTDiscovery,
    ConditionalRule,
)
from .tomas_learner import (
    TOMASLearner,
    CausalPatternExtractor,
    DSLMacroAbstractor,
    LibraryManager,
    MacroCandidate,
    ActionTrace,
    EpisodeTrace,
    OnlinePSIAudit,
    PhysicalNARConv,
    GaussExGuard,
    FastPathDispatcher,
    extract_topo_features,
    extract_topo_features_vec,
    # v3.6.0 — 物理紧化 + Ψ-截断 + 进化增益
    PhysicalGaussExGuard,
    PsiCutController,
    compute_scaling_law_v36,
    classify_task_complexity,
    # v3.7.0 — 流贯归约框架
    PhysicalCompactificationReduction,
    # v3.8.0 — Thinker-Performer双轨 + Interactive ZKP Loop
    ThinkerPerformerPipeline,
    InteractiveZKPLoop,
    PersistentKVCache,
    # v3.9.0 — Sleep-Step Algorithm 1 + Abductive Lift + Ω_topo Trigger + MoonshineProver
    MoonshineProver,
    check_breakthrough_and_trigger,
    # v3.10.0 — 博弈即降维 + 神经流形与因果紧化 (6 new concepts)
    GibbsEnsemble,
    IDOFramework,
    QuantumContextualEvaluator,
    CHLIsomorphism,
    compute_effective_search_depth,
    UniverseZKP,
    # v3.11.0 — 八元数熵重整化 + IDO-Agent (7 new concepts)
    Octonion,
    KappaAlgorithmOperator,
    LogRenormalizationMachine,
    IDONPEPReduction,
    IDOVonNeumannEntropy,
    MaximumEntropyReduction,
    IDOAgent,
    # v3.12.0 — 局部质量贝叶斯 + 智能体记忆 + 耦合振子 (8 new concepts)
    LocalMassBayesianInference,
    REKLDirectionalEvaluator,
    IDOGaussExVariationalObjective,
    TOMASMemoryArchive,
    MemoryUnit,
    ICMetric,
    PsiCutAwareMerge,
    KuramotoOscillator,
    PhysicalGaussExConstraint,
)

# ── v3.13.0 — κ-Priority Search (κ-PS) from game_solvers ──
from .game_solvers import (
    solve_kappa_priority_search,
    _estimate_ic_game_state,
    _compute_gex_residual,
)

# ── v3.15.0 — Δ-State Engine + RHAE Budget Controller ──
from .delta_state import (
    SolverAborted,
    BudgetExceeded,
    Node,
    ActionSpace,
    ReplayEngine,
    LayoutHasher,
    GaussExVerifier,
    compute_coin_proximity_position,
    structural_bfs,
    parametric_bfs,
    kappa_priority_refine,
    MAX_RHAE_PER_TASK,
    GEX_PASS_THRESHOLD,
    GEX_FAIL_THRESHOLD,
    LS20_BUDGET_MULT,
    DEFAULT_MAX_DEPTH,
    MAX_REPLAY_CACHE,
    STRUCT_BFS_BUDGET,
    PARAM_BFS_BUDGET,
    MIN_RHAE_TO_KEEP,
    ABORT_RHAE_THRESHOLD,
    DEAD_ZERO_RATIO,
)

from .rhae_controller import (
    CoinCollector,
    RHAEBudgetController,
    create_game_task,
    ls20_estimate_human_steps,
)

# ── MetaSnapNet可选集成 ──
try:
    from .meta_snap_net import (
        TopoFeatureExtractor,
        ProgramNodeFeatureExtractor,
        MetaSnapNet,
        MetaSnapBeamScorer,
        MetaSnapTrainingExample,
        MetaSnapDataCollector,
        SPPTrainer,
        HAS_TORCH as META_SNAP_HAS_TORCH,
    )
except ImportError:
    # meta_snap_net.py不可用时, 导出空值
    TopoFeatureExtractor = None  # type: ignore
    ProgramNodeFeatureExtractor = None  # type: ignore
    MetaSnapNet = None  # type: ignore
    MetaSnapBeamScorer = None  # type: ignore
    MetaSnapTrainingExample = None  # type: ignore
    MetaSnapDataCollector = None  # type: ignore
    SPPTrainer = None  # type: ignore
    META_SNAP_HAS_TORCH = False

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
    "UniversalSolverPipeline",
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
    # P1-1/P1-2: NARLA audit modules
    "PsiAuditor",
    "PsiAuditEntry",
    # P1-8: Conditional ΔT Discovery
    "ConditionalDeltaTDiscovery",
    "ConditionalRule",
    # TOMAS Sleep-Step Learning
    "TOMASLearner",
    "CausalPatternExtractor",
    "DSLMacroAbstractor",
    "LibraryManager",
    "MacroCandidate",
    "ActionTrace",
    "EpisodeTrace",
    # 拓扑饱和修正 v3.5.0 — OnlinePSIAudit + PhysicalNARConv
    "OnlinePSIAudit",
    "PhysicalNARConv",
    "GaussExGuard",
    "FastPathDispatcher",
    "extract_topo_features",
    "extract_topo_features_vec",
    # v3.6.0 — 物理紧化 + Ψ-截断 + 进化增益
    "PhysicalGaussExGuard",
    "PsiCutController",
    "compute_scaling_law_v36",
    "classify_task_complexity",
    # v3.7.0 — 流贯归约框架
    "PhysicalCompactificationReduction",
    # v3.8.0 — Thinker-Performer双轨 + Interactive ZKP Loop
    "ThinkerPerformerPipeline",
    "InteractiveZKPLoop",
    "PersistentKVCache",
    # v3.9.0 — Sleep-Step Algorithm 1 + Abductive Lift + Ω_topo Trigger + MoonshineProver
    "MoonshineProver",
    "check_breakthrough_and_trigger",
    # v3.10.0 — 博弈即降维 + 神经流形与因果紧化 (6 new concepts)
    "GibbsEnsemble",
    "IDOFramework",
    "QuantumContextualEvaluator",
    "CHLIsomorphism",
    "compute_effective_search_depth",
    "UniverseZKP",
    # v3.11.0 — 八元数熵重整化 + IDO-Agent (7 new concepts)
    "Octonion",
    "KappaAlgorithmOperator",
    "LogRenormalizationMachine",
    "IDONPEPReduction",
    "IDOVonNeumannEntropy",
    "MaximumEntropyReduction",
    "IDOAgent",
    # v3.12.0 — 局部质量贝叶斯 + 智能体记忆 + 耦合振子 (8 new concepts)
    "LocalMassBayesianInference",
    "REKLDirectionalEvaluator",
    "IDOGaussExVariationalObjective",
    "TOMASMemoryArchive",
    "MemoryUnit",
    "ICMetric",
    "PsiCutAwareMerge",
    "KuramotoOscillator",
    "PhysicalGaussExConstraint",
    # v3.13.0 — κ-Priority Search (κ-PS) from game_solvers
    "solve_kappa_priority_search",
    "_estimate_ic_game_state",
    "_compute_gex_residual",
    # v3.15.0 — Δ-State Engine (IDO 流贯)
    "SolverAborted",
    "BudgetExceeded",
    "Node",
    "ActionSpace",
    "ReplayEngine",
    "LayoutHasher",
    "GaussExVerifier",
    "compute_coin_proximity_position",
    "structural_bfs",
    "parametric_bfs",
    "kappa_priority_refine",
    "MAX_RHAE_PER_TASK",
    "GEX_PASS_THRESHOLD",
    "GEX_FAIL_THRESHOLD",
    "LS20_BUDGET_MULT",
    "DEFAULT_MAX_DEPTH",
    "MAX_REPLAY_CACHE",
    "STRUCT_BFS_BUDGET",
    "PARAM_BFS_BUDGET",
    "MIN_RHAE_TO_KEEP",
    "ABORT_RHAE_THRESHOLD",
    "DEAD_ZERO_RATIO",
    # v3.15.0 — RHAE Budget Controller
    "CoinCollector",
    "RHAEBudgetController",
    "create_game_task",
    "ls20_estimate_human_steps",
    # MetaSnapNet κ-Snap Beam Scoring (SPP集成)
    "TopoFeatureExtractor",
    "ProgramNodeFeatureExtractor",
    "MetaSnapNet",
    "MetaSnapBeamScorer",
    "MetaSnapTrainingExample",
    "MetaSnapDataCollector",
    "SPPTrainer",
]
