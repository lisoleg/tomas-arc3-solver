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
from .game_profiles import GameProfile, GameProfileRegistry, HybridGameProfile
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

# ── v3.16.0 — Neural-Inspired DSL (神经启发架构: LSTM/ResNet/Transformer/Hopfield) ──
from .neural_dsl import (
    RecurrentDSL,
    GatedDSL,
    AdvancedDSL,
    inject_interneurons,
    estimate_ic_with_motifs,
    neuro_inspired_kps_search,
    neurally_gated_expand,
)

# ── v3.17.0 — 天行相变 + YinLong 八元数 + DOGA + PTS孤子判别 ──
from .neural_dsl import (
    YinLongDSL,
    TianxingGaussExVerifier,
    is_soliton,
    is_soliton_from_sprite,
    doga_score,
    solve_tianxing,
    estimate_order_anchoring,
    estimate_alienation,
    estimate_structural_phase,
    yinlong_tensor_product,
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

# ── v3.18.0 — 四层混合搜索架构 (Hybrid Search Pipeline) ──
from .hybrid_search_engine import (
    CandidateSet,
    EvaluatedCandidateSet,
    PipelineStrategies,
    L1CandidateGenerator,
    L2StructurePruner,
    L3ResidualEvaluator,
    L4DecisionSelector,
    HybridSearchPipeline,
)
from .l1_strategies import (
    BFSPathCandidateGenerator,
    DFSEnumerationCandidateGenerator,
    WallBFSCandidateGenerator,
    DeltaStateReplayCandidateGenerator,
    DirectComputeCandidateGenerator,
)
from .l2_strategies import (
    PassThroughPruner,
    ComboSymmetryPruner,
    PrimeSignaturePruner,
    MatroidConstraintPruner,
)
from .l3_strategies import (
    GaussExEvaluation,
    KappaSnapEvaluation,
    DeadZeroFuseEvaluation,
    AsymIndexEvaluation,
    PassThroughEvaluation,
)
from .l4_strategies import (
    KappaSelector,
    LiuSelector,
)
from .kappa_selector import (
    KappaEtaAscendSelector,
    LiuPrioritySelector,
    KAPPA_DELTA_K,
    KAPPA_MIN_CONFIDENCE,
)
from .wall_bfs import (
    WallBFSEngine,
    WALL_BFS_MAX_DEPTH,
    WALL_BFS_MAX_NODES,
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
    "HybridGameProfile",
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
    # v3.16.0 — Neural-Inspired DSL
    "RecurrentDSL",
    "GatedDSL",
    "AdvancedDSL",
    "inject_interneurons",
    "estimate_ic_with_motifs",
    "neuro_inspired_kps_search",
    "neurally_gated_expand",
    # v3.17.0 — 天行相变 + YinLong 八元数 + DOGA + PTS孤子判别
    "YinLongDSL",
    "TianxingGaussExVerifier",
    "is_soliton",
    "is_soliton_from_sprite",
    "doga_score",
    "solve_tianxing",
    "estimate_order_anchoring",
    "estimate_alienation",
    "estimate_structural_phase",
    "yinlong_tensor_product",
    # MetaSnapNet κ-Snap Beam Scoring (SPP集成)
    "TopoFeatureExtractor",
    "ProgramNodeFeatureExtractor",
    "MetaSnapNet",
    "MetaSnapBeamScorer",
    "MetaSnapTrainingExample",
    "MetaSnapDataCollector",
    "SPPTrainer",
    # v3.18.0 — 四层混合搜索架构 (Hybrid Search Pipeline)
    "CandidateSet",
    "EvaluatedCandidateSet",
    "PipelineStrategies",
    "L1CandidateGenerator",
    "L2StructurePruner",
    "L3ResidualEvaluator",
    "L4DecisionSelector",
    "HybridSearchPipeline",
    # v3.18.0 — L1候选生成策略
    "BFSPathCandidateGenerator",
    "DFSEnumerationCandidateGenerator",
    "WallBFSCandidateGenerator",
    "DeltaStateReplayCandidateGenerator",
    "DirectComputeCandidateGenerator",
    # v3.18.0 — L2结构剪枝策略
    "PassThroughPruner",
    "ComboSymmetryPruner",
    "PrimeSignaturePruner",
    "MatroidConstraintPruner",
    # v3.18.0 — L3残差评估策略
    "GaussExEvaluation",
    "KappaSnapEvaluation",
    "DeadZeroFuseEvaluation",
    "AsymIndexEvaluation",
    "PassThroughEvaluation",
    # v3.18.0 — L4决策融合策略
    "KappaSelector",
    "LiuSelector",
    # v3.18.0 — κ-优选η升序核心
    "KappaEtaAscendSelector",
    "LiuPrioritySelector",
    "KAPPA_DELTA_K",
    "KAPPA_MIN_CONFIDENCE",
    # v3.18.0 — Wall-BFS引擎
    "WallBFSEngine",
    "WALL_BFS_MAX_DEPTH",
    "WALL_BFS_MAX_NODES",
]
