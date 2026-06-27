# LS20 RG-Flow Solver 系统架构设计文档

> **项目**: LS20 RG-Flow Solver Refactor — 将 LS20 solver 从贪婪迭代架构重构为 RG-Flow 预规划架构
> **架构师**: 高见远 (Gao)
> **版本**: v5.0 (对应 PRD v5.0)
> **日期**: 2025-07-01
> **技术栈**: Python 3.11+ (无新依赖)
> **仓库**: `tomas-arc3-solver`

---

## 目录

- [Part A: 系统设计](#part-a-系统设计)
  - [1. 实现方案 + 框架选型](#1-实现方案--框架选型)
  - [2. 文件列表及相对路径](#2-文件列表及相对路径)
  - [3. 数据结构和接口（类图）](#3-数据结构和接口类图)
  - [4. 程序调用流程（时序图）](#4-程序调用流程时序图)
  - [5. 待明确事项](#5-待明确事项)
- [Part B: 任务分解](#part-b-任务分解)
  - [6. 依赖包列表](#6-依赖包列表)
  - [7. 任务列表](#7-任务列表)
  - [8. 共享知识](#8-共享知识)
  - [9. 任务依赖图](#9-任务依赖图)

---

# Part A: 系统设计

## 1. 实现方案 + 框架选型

### 1.1 核心技术挑战

| # | 挑战 | 当前瓶颈 | RG-Flow解决方案 |
|---|------|----------|----------------|
| C1 | **贪婪迭代浪费步数** | L1 iter0-4 贪婪coin收集浪费≈40步 | Anti-Monotonicity Budget Control: coin仅在κ-reduction > cost时收集 |
| C2 | **BFS搜索空间无限** | `_solve_ls20_delta_state_bfs` 使用O(N²) BFS | Coset-Prioritized Navigation: 330陪集×D₄=2640有界路径 |
| C3 | **无全局时间规划** | L1 45s预算耗尽在pos=(14,15) state matched | RG-Flow β(κ)驱动时间分配: 5s UV提取 + 8s coset搜索 + 执行 |
| C4 | **状态路径无预计算** | changer触发后逐步决策(走哪条路、何时收集) | StatePathSolver预计算最优(position,state)序列 |
| C5 | **无Level分类** | L0和L2使用相同策略，效率低 | Sporadic Group Filter: 4类对称路由(Niemeier/Möbius/Monster/Generic) |
| C6 | **IDO模块接口不匹配** | 10个理论模块为ARC-3通用设计，非LS20专用 | LS20→IDO适配层: game→IDO数据转换 |

### 1.2 渐进替代方案（Q1选B）

**关键设计决策**: 仅替换 Stage 2 (`_solve_ls20_delta_state_bfs`)，保留 Stage 1 和 Stage 3 作为 fallback。

```
当前管线 (solve_ls20):
  Stage1: κ-gradient oracle replay (5s, 贪婪迭代)      ─── 保留
  Stage2: _solve_ls20_delta_state_bfs (45s, 贪婪迭代BFS) ─── 替换为 RG-Flow Preplan
  Stage3: κ-gradient direct / κ-PS BFS fallback           ─── 保留

目标管线 (solve_ls20 v5.0):
  Stage1: κ-gradient oracle replay (5s)                   ─── 保留
  Stage2: RG-Flow Preplan (UV→IR坍缩)                     ─── 新增
  Stage3: κ-gradient direct / κ-PS BFS fallback           ─── 保留 (RG-Flow失败时激活)
```

**渐进替代的优势**:
1. **风险可控**: RG-Flow失败时自动fallback到Stage3，不比当前更差
2. **可观测**: Stage1仍作为快速探测，提供UV特征初筛
3. **回归安全**: 已通过的L0 (RHAE=115) 不会退化

### 1.3 框架选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 核心语言 | Python 3.11+ | 项目已有技术栈，无引入新语言理由 |
| 数值计算 | NumPy (已有) | κ-entropy计算、特征向量操作 |
| 游戏接口 | arcengine.ActionInput (已有) | LS20的action接口，无需改变 |
| 适配层 | LS20IDOAdapter (新增) | game→IDO数据格式转换，桥接理论模块 |
| 验证引擎 | delta_state.ReplayEngine (已有) | 零拷贝路径验证，复用P1-4 |
| Level分类 | SporadicGroupFilter (已有) | 4类对称路由，需LS20适配 |
| **新依赖** | **无** | 10个理论模块已import验证通过，3个新模块纯Python |

### 1.4 分段预规划方案（Q2选B）

L5/L6 baseline需192/186步，超长路径处理策略：

```
Segmented RG-Flow (分段坍缩):
  Segment 1: start→changer₁ (κ坍缩段1)
  Segment 2: changer₁→changer₂ (κ坍缩段2)
  Segment 3: changer₂→goal (κ坍缩段3)

每段独立:
  - 独立κ-entropy校验 (Δκ_segment ≥ Δκ理论值/段数)
  - 独立Anti-Monotonicity剪枝
  -段间coin收集决策 (κ-reduction > cost才收集)
```

### 1.5 Anti-Monotonicity tolerance（Q3选B）

```python
# 允许5%噪声容忍，避免错杀有效路径
IDOConfig(strict=False, tolerance=0.05)
```

### 1.6 一次性Level分类（Q4选A）

```python
# EML感知在plan前一次性执行，不再每步重新检测
level_type = SporadicGroupFilter.classify_sporadic_group(game_grid)
# level_type ∈ {"Niemeier", "Moebius", "Monster", "Generic"}
# → 路由到对应求解策略
```

### 1.7 LS20→IDO适配层（Q5选B）

10个理论模块为ARC-3通用设计，需要适配层将LS20 game数据转换为IDO理论模块输入格式。

---

## 2. 文件列表及相对路径

### 2.1 新增文件

```
tomas-arc3-solver/src/agent/
├── rg_flow_planner.py              # NEW ~400行 — RG-Flow预规划主入口
├── state_path_solver.py            # NEW ~300行 — 状态路径序列生成
├── kappa_budget_controller.py      # NEW ~200行 — Anti-Monotonicity预算控制
├── ls20_ido_adapter.py             # NEW ~150行 — LS20→IDO数据适配层
```

### 2.2 修改文件

```
tomas-arc3-solver/src/agent/
├── game_solvers.py                 # MODIFIED — solve_ls20() Stage2替换
├── __init__.py                     # MODIFIED — 新增模块import
├── game_configs.py                 # MODIFIED — 新增RG-Flow配置常量
```

### 2.3 已有理论模块（不变，仅被调用）

```
tomas-arc3-solver/src/agent/
├── rg_flow.py                      # 478行 — RG Flow β函数 (被rg_flow_planner调用)
├── coset_search.py                 # 688行 — 陪集优先搜索 (被state_path_solver调用)
├── ido_axioms.py                   # 550行 — 反单调性+κ-entropy (被kappa_budget_controller调用)
├── sporadic_group_filter.py        # 718行 — 散在群分类 (被rg_flow_planner调用)
├── kappa_winding.py                # 522行 — κ-Winding相位 (P2调试可视化)
├── semantic_constants.py           # 445行 — 语义常数 (被ls20_ido_adapter调用)
├── ramanujan_conjecture.py         # 770行 — Ramanujan猜想 (P2数列模式)
├── octonion_phase.py               # 598行 — 八元数相位 (P1多触发规划)
├── eml_perceiver.py                # 897行 — EML感知器 (被ls20_ido_adapter调用)
├── hungarian_align.py              # 344行 — Hungarian对齐 (P1状态匹配)
├── delta_state.py                  # 1174行 — ReplayEngine零拷贝 (被state_path_solver调用)
├── oracle_adapters.py              # 1287行 — LS20Adapter (被ls20_ido_adapter调用)
```

### 2.4 测试文件

```
tomas-arc3-solver/tests/
├── test_rg_flow_planner.py         # NEW — RG-Flow预规划测试
├── test_state_path_solver.py       # NEW — 状态路径测试
├── test_kappa_budget_controller.py # NEW — 预算控制测试
├── test_ls20_ido_adapter.py        # NEW — IDO适配层测试
```

### 2.5 架构文档

```
tomas-arc3-solver/docs/architecture/
├── system_design.md                # NEW — 本文档
├── class-diagram.mermaid           # NEW — 类图
├── sequence-diagram.mermaid        # NEW — 时序图
```

---

## 3. 数据结构和接口（类图）

> 完整类图见 `docs/architecture/class-diagram.mermaid`

### 3.1 RGFlowPlanner — RG-Flow预规划主入口

```python
class RGFlowPlanner:
    """RG-Flow预规划器 — UV→IR坍缩求解LS20

    替代 _solve_ls20_delta_state_bfs 的贪婪迭代架构。
    Phase1: UV→IR预规划 → Phase2: 一次性执行

    调用链:
      plan() → _uv_feature_extraction() → _coset_state_search()
              → _anti_monotonicity_prune() → _execute_plan()
    """

    def __init__(
        self,
        ido_config: IDOConfig = IDOConfig(strict=False, tolerance=0.05),
        max_total_time: float = 45.0,
        max_bfs_time: float = 8.0,
    ) -> None:
        """初始化预规划器。

        Args:
            ido_config: IDO反单调性配置 (tolerance=0.05噪声容忍)
            max_total_time: 总时间预算 (秒)
            max_bfs_time: 单次coset搜索时间上限 (秒)
        """
        ...

    def plan(self, game: Any, level_idx: int) -> list[tuple] | None:
        """RG-Flow预规划主入口。

        渐进替代 Stage2: 调用此方法替换 _solve_ls20_delta_state_bfs。

        Args:
            game: ARC-AGI-3游戏对象
            level_idx: 关卡索引 (0-6)

        Returns:
            动作序列 [(direction, data), ...] 或 None (预规划失败)
        """
        ...

    def _uv_feature_extraction(self, game: Any, adapter: LS20Adapter) -> 'UVFeatures':
        """从EML感知器提取11维UV特征。

        调用 eml_perceiver.EMLPerceiver + ls20_ido_adapter.LS20IDOAdapter

        Args:
            game: 游戏对象
            adapter: LS20Adapter (位置/状态提取)

        Returns:
            UVFeatures — 11维特征向量
        """
        ...

    def _classify_level(self, game: Any, adapter: LS20Adapter) -> str:
        """Sporadic Group Level分类 (一次性, Q4选A)。

        调用 sporadic_group_filter.classify_sporadic_group()

        Returns:
            level_type ∈ {"Niemeier", "Moebius", "Monster", "Generic"}
        """
        ...

    def _coset_state_search(
        self,
        uv_features: 'UVFeatures',
        goal_state: 'StateTuple',
        level_type: str,
        game: Any,
        adapter: LS20Adapter,
    ) -> 'StatePath | None':
        """330陪集×D₄=2640路径搜索最优(position,state)序列。

        调用 coset_search.CosetPrioritizedSearch + state_path_solver.StatePathSolver

        Args:
            uv_features: UV特征 (11维)
            goal_state: 目标状态 (rotation, color, shape)
            level_type: Level分类类型
            game: 游戏对象 (用于BFS路径验证)
            adapter: LS20Adapter

        Returns:
            StatePath — 最优状态路径序列，或 None
        """
        ...

    def _anti_monotonicity_prune(
        self,
        candidates: list['StatePath'],
        current_kappa: float,
    ) -> list['StatePath']:
        """Anti-Monotonicity剪枝: κ增加的候选路径立即剪枝。

        调用 ido_axioms.verify_anti_monotonicity() + kappa_budget_controller

        Args:
            candidates: 候选状态路径列表
            current_kappa: 当前信息基数κ

        Returns:
            剪枝后的有效候选列表
        """
        ...

    def _execute_plan(
        self,
        game: Any,
        state_path: 'StatePath',
        adapter: LS20Adapter,
    ) -> list[tuple]:
        """按预规划一次性执行，coin仅在κ-reduction > cost时收集。

        调用 kappa_budget_controller.KappaBudgetController.should_collect_coin()

        Args:
            game: 游戏对象
            state_path: 预计算的状态路径
            adapter: LS20Adapter

        Returns:
            动作序列 [(direction, data), ...]
        """
        ...

    def _segmented_preplan(
        self,
        game: Any,
        level_idx: int,
        adapter: LS20Adapter,
        n_segments: int = 3,
    ) -> list[tuple] | None:
        """分段预规划 (L5/L6超长路径)。

        每段独立κ-entropy校验 + Anti-Monotonicity剪枝 + coin决策。
        段间通过changer触发衔接。

        Args:
            game: 游戏对象
            level_idx: 关卡索引 (5或6)
            adapter: LS20Adapter
            n_segments: 分段数量 (默认3)

        Returns:
            动作序列 或 None
        """
        ...

    def _verify_kappa_release(
        self,
        state_path: 'StatePath',
        expected_delta_kappa: float,
    ) -> bool:
        """κ-Entropy GaussEx校验。

        调用 ido_axioms.kappa_entropy_release()

        Args:
            state_path: 状态路径
            expected_delta_kappa: 预期Δκ理论值

        Returns:
            True if Δκ ≥ expected_delta_kappa
        """
        ...
```

### 3.2 StatePathSolver — 状态路径序列生成

```python
@dataclass
class StateTuple:
    """LS20 状态元组 (position + rotation/color/shape)"""
    position: tuple[int, int]   # (x, y) 像素坐标
    rotation: int               # rotation状态 (0-3)
    color: int                  # color状态
    shape: int                  # shape状态

@dataclass
class ChangerStep:
    """changer触发步骤"""
    changer_pos: tuple[int, int]    # changer位置
    triggers_needed: int            # 需要触发次数 (rotation 0→3需3次)
    leave_return_path: list[tuple]  # 离开-返回路径动作序列
    kappa_reduction: float          # 该步骤的κ-reduction值

@dataclass
class StatePath:
    """预计算的状态路径序列"""
    start_state: StateTuple             # 起始状态
    goal_state: StateTuple              # 目标状态
    changer_sequence: list[ChangerStep] # changer触发序列
    total_kappa: float                  # 总κ-release
    plan_actions: list[tuple]           # 完整动作序列 [(dir, data), ...]
    level_type: str                     # Level分类类型
    is_segmented: bool                  # 是否为分段预规划 (L5/L6)
    segment_index: int                  # 分段索引 (0=第一段)

class StatePathSolver:
    """状态路径序列生成器

    核心职责:
    1. 从UV特征+goal状态计算最优(position,state)序列
    2. 多触发路径: changer需N次触发时计算最优leave-and-return
    3. 路径BFS: 两点间移动路径生成
    """

    def __init__(
        self,
        coset_search_engine: 'CosetPrioritizedSearch',
        replay_engine: 'ReplayEngine',
        hungarian_align_fn: Callable,
    ) -> None:
        """初始化路径求解器。

        Args:
            coset_search_engine: 陪集优先搜索引擎 (已有模块)
            replay_engine: ReplayEngine零拷贝验证 (已有模块)
            hungarian_align_fn: Hungarian对齐函数 (已有模块)
        """
        ...

    def compute_state_path_sequence(
        self,
        start_state: StateTuple,
        goal_state: StateTuple,
        changer_positions: list[tuple[int, int]],
        changer_effects: dict[tuple, dict],
        game: Any,
        adapter: LS20Adapter,
    ) -> StatePath | None:
        """计算最优状态路径序列。

        核心算法:
        1. 分析start→goal需要哪些状态维度变化
        2. 确定哪些changer可以提供所需变化
        3. 觡划changer访问顺序 (κ-reduction优先)
        4. 生成完整(position,state)序列

        Args:
            start_state: 起始状态元组
            goal_state: 目标状态元组
            changer_positions: 所有changer位置
            changer_effects: changer效果映射 {(x,y): {"rot": delta, "color": delta}}
            game: 游戏对象
            adapter: LS20Adapter

        Returns:
            StatePath 或 None (路径不可达)
        """
        ...

    def compute_changer_trigger_sequence(
        self,
        current_state: StateTuple,
        target_state: StateTuple,
        changer_pos: tuple[int, int],
        changer_effect: dict,
        triggers_needed: int,
        game: Any,
        adapter: LS20Adapter,
    ) -> list[ChangerStep]:
        """计算changer多次触发的leave-and-return路径。

        例如: rotation 0→3 需要3次触发
          → 离开changer→再返回×3

        Args:
            current_state: 当前状态
            target_state: 目标维度状态
            changer_pos: changer位置
            changer_effect: changer效果
            triggers_needed: 需要触发次数
            game: 游戏对象
            adapter: LS20Adapter

        Returns:
            ChangerStep序列
        """
        ...

    def compute_leave_and_return_path(
        self,
        changer_pos: tuple[int, int],
        game: Any,
        adapter: LS20Adapter,
        retreat_steps: int = 1,
    ) -> list[tuple]:
        """计算离开changer再返回的最短路径。

        Args:
            changer_pos: changer位置
            game: 游戏对象
            adapter: LS20Adapter
            retreat_steps: 撤退步数

        Returns:
            动作序列 [(dir, data), ...]
        """
        ...

    def compute_bfs_path(
        self,
        start_pos: tuple[int, int],
        target_pos: tuple[int, int],
        game: Any,
        adapter: LS20Adapter,
        avoid_positions: set[tuple[int, int]] | None = None,
        max_steps: int = 50,
    ) -> list[tuple] | None:
        """两点间Wall-Map BFS路径 (复用_lightweight_bfs)。

        Args:
            start_pos: 起始位置
            target_pos: 目标位置
            game: 游戏对象
            adapter: LS20Adapter
            avoid_positions: 需要避开的position集合
            max_steps: 最大BFS步数

        Returns:
            动作序列 或 None
        """
        ...

    def verify_path_on_replay_engine(
        self,
        game: Any,
        actions: list[tuple],
        target_state: StateTuple,
    ) -> bool:
        """用ReplayEngine零拷贝验证路径是否可达目标状态。

        Args:
            game: 游戏对象
            actions: 动作序列
            target_state: 预期目标状态

        Returns:
            True if 路径验证成功且状态匹配
        """
        ...

    def compute_segmented_paths(
        self,
        start_state: StateTuple,
        goal_state: StateTuple,
        changer_sequence: list[ChangerStep],
        n_segments: int,
        game: Any,
        adapter: LS20Adapter,
    ) -> list[StatePath]:
        """分段路径生成 (L5/L6超长路径)。

        将长路径拆分为多个独立段:
          Segment[i]: changer_i → changer_{i+1} (段间κ独立校验)

        Args:
            start_state: 起始状态
            goal_state: 目标状态
            changer_sequence: changer访问序列
            n_segments: 分段数量
            game: 游戏对象
            adapter: LS20Adapter

        Returns:
            分段StatePath列表
        """
        ...
```

### 3.3 KappaBudgetController — Anti-Monotonicity预算控制

```python
@dataclass
class CoinDecision:
    """coin收集决策结果"""
    should_collect: bool        # 是否收集
    kappa_reduction: float      # κ-reduction值 (收集带来的信息压缩)
    action_cost: float          # action_cost (收集需要的步数代价)
    coin_position: tuple[int, int]  # coin位置
    reason: str                 # 决策理由

@dataclass
class TimeBudget:
    """时间预算分配"""
    total_budget: float         # 总时间预算 (秒)
    uv_extraction_time: float   # UV特征提取时间分配
    coset_search_time: float    # coset搜索时间分配
    execution_time: float       # 执行时间分配
    verification_time: float    # κ-verification时间分配
    fallback_reserve: float     # fallback保留时间

class KappaBudgetController:
    """Anti-Monotonicity预算控制 — coin收集决策 + 时间预算分配

    核心原则: κ必须单调递减(容忍0.05噪声)
    → coin仅在κ-reduction > action-cost时收集
    → 时间预算按β(κ)驱动分配
    """

    def __init__(
        self,
        ido_config: IDOConfig = IDOConfig(strict=False, tolerance=0.05),
        total_time_budget: float = 45.0,
    ) -> None:
        """初始化预算控制器。

        Args:
            ido_config: IDO配置 (tolerance=0.05)
            total_time_budget: 总时间预算 (秒)
        """
        ...

    def should_collect_coin(
        self,
        current_kappa: float,
        coin_cost: float,
        kappa_reduction: float,
        remaining_steps: int,
        min_reserve: int = 5,
    ) -> CoinDecision:
        """Anti-Monotonicity coin收集决策。

        核心规则: κ-reduction > action_cost 时才收集

        L1预期:
          - changer→goal路径中κ-reduction=0, action-cost>0 → SKIP
          - 只有真正需要步数重置且κ-reduction>cost时才收集

        Args:
            current_kappa: 当前κ值
            coin_cost: 收集coin需要的步数代价
            kappa_reduction: 收集coin带来的κ-reduction
            remaining_steps: 剩余步数
            min_reserve: 最低步数储备 (默认5)

        Returns:
            CoinDecision — 是否收集 + 决策理由
        """
        ...

    def allocate_time_budget(
        self,
        level_idx: int,
        level_type: str,
        path_length_estimate: int,
    ) -> TimeBudget:
        """β(κ)驱动时间预算分配。

        时间分配策略:
          - UV提取: ~5s (EML感知11维特征)
          - coset搜索: ~8s (2640有限路径)
          - 执行: 按路径长度×步耗时估计
          - κ-verification: ~2s (GaussEx校验)
          - fallback保留: 5-10s (Stage3 fallback)

        Args:
            level_idx: 关卡索引
            level_type: Level分类类型
            path_length_estimate: 预估路径长度

        Returns:
            TimeBudget — 时间预算分配
        """
        ...

    def compute_kappa_reduction(
        self,
        before_kappa: float,
        after_kappa: float,
    ) -> float:
        """计算κ-reduction值。

        Args:
            before_kappa: 执行前κ值
            after_kappa: 执行后κ值

        Returns:
            κ-reduction = before_kappa - after_kappa (必须≥0+tolerance)
        """
        ...

    def compute_action_cost(
        self,
        coin_pos: tuple[int, int],
        player_pos: tuple[int, int],
        step_size: int,
        game: Any,
        adapter: LS20Adapter,
    ) -> float:
        """计算收集coin的动作代价。

        考虑:
          - 从当前位置到coin位置的BFS路径长度
          - 从coin位置返回最近路径点的BFS路径长度
          - 状态变更代价 (是否经过changer)

        Args:
            coin_pos: coin位置
            player_pos: 玩家当前位置
            step_size: 网格步长
            game: 游戏对象
            adapter: LS20Adapter

        Returns:
            动作代价 (步数)
        """
        ...

    def check_emergency_collect(
        self,
        remaining_steps: int,
        min_reserve: int,
        current_kappa: float,
        coin_options: list[tuple[tuple[int, int], float]],
        game: Any,
        adapter: LS20Adapter,
    ) -> CoinDecision | None:
        """紧急coin收集检查 (步数不足时)。

        当 remaining_steps < min_reserve 时:
          - 寻找κ-reduction/cost最优比的coin
          - 如果最优coin的κ-reduction > cost → 收集
          - 否则 → 不收集 (宁可步数耗尽也不违反Anti-Monotonicity)

        Args:
            remaining_steps: 剩余步数
            min_reserve: 最低储备
            current_kappa: 当前κ值
            coin_options: [(coin_pos, kappa_reduction)] 可选coin列表
            game: 游戏对象
            adapter: LS20Adapter

        Returns:
            CoinDecision 或 None (无需紧急收集)
        """
        ...
```

### 3.4 LS20IDOAdapter — LS20→IDO数据适配层

```python
@dataclass
class UVFeatures:
    """LS20 UV特征 (11维EML感知 + LS20游戏特定)"""
    position: tuple[int, int]              # 玩家位置 (x, y)
    state_dimensions: dict[str, int]       # 状态维度值 {"rotation": 0, "color": 1, "shape": 5}
    goal_state: dict[str, int]             # 目标状态要求
    changer_positions: list[tuple[int, int]]   # 所有changer位置
    changer_effects: dict[tuple, dict]     # changer效果映射
    wall_positions: set[tuple[int, int]]   # 墙位置集合
    coin_positions: list[tuple[int, int]]  # coin位置列表
    goal_positions: list[tuple[int, int]]  # goal位置列表
    eml_features: list[dict[str, Any]]     # EML感知11维特征 (来自EMLPerceiver)
    kappa_uv: float                        # UV信息基数
    kappa_ir: float                        # IR信息基数
    compression_ratio: float               # κ_UV / κ_IR
    step_size: int                         # 网格步长
    grid_size: int                         # 网格尺寸

class LS20IDOAdapter:
    """LS20→IDO数据适配层

    核心职责: 将LS20 game对象的数据转换为IDO理论模块所需的格式。
    10个理论模块为ARC-3通用设计，此适配层提供LS20专用转换。

    调用链:
      game_to_uv_features() → LS20Adapter + EMLPerceiver
      game_to_coset_input()  → UVFeatures → coset_search所需格式
      game_to_ido_input()    → UVFeatures → ido_axioms所需格式
    """

    def __init__(self) -> None:
        """初始化适配层。"""
        ...

    def game_to_uv_features(
        self,
        game: Any,
        level_idx: int,
        ls20_adapter: LS20Adapter,
    ) -> UVFeatures:
        """从game对象提取UVFeatures。

        调用:
          - LS20Adapter (位置/状态/changer/wall/goal提取)
          - EMLPerceiver (11维EML特征提取)
          - ido_axioms.estimate_kappa_uv/ir (κ值估算)

        Args:
            game: ARC-AGI-3游戏对象
            level_idx: 关卡索引
            ls20_adapter: LS20Adapter实例

        Returns:
            UVFeatures — 完整的LS20 UV特征
        """
        ...

    def game_to_coset_input(
        self,
        uv_features: UVFeatures,
    ) -> dict:
        """将UVFeatures转换为coset_search模块所需的输入格式。

        coset_search需要:
          - 11维特征向量 (eml_features)
          - 目标IR维度数 (target_ir_dims=4)
          - grid (用于特征提取)

        Args:
            uv_features: UV特征

        Returns:
            coset_search兼容的输入字典
        """
        ...

    def game_to_ido_input(
        self,
        uv_features: UVFeatures,
    ) -> dict:
        """将UVFeatures转换为ido_axioms模块所需的输入格式。

        ido_axioms需要:
          - kappa_uv, kappa_ir (信息基数)
          - IDOConfig (反单调性配置)
          - grid (用于κ估算)

        Args:
            uv_features: UV特征

        Returns:
            ido_axioms兼容的输入字典
        """
        ...

    def game_to_sporadic_input(
        self,
        game: Any,
        level_idx: int,
        ls20_adapter: LS20Adapter,
    ) -> dict:
        """将game数据转换为sporadic_group_filter所需输入。

        sporadic_group_filter需要:
          - grid (用于对称检测)
          - JinlingSphere列表 (来自EMLPerceiver)

        Args:
            game: 游戏对象
            level_idx: 关卡索引
            ls20_adapter: LS20Adapter

        Returns:
            sporadic_group_filter兼容的输入字典
        """
        ...

    def extract_changer_effects(
        self,
        game: Any,
        ls20_adapter: LS20Adapter,
    ) -> dict[tuple[int, int], dict[str, int]]:
        """提取changer效果映射。

        返回: {(changer_x, changer_y): {"rot": delta, "color": delta, "shape": delta}}
        通过模拟步进检测每个changer的状态变更效果。

        Args:
            game: 游戏对象
            ls20_adapter: LS20Adapter

        Returns:
            changer效果映射字典
        """
        ...

    def extract_state_requirements(
        self,
        game: Any,
        ls20_adapter: LS20Adapter,
    ) -> dict[str, int]:
        """提取goal的状态要求。

        返回: {"rotation": required_rot, "color": required_color, "shape": required_shape}

        Args:
            game: 游戏对象
            ls20_adapter: LS20Adapter

        Returns:
            状态要求字典
        """
        ...
```

### 3.5 solve_ls20() 修改后的管线接口

```python
def solve_ls20(game: Any, level_idx: int) -> list[tuple] | None:
    """LS20 solver using RG-Flow Preplan pipeline.

    v5.0 管线 (渐进替代, Q1选B):
      Stage1: κ-gradient oracle replay (5s, 保留)
      Stage2: RG-Flow Preplan (新增, 替换 _solve_ls20_delta_state_bfs)
      Stage3: κ-gradient direct / κ-PS BFS fallback (保留)

    RG-Flow Preplan流程:
      → EML感知UV特征 (11维)
      → Sporadic Group分类 (一次性)
      → Coset-Prioritized搜索 (2640有限路径)
      → Anti-Monotonicity剪枝
      → κ-Entropy GaussEx校验
      → 一次性执行 (coin仅在κ-reduction>cost时收集)

    Fallback: RG-Flow失败 → Stage3 κ-gradient/κ-PS BFS
    """
    # ── Stage 1: κ-gradient oracle replay (保留, 5s预算) ──
    plan = _solve_oracle_replay(game, "ls20", level_idx, max_steps=300, max_time=5.0)
    if plan is not None:
        return plan

    # ── Stage 2: RG-Flow Preplan (新增, 替换delta_state_bfs) ──
    planner = RGFlowPlanner()
    plan = planner.plan(game, level_idx)
    if plan is not None:
        return plan

    # ── Stage 3: κ-gradient direct + κ-PS BFS fallback (保留) ──
    plan = _solve_ls20_kappa_gradient_direct(game, level_idx)
    if plan is not None:
        return plan

    plan = _solve_ls20_kappa_ps_bfs(game, level_idx)
    if plan is not None:
        return plan

    return None
```

---

## 4. 程序调用流程（时序图）

> 完整时序图见 `docs/architecture/sequence-diagram.mermaid`

### 4.1 主管线流程 (solve_ls20 → RGFlowPlanner)

```
solve_ls20(game, level_idx)
  │
  ├─ Stage1: _solve_oracle_replay(game, "ls20", level_idx)  ─── [5s预算, 保留]
  │   └─ if plan≠None → return plan
  │
  ├─ Stage2: RGFlowPlanner.plan(game, level_idx)            ─── [新增, 替换delta_state_bfs]
  │   │
  │   ├─ LS20IDOAdapter.game_to_uv_features(game, level_idx, adapter)
  │   │   ├─ LS20Adapter 提取player/walls/changers/goals/coins
  │   │   ├─ EMLPerceiver 提取11维特征
  │   │   ├─ ido_axioms.estimate_kappa_uv/ir → κ_UV, κ_IR
  │   │   └─ → UVFeatures
  │   │
  │   ├─ RGFlowPlanner._classify_level(game, adapter)
  │   │   └─ LS20IDOAdapter.game_to_sporadic_input → sporadic_group_filter.classify
  │   │   └─ level_type ∈ {Niemeier, Moebius, Monster, Generic}
  │   │
  │   ├─ RGFlowPlanner._coset_state_search(uv_features, goal_state, level_type, game, adapter)
  │   │   ├─ LS20IDOAdapter.game_to_coset_input → coset_search格式
  │   │   ├─ CosetPrioritizedSearch → 330陪集×D₄=2640路径搜索
  │   │   ├─ StatePathSolver.compute_state_path_sequence → 状态路径序列
  │   │   │   ├─ 分析start→goal状态维度差异
  │   │   │   ├─ 规划changer访问顺序
  │   │   │   ├─ compute_changer_trigger_sequence → 多触发路径
  │   │   │   └─ compute_bfs_path → 两点间Wall-Map BFS
  │   │   └─ → StatePath候选列表
  │   │
  │   ├─ RGFlowPlanner._anti_monotonicity_prune(candidates, current_kappa)
  │   │   ├─ KappaBudgetController.should_collect_coin → 每步coin决策
  │   │   ├─ ido_axioms.verify_anti_monotonicity → κ增加即剪枝
  │   │   └─ → 剪枝后候选列表
  │   │
  │   ├─ RGFlowPlanner._verify_kappa_release(state_path, expected_delta_kappa)
  │   │   └─ ido_axioms.kappa_entropy_release → Δκ校验
  │   │
  │   ├─ RGFlowPlanner._execute_plan(game, state_path, adapter)
  │   │   ├─ 按预规划路径一次性执行
  │   │   ├─ KappaBudgetController.should_collect_coin → coin决策
  │   │   ├─ 检查remaining_steps → 紧急coin收集?
  │   │   └─ → 动作序列 [(dir, data), ...]
  │   │
  │   └─ if plan≠None → return plan
  │
  ├─ Stage3: _solve_ls20_kappa_gradient_direct(game, level_idx)  ─── [保留]
  │   └─ if plan≠None → return plan
  │
  ├─ Stage3: _solve_ls20_kappa_ps_bfs(game, level_idx)          ─── [保留]
  │   └─ if plan≠None → return plan
  │
  └─ return None
```

### 4.2 分段预规划流程 (L5/L6)

```
RGFlowPlanner._segmented_preplan(game, level_idx=5/6, n_segments=3)
  │
  ├─ Segment 1: start→changer₁
  │   ├─ UV特征提取 (Segment1的start→changer₁)
  │   ├─ coset搜索 (Segment1路径)
  │   ├─ Anti-Monotonicity剪枝 (Segment1 κ-check)
  │   ├─ κ-entropy校验 (Δκ₁ ≥ Δκ理论值/3)
  │   ├─ coin决策 (Segment1内是否收集)
  │   └─ → segment1_actions
  │
  ├─ Segment 2: changer₁→changer₂
  │   ├─ 以changer₁触发后状态为新的start
  │   ├─ UV特征重新提取 (Segment2)
  │   ├─ coset搜索 (Segment2路径)
  │   ├─ Anti-Monotonicity剪枝
  │   ├─ κ-entropy校验 (Δκ₂ ≥ Δκ理论值/3)
  │   └─ → segment2_actions
  │
  ├─ Segment 3: changer₂→goal
  │   ├─ 以changer₂触发后状态为新的start
  │   ├─ coset搜索 → BFS到goal
  │   ├─ κ-entropy校验 (Δκ₃ ≥ Δκ理论值/3)
  │   └─ → segment3_actions
  │
  └─ → concatenate(segment1 + segment2 + segment3)
```

### 4.3 Fallback流程 (RG-Flow失败 → Stage3)

```
RGFlowPlanner.plan() → None  (预规划失败原因)
  │
  ├─ UV特征提取超时 → fallback Stage3
  ├─ coset搜索无有效路径 → fallback Stage3
  ├─ Anti-Monotonicity剪枝全部候选 → fallback Stage3
  ├─ κ-entropy校验失败 → fallback Stage3
  ├─ 执行中状态不匹配 → fallback Stage3
  │
  └─ Stage3: _solve_ls20_kappa_gradient_direct
     │   ├─ κ-gradient方向导航 (不绕墙)
     │   └─ if fail → _solve_ls20_kappa_ps_bfs
     │       ├─ κ-PS BFS搜索复合状态(position,rot,color,shape)
     │       └─ 最后防线
```

### 4.4 Anti-Monotonicity coin决策流程

```
KappaBudgetController.should_collect_coin(current_kappa, coin_cost, kappa_reduction)
  │
  ├─ Step 1: 计算κ-reduction
  │   └─ compute_kappa_reduction(before, after)
  │
  ├─ Step 2: 计算action_cost
  │   └─ compute_action_cost(coin_pos, player_pos, step_size, game, adapter)
  │   ├─ BFS路径: player→coin
  │   ├─ BFS路径: coin→return_point
  │   └─ 总代价 = BFS(player→coin) + BFS(coin→return)
  │
  ├─ Step 3: Anti-Monotonicity判定
  │   └─ if kappa_reduction > action_cost + tolerance(0.05):
  │   │   → COLLECT (κ净减少，符合IDO定律)
  │   └─ else:
  │   │   → SKIP (违反Anti-Monotonicity)
  │
  ├─ Step 4: 步数储备检查
  │   └─ if remaining < min_reserve(5):
  │   │   → 紧急收集检查 (优先κ-reduction/cost最优比coin)
  │   │   → 如果最优coin也不符合Anti-Monotonicity → 不收集
  │
  └─ → CoinDecision(should_collect, reason)
```

---

## 5. 待明确事项

| # | 事项 | 当前假设 | 影响范围 | 需澄清方 |
|---|------|----------|----------|----------|
| A1 | **changer效果提取方法**: 如何确定每个changer的状态变更效果(rotation/color/shape各delta多少)? | 通过模拟步进(deepcopy+perform_action)检测每个changer的效果 | `LS20IDOAdapter.extract_changer_effects`, `StatePathSolver.compute_changer_trigger_sequence` | 开发者(实测验证) |
| A2 | **goal状态要求提取**: goal sprite的状态要求如何获取? 是从sprite属性读取还是需要模拟探测? | 假设LS20Adapter已提供goal状态要求属性 | `LS20IDOAdapter.extract_state_requirements` | 开发者(实测验证) |
| A3 | **分段预规划段数确定**: L5/L6的分段数量(n_segments)如何确定? 是固定3段还是按changer数量动态调整? | 假设按changer数量动态调整: n_segments = len(changer_sequence) | `RGFlowPlanner._segmented_preplan` | 开发者+架构师(实测验证) |
| A4 | **EMLPerceiver对LS20的适用性**: EMLPerceiver的6-stage pipeline是为ARC网格设计的，LS20是2D游戏而非网格。如何将LS20游戏场景转换为EML可处理的grid? | 假设LS20IDOAdapter将游戏场景渲染为grid(grid_size×grid_size的2D数组)，再送入EML | `LS20IDOAdapter.game_to_uv_features` | 开发者(实测验证) |
| A5 | **coset_search的LS20场景适配**: coset_search.py假设输入是ARC grid(numpy array)，LS20游戏状态(position+state_tuple)不是grid。如何将LS20状态映射到coset空间? | 假设LS20IDOAdapter将LS20状态维度映射到11维UV特征空间，coset搜索在特征空间而非grid空间进行 | `LS20IDOAdapter.game_to_coset_input`, `StatePathSolver.compute_state_path_sequence` | 架构师(设计决策) |
| A6 | **κ值在LS20中的具体含义**: ido_axioms中的κ_UV/κ_IR估算公式是为ARC grid设计的(log2(unique_colors × shapes × area))。LS20的κ应该如何估算? | 假设κ_UV = log2(状态维度空间大小)，κ_IR = log2(目标状态空间大小)；L1: κ_UV=log2(4×4×...), κ_IR=1 | `KappaBudgetController.compute_kappa_reduction`, `RGFlowPlanner._verify_kappa_release` | 架构师(理论对齐) |
| A7 | **ReplayEngine集成方式**: delta_state.py的ReplayEngine设计为grid mode(game mode)两种。LS20属于game mode。如何将StatePathSolver的路径与ReplayEngine对接? | 假设StatePathSolver直接使用game mode ReplayEngine验证路径(已有deepcopy+perform_action流程) | `StatePathSolver.verify_path_on_replay_engine` | 开发者(实测验证) |
| A8 | **_solve_ls20_delta_state_bfs完全移除还是保留?**: 渐进替代方案下，旧Stage2代码是删除还是注释保留? | 假设注释保留(标记为_DEPRECATED)，方便回退调试 | `game_solvers.py` | 产品经理+开发者 |

---

# Part B: 任务分解

## 6. 依赖包列表

**无新增依赖**。全部使用项目已有依赖:

```txt
# === 已有依赖 (不变) ===
numpy>=1.26.0           # κ-entropy计算、特征向量操作
arcengine               # LS20游戏接口 (ActionInput)
pytest>=8.0.0           # 测试框架

# === 已有理论模块 (import验证通过, 10个) ===
# rg_flow.py            — RG Flow β函数
# coset_search.py       — 陪集优先搜索
# ido_axioms.py         — 反单调性 + κ-entropy
# sporadic_group_filter.py — 散在群分类
# kappa_winding.py      — κ-Winding相位
# semantic_constants.py — 语义常数
# ramanujan_conjecture.py — Ramanujan猜想
# octonion_phase.py     — 八元数相位
# eml_perceiver.py      — EML感知器
# hungarian_align.py    — Hungarian对齐

# === 已有基础设施模块 ===
# delta_state.py        — ReplayEngine零拷贝
# oracle_adapters.py    — LS20Adapter
# game_configs.py       — 游戏配置
```

---

## 7. 任务列表 (≤5个, 按依赖排序)

### T01: 基础设施 + LS20→IDO适配层

| 字段 | 内容 |
|------|------|
| **Task ID** | T01 |
| **Task Name** | 基础设施 + LS20→IDO数据适配层 |
| **Priority** | P0 |
| **Source Files** | `src/agent/ls20_ido_adapter.py` (新建~150行), `src/agent/rg_flow_planner.py` (新建骨架~50行), `src/agent/state_path_solver.py` (新建骨架~40行), `src/agent/kappa_budget_controller.py` (新建骨架~30行), `src/agent/__init__.py` (修改, 新增import), `src/agent/game_configs.py` (修改, 新增RG-Flow常量) |
| **Dependencies** | 无 |
| **预估总行数** | ~320行 (含骨架) |
| **关键实现要点** | 1. `ls20_ido_adapter.py`: 实现UVFeatures dataclass + LS20IDOAdapter类, 包含game_to_uv_features(), game_to_coset_input(), game_to_ido_input(), game_to_sporadic_input(), extract_changer_effects(), extract_state_requirements() 的完整实现<br>2. 3个新模块骨架: 定义类接口(dataclass + method签名 + docstring), 不含实现细节, 仅import已有理论模块<br>3. `__init__.py`: 新增 RGFlowPlanner, StatePathSolver, KappaBudgetController, LS20IDOAdapter 的import<br>4. `game_configs.py`: 新增 RG_FLOW_MAX_TIME=45.0, RG_FLOW_BFS_TIME=8.0, ANTI_MONO_TOLERANCE=0.05, SEGMENTED_PREPLAN_LEVELS=[5,6], DEFAULT_N_SEGMENTS=3 |

### T02: RG-Flow核心管线 (P0-1/P0-2/P0-3/P0-4)

| 字段 | 内容 |
|------|------|
| **Task ID** | T02 |
| **Task Name** | RG-Flow核心管线实现 |
| **Priority** | P0 |
| **Source Files** | `src/agent/rg_flow_planner.py` (完整实现~400行), `src/agent/state_path_solver.py` (完整实现~300行), `src/agent/kappa_budget_controller.py` (完整实现~200行), `src/agent/ls20_ido_adapter.py` (增强: changer效果模拟+状态要求提取, 追加~50行) |
| **Dependencies** | T01 |
| **预估总行数** | ~950行 |
| **关键实现要点** | 1. `rg_flow_planner.py`: 完整实现plan()主流程 — UV特征提取→Level分类→coset搜索→Anti-Monotonicity剪枝→κ-verification→执行; _uv_feature_extraction()调用LS20IDOAdapter+EMLPerceiver; _coset_state_search()调用CosetPrioritizedSearch+StatePathSolver; _anti_monotonicity_prune()调用ido_axioms.verify_anti_monotonicity; _execute_plan()调用KappaBudgetController; _verify_kappa_release()调用ido_axioms.kappa_entropy_release<br>2. `state_path_solver.py`: 完整实现compute_state_path_sequence() — 分析状态维度差异→规划changer访问→生成路径; compute_changer_trigger_sequence() — 多触发leave-and-return路径; compute_leave_and_return_path() — 撤退再返回; compute_bfs_path() — 复用_lightweight_bfs逻辑但作为StatePathSolver方法; verify_path_on_replay_engine() — 使用ReplayEngine零拷贝验证<br>3. `kappa_budget_controller.py`: 完整实现should_collect_coin() — κ-reduction>cost判定; compute_kappa_reduction() — κ差值计算; compute_action_cost() — BFS路径代价; allocate_time_budget() — β(κ)驱动时间分配; check_emergency_collect() — 步数不足紧急检查<br>4. L1预期路径: `(29,40)→BFS→(49,45)[changer×3]→BFS→(14,40)[goal]` |

### T03: 管线集成 + Level分类 + κ-Verification (P0集成 + P1-1 + P1-2)

| 字段 | 内容 |
|------|------|
| **Task ID** | T03 |
| **Task Name** | 管线集成 + Level分类 + κ-Verification |
| **Priority** | P0 (集成) + P1 (分类/验证) |
| **Source Files** | `src/agent/game_solvers.py` (修改solve_ls20, ~30行变更), `src/agent/sporadic_group_filter.py` (修改, 新增LS20 classify_ls20_level方法~50行), `src/agent/ido_axioms.py` (修改, 新增verify_ls20_kappa_release方法~30行), `src/agent/rg_flow_planner.py` (增强: fallback逻辑+时间预算感知, 追加~30行) |
| **Dependencies** | T01, T02 |
| **预估变更行数** | ~140行 |
| **关键实现要点** | 1. `game_solvers.py solve_ls20()`: 替换Stage2调用 — `plan = _solve_ls20_delta_state_bfs(game, level_idx)` → `plan = RGFlowPlanner().plan(game, level_idx)`; 旧代码标记为`_DEPRECATED_solve_ls20_delta_state_bfs`保留(注释); Stage1/3不变<br>2. `sporadic_group_filter.py`: 新增classify_ls20_level(game, level_idx) — L0→Möbius(rotation单维); L1→Möbius; L2→Monster(color+rotation双维); L3-L6按EML特征动态分类<br>3. `ido_axioms.py`: 新增verify_ls20_kappa_release(state_path, expected_delta_kappa) — LS20专用κ-entropy校验; L0: Δκ≥log(4/1)≈1.39nat; L1: Δκ≥log(4/1)≈1.39nat; L2+: 按实际状态空间计算<br>4. `rg_flow_planner.py`: 增强plan()方法 — 添加时间预算感知(超时→返回None→fallback); 添加分段预规划入口判断(level_idx∈[5,6]→调用_segmented_preplan) |

### T04: 多触发规划 + 分段预规划 + Replay集成 (P1-3 + P1-4 + L5/L6)

| 字段 | 内容 |
|------|------|
| **Task ID** | T04 |
| **Task Name** | 多触发规划 + 分段预规划 + ReplayEngine集成 |
| **Priority** | P1 |
| **Source Files** | `src/agent/rg_flow_planner.py` (追加_segmented_preplan完整实现~80行), `src/agent/octonion_phase.py` (修改, 新增compute_multi_trigger_phase方法~40行), `src/agent/delta_state.py` (修改, 新增verify_ls20_state_path方法~30行), `src/agent/state_path_solver.py` (增强: compute_segmented_paths实现~50行, compute_changer_trigger_sequence增强~30行) |
| **Dependencies** | T01, T02, T03 |
| **预估变更行数** | ~230行 |
| **关键实现要点** | 1. `rg_flow_planner.py _segmented_preplan()`: L5/L6分段预规划 — 每段独立UV提取+coset搜索+κ校验; 段间changer触发衔接; 每段coin决策独立; 全段κ-release校验(总Δκ≥理论值)<br>2. `octonion_phase.py`: 新增compute_multi_trigger_phase(changer_pos, triggers_needed) — 用八元体相位计算多触发最优路径(离开方向选择、撤退步数优化)<br>3. `delta_state.py`: 新增verify_ls20_state_path(game, actions, expected_state) — ReplayEngine验证路径+状态匹配检查(position+rotation+color+shape四维全匹配)<br>4. `state_path_solver.py compute_segmented_paths()`: 将长changer序列拆分为多段StatePath; compute_changer_trigger_sequence增强 — 使用octonion_phase选择最优离开方向 |

### T05: 测试 + Benchmark + Debug可视化

| 字段 | 内容 |
|------|------|
| **Task ID** | T05 |
| **Task Name** | 测试 + Benchmark + Debug可视化 |
| **Priority** | P1 (测试) + P2 (可视化) |
| **Source Files** | `tests/test_rg_flow_planner.py` (新建~200行), `tests/test_state_path_solver.py` (新建~150行), `tests/test_kappa_budget_controller.py` (新建~100行), `tests/test_ls20_ido_adapter.py` (新建~80行), `bench_ls20.py` (修改, 新增RG-Flow模式选项~20行) |
| **Dependencies** | T01, T02, T03, T04 |
| **预估总行数** | ~550行 |
| **关键实现要点** | 1. `test_rg_flow_planner.py`: 测试plan()主流程 — mock game对象; UV特征提取正确性; coset搜索返回非None; Anti-Monotonicity剪枝效果; κ-entropy校验; 执行plan生成合法action序列; fallback到Stage3<br>2. `test_state_path_solver.py`: 测试compute_state_path_sequence — 状态维度分析正确; changer访问顺序合理; 多触发leave-and-return路径; BFS路径可达; 分段路径衔接<br>3. `test_kappa_budget_controller.py`: 测试should_collect_coin — κ-reduction>cost→collect; κ-reduction<cost→skip; 紧急收集逻辑; tolerance=0.05容忍<br>4. `test_ls20_ido_adapter.py`: 测试game_to_uv_features — UVFeatures字段完整; game_to_coset_input格式兼容; changer效果提取; 状态要求提取<br>5. `bench_ls20.py`: 新增--mode rg_flow选项 — 跑RG-Flow Preplan而非旧delta_state_bfs; 输出L0-L6 RHAE + solve_time |

---

## 8. 共享知识 (跨文件约定)

### 8.1 编码规范

```yaml
Python版本: 3.11+, 使用 from __future__ import annotations
代码风格: 项目已有风格 (snake_case函数, PascalCase类, UPPER_SNAKE_CASE常量)
类型标注: 所有public方法必须有type hints + docstring (Google风格)
私有方法: 前缀下划线 (_uv_feature_extraction, _coset_state_search)
文档字符串: 每个class和public method必须有, 含Args/Returns/Raises
```

### 8.2 数据格式约定

```yaml
动作序列格式: list[tuple[int, Any]] — [(direction, data), ...]
  direction: 1=LEFT, 2=RIGHT, 3=UP, 4=DOWN (LS20映射, 已在v3.18.3检测)
  data: None (LS20只处理方向移动, 无ACTION6)

状态元组格式: StateTuple(position=(x,y), rotation=int, color=int, shape=int)
  position: (像素坐标x, 像素坐标y), 坐标系与LS20Adapter一致
  rotation: 0-3 (循环切换)
  color: 颜色索引
  shape: 形状索引

UV特征格式: UVFeatures dataclass (见§3.4)
  所有位置使用像素坐标(与LS20Adapter一致), 不是grid坐标
  step_size: 从LS20Adapter.gisrhqpee获取(默认5)

κ值约定:
  κ_UV = log2(状态维度空间大小) — LS20专用公式(非ARC grid公式)
  κ_IR = log2(目标状态空间大小)
  Δκ = ln(κ_UV / κ_IR) — 自然对数(非log2)
  Anti-Monotonicity tolerance = 0.05 (Q3选B)

时间预算约定:
  总预算: 45.0s (L0-L4), 60.0s (L5-L6分段)
  BFS单次: 8.0s
  UV提取: 5.0s
  fallback保留: 5-10s
```

### 8.3 错误处理策略

```yaml
预规划超时: RGFlowPlanner.plan() 设定max_total_time, 超时返回None → fallback Stage3
BFS路径不可达: StatePathSolver.compute_bfs_path() 返回None → 尝试下一候选路径
Anti-Monotonicity全部剪枝: _anti_monotonicity_prune() 返回空列表 → fallback Stage3
κ-entropy校验失败: _verify_kappa_release() 返回False → 尝试下一候选路径
ReplayEngine验证失败: verify_path_on_replay_engine() 返回False → 路径标记为无效
游戏对象异常: deepcopy(game)失败 → 返回None (安全退出)
紧急步数耗尽: remaining < min_reserve 且无符合条件的coin → 尝试直奔goal
```

### 8.4 日志规范

```yaml
日志级别:
  DEBUG: coset搜索细节(BFS节点数/剪枝数/陪集枚举进度)
  INFO: 任务进度(Stage进入/UV特征提取完成/路径生成)
  WARNING: 降级(fallback激活/κ校验失败/Anti-Mono剪枝)
  ERROR: 致命错误(game对象异常/adapter初始化失败)

关键日志点:
  [RG-Flow] Stage2 entered: level_idx=X, level_type=Y
  [RG-Flow] UV extraction done: κ_UV=X.XX, κ_IR=X.XX, ratio=X.XX
  [RG-Flow] Coset search: X candidates found, Y pruned by anti-mono
  [RG-Flow] κ-verification: Δκ=X.XX (expected ≥ Y.YY) → PASS/FAIL
  [RG-Flow] Plan executed: N steps, M coins collected, time=X.Xs
  [RG-Flow] Fallback: reason=X → entering Stage3

Debug开关 (_DEBUG_L2=True 已存在于game_solvers.py):
  RG-Flow也使用 _DEBUG_L2 标志控制详细输出
```

### 8.5 模块调用约定

```yaml
LS20IDOAdapter: 所有理论模块的调用必须经过此适配层
  → eml_perceiver: 通过 game_to_uv_features() 调用
  → coset_search: 通过 game_to_coset_input() 调用
  → ido_axioms: 通过 game_to_ido_input() 调用
  → sporadic_group_filter: 通过 game_to_sporadic_input() 调用
  禁止: 直接将game对象传给理论模块

已有模块禁止修改核心逻辑:
  rg_flow.py, coset_search.py, ido_axioms.py 等10个模块
  只允许: 新增LS20专用方法(如classify_ls20_level, verify_ls20_kappa_release)
  不允许: 修改已有方法的签名或核心逻辑

ReplayEngine复用:
  StatePathSolver使用delta_state.ReplayEngine做路径验证
  不重新实现BFS — 复用game_solvers.py中的_lightweight_bfs逻辑
  (通过StatePathSolver.compute_bfs_path()包装调用)
```

---

## 9. 任务依赖图

```mermaid
graph TD
    T01[T01: 基础设施 + LS20→IDO适配层<br/>P0 | ~320行]
    T02[T02: RG-Flow核心管线<br/>P0 | ~950行]
    T03[T03: 管线集成 + Level分类 + κ-Verification<br/>P0+P1 | ~140行]
    T04[T04: 多触发 + 分段预规划 + Replay集成<br/>P1 | ~230行]
    T05[T05: 测试 + Benchmark + Debug<br/>P1+P2 | ~550行]

    T01 --> T02
    T01 --> T03
    T02 --> T03
    T02 --> T04
    T03 --> T04
    T01 --> T05
    T02 --> T05
    T03 --> T05
    T04 --> T05

    style T01 fill:#4CAF50,color:#fff
    style T02 fill:#2196F3,color:#fff
    style T03 fill:#2196F3,color:#fff
    style T04 fill:#FF9800,color:#fff
    style T05 fill:#9C27B0,color:#fff
```

---

> **文档版本**: v5.0 | **架构师**: 高见远 (Gao) | **状态**: 待团队评审
