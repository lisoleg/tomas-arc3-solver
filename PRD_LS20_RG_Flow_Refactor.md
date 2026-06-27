# PRD: LS20 RG-Flow Solver 重构

> **版本**: v6.0 (增量PRD — 吸纳势态知感+Re-Inflow)
> **作者**: 许清楚 (Xu), Product Manager
> **日期**: 2026-06-27
> **参考文章**:
> - [IDO/TOMAS RG流与陪集拓扑](https://mp.weixin.qq.com/s/_3Ph6lKUGRFxpHztvMeMcQ) — 宇宙学-物理层：RG流、330陪集、拉马努金机
> - [势态知感压缩态势感知](https://mp.weixin.qq.com/s/j4XjnkRUE2ohKCdT9jCvxQ) — 认知-工程层：SA→SM压缩、四维度机制、Re-Inflow回溯

---

## 1. 项目信息

| 字段 | 值 |
|------|------|
| Language | 中文 |
| Programming Language | Python 3.11+ (已有项目技术栈) |
| Project Name | `ls20_rg_flow_solver_refactor` |
| 原始需求复述 | 基于两篇IDO/TOMAS文章的RG-Flow理论+势态知感(SA→SM)压缩，将LS20 solver从贪婪迭代架构重构为预规划+认知压缩架构，使L0-L6全通(RHAE ≥ 80) |

---

## 2. 产品定义

### 2.1 产品目标

| # | 目标 | 可验证指标 |
|---|------|-----------|
| G1 | **LS20 L0-L6 全通** | bench_ls20.py 输出 `ALL_PASS: YES`，7/7关卡RHAE ≥ 80 |
| G2 | **L1 时间预算内完成** | L1 solve time < 30s (当前55s timeout失败) |
| G3 | **信息压缩效率** | plan的κ-release ≥ Δκ理论值(≥ 4.32 nat for 3×4=12态空间) |
| G4 | **势态知感压缩比** | CR = IC(SA)/IC(SM) ≥ 10³ (TB级像素流→KB级语义向量) |

### 2.2 双层理论框架（物理层+认知层）

**物理-认知同构公理**：宏观物理(RG流 K=11→4)与微观认知(SA→SM压缩)是同一信息动力学过程在不同尺度上的投影。ARC-AGI-3 是二维网格上的微型宇宙演化。

| 层次 | 理论 | 核心机制 | LS20映射 |
|------|------|----------|----------|
| **物理层** (宇宙学) | IDO RG流 + 陪集拓扑 | UV(11D高熵)→IR(4D低熵)坍缩 | 解空间约束: 330陪集替代无限搜索 |
| **认知层** (工程学) | 势态知感 SA→SM | SA(Inflow)→κ-Snap归约→SM(决策) | 游戏观测→压缩决策路径 |

ARC-AGI-3求解 = 两阶段信息事件：
```
[LS20 游戏状态] → SA(Inflow原始观测) → [EML Perceive 11维] → [陪集 Selection 330]
→ [κ-Snap + 先验知 + 价值算计] → SM(压缩决策) → [动作序列执行]
```

### 2.3 用户故事

| # | 角色 | 功能 | 价值 |
|---|------|------|------|
| US1 | Solver开发者 | Solver在L1**预计算**最优状态路径序列 `(rot=0→changer@3次→rot=3→goal)`，一次性执行而非逐步迭代 | 从55s超时失败→<30s通关，消除时间浪费 |
| US2 | Solver开发者 | Solver用**陪集剪枝**将导航搜索从无限BFS扩展缩减到330×D₄=2640有限coset路径 | BFS搜索空间从O(N²)→O(2640)，时间可控 |
| US3 | Solver开发者 | **Anti-monotonicity**确保coin收集只在κ-reduction > action-cost时发生（当前贪婪收集浪费约40步） | 步数利用率从≈50%→≥90%，RHAE显著提升 |
| US4 | Solver开发者 | **κ-entropy verification**确保plan的信息压缩比与理论κ-release匹配（GaussEx校验） | plan正确性有理论保证，减少random trial |
| US5 | Solver开发者 | **Sporadic Group Filter**对Level按对称类型分类，路由到对应求解策略 | L2(color+rotation双维度)用Monster策略、L0(rotation单维度)用Möbius策略 |
| US6 | Solver开发者 | **势态知感(SA→SM)四维度压缩**：认知(先验注入)、时间(相位拟合)、价值(效用剪枝)、交互(ψ-Anchor) | 从单维度贪心→四维度协同压缩，全局最优 |
| US7 | Solver开发者 | **Re-Inflow解压回溯**：当SM剪枝导致GaussEx残差超预期(Dead-Zero)时，回溯SA缓存重跑κ-Snap | 防止误剪枝(如跳过必需coin)，容错自愈 |
| US8 | Solver开发者 | **压缩比CR量化**：CR=IC(SA)/IC(SM)≥10³，TB级游戏观测→KB级决策向量 | 智能水平可量化，迭代优化有方向 |

---

## 3. 核心变更映射

### 3.1 架构变更对照表

| 当前架构 (UV高熵态) | 目标架构 (IR低熵态) | IDO/TOMAS对应 | 预期效果 |
|----------------------|----------------------|---------------|----------|
| 贪婪迭代 (逐步决策) | **RG-Flow预规划** (先算路径再执行) | UV→IR坍缩 β(κ) = -κ × log(κ/κ_ir)/κ_uv | 消除时间浪费，L1从55s→<30s |
| 局部BFS (O(N²)位置搜索) | **Coset-Prioritized Navigation** (330陪集×24对称=7920路径) | |S₁₁|/(|S₇|×|S₄|) = 330 | BFS从无限→有界，搜索时间可控 |
| 贪婪coin收集 (浪费约40步) | **Anti-Monotonicity Budget Control** (仅在κ-reduction > cost时收集) | κ必须单调递减，κ增加即剪枝 | 步数利用率从≈50%→≥90% |
| deepcopy验证 (每步一次) | **κ-Entropy GaussEx Verification** (信息压缩比校验) | Δκ = ln(IC_UV/IC_IR) | 正确性有理论保证 |
| 无Level分类 | **Sporadic Group Filter** (4类对称路由) | 26散在群→Niemeier/Möbius/Monster/Generic | 策略精准匹配关卡类型 |
| 单维度贪心(仅位置最短) | **SA→SM四维度压缩** (认知+时间+价值+交互) | 刘伟势态知感: 知滤感/势代态/算计统计算/人机折叠 | 从局部最优→全局四维协同最优 |
| 无回溯容错 | **Re-Inflow解压回溯** (Dead-Zero误剪枝→SA缓存→重跑κ-Snap) | SM剪枝过激→保留SA→Re-Inflow | 容错自愈，防止误剪必需路径 |
| 无压缩比量化 | **CR = IC(SA)/IC(SM)** 压缩比度量 | 智能水平与CR正相关，需GaussEx约束 | 智能水平可量化迭代 |

### 3.2 管线对照

**当前管线** (solve_ls20 3-stage):
```
Stage1: κ-gradient oracle replay (5s, 贪婪迭代)
Stage2: _solve_ls20_delta_state_bfs (45s, 贪婪迭代BFS)
Stage3: κ-gradient direct / κ-PS BFS fallback
```

**目标管线** (RG-Flow + SA→SM 3-phase):
```
Phase1: SA(Inflow) — 原始态势感知采样
  → 游戏状态观测(wall/changer/coin/player位置)
  → EML Perceiver提取11维UV特征
  → 保留SA缓存(供Re-Inflow回溯)

Phase2: SM(κ-Snap归约) — 势态知感压缩决策
  → 四维度压缩: 认知(先验注入) + 时间(相位拟合) + 价值(效用剪枝) + 交互(ψ-Anchor)
  → Coset-Prioritized搜索330陪集×24对称=7920路径
  → Anti-Monotonicity剪枝(κ增加即剪)
  → 生成最优状态路径序列
  → Dead-Zero检测 → Re-Inflow回溯(如需)

Phase3: Execution(IR一次性执行) — 坍缩态执行
  → 按预规划路径执行动作
  → κ-Entropy GaussEx校验(Δκ≥理论值 AND |κ_entropy-14.32|<0.5)
  → Coin仅在κ-reduction > cost时收集
  → CR = IC(SA)/IC(SM) ≥ 10³ 压缩比校验
```

### 3.3 四维度压缩机制详表

| 维度 | IDO机制 | LS20算法实现 | L1案例 |
|------|---------|-------------|--------|
| **认知(知滤感)** | 先验注入与贝叶斯上加权 | 已知rotation=0→需rot=3，注入先验: changer需3次触发 | 知: goal需rot=3 → 滤: 跳过无用coin → 感: 导航changer |
| **时间(势代态)** | 流贯相位拟合与突变检测 | 触发序列规划: rot=0→1→2→3, 关键节点锚定changer位置 | 势: 状态演变方向 → 代: 触发周期 → 态: 状态锚点 |
| **价值(算计统计算)** | 价值加权因果剪枝 | coin收集决策: κ-reduction vs action-cost权衡 | 算: 评估coin价值 → 计: 预算控制 → 统: 总成本 → 计算: 最优路径 |
| **交互(人机折叠)** | ψ-Anchor自适应带宽 | ψ-Anchor置信度驱动: 高置信直接执行，低置信展开验证 | 人: 预规划确定性高 → 机: 直接执行 → 折叠: 减少验证开销 |

### 3.4 Re-Inflow解压回溯机制

```
正常路径: SA(Inflow) → SM(κ-Snap压缩) → Execution → GaussEx ✓

误剪枝路径: SA(Inflow) → SM(κ-Snap压缩) → Execution → GaussEx ✗
  → Dead-Zero检测: residual > EPS or |κ_entropy-14.32| > TOL
  → Re-Inflow触发: 回溯SA缓存 → 释放被剪枝的候选 → 重跑κ-Snap
  → 第二轮SM → Execution → GaussEx ✓ (容错自愈)
```

**LS20 L1案例**: 若Anti-Monotonicity误剪了必需的coin收集路径(coin位于changer→goal路径上的必经点):
- Dead-Zero: GaussEx发现plan无法到达goal(路径被阻断)
- Re-Inflow: 回溯SA缓存 → 重新评估coin价值 → 发现coin κ-reduction > cost
- 重跑κ-Snap: 包含coin收集 → plan修正 → 通关

---

## 4. 技术规范

### 4.1 需求池

#### P0 — Must Have (核心通路)

| # | 需求 | 描述 | 验收标准 |
|---|------|------|----------|
| P0-1 | **RG-Flow Preplan模块** | 替代`_solve_ls20_delta_state_bfs`的贪婪迭代，实现UV→IR预规划 | L1通关时间<30s，plan非None |
| P0-2 | **状态路径序列生成** | 预计算最优 `(position, state)` 序列：`start→changer×N→goal` | L1: `pos(29,40)→changer@49,45×3→goal@14,40` 自动生成 |
| P0-3 | **SA→SM四维度压缩** | 势态知感压缩: 认知(先验)+时间(相位)+价值(效用)+交互(ψ-Anchor) | 四维度协同决策，替代单维度贪心 |
| P0-4 | **Re-Inflow解压回溯** | Dead-Zero误剪枝→回溯SA缓存→重跑κ-Snap | 容错自愈: 误剪路径可自动修正 |
| P0-5 | **Anti-Monotonicity Budget Control** | coin收集仅在κ-reduction > action-cost时触发 | L1步数浪费从≈40步→≤5步 |
| P0-6 | **Coset-Prioritized Navigation** | 用330陪集×24对称=7920有界路径替代无限BFS | BFS节点数≤7920，时间≤8s/路径 |

#### P1 — Should Have (效率提升)

| # | 需求 | 描述 | 验收标准 |
|---|------|------|----------|
| P1-1 | **κ-Entropy GaussEx Verification** | plan完成后校验Δκ≥Δκ理论值 AND |κ_entropy-14.32|<0.5 | L0: Δκ≥log(4/1)≈1.39 nat; L1: Δκ≥4.32 nat |
| P1-2 | **Sporadic Group Level分类** | 对L0-L6按对称类型分类路由求解策略 | L0→Möbius(rotation单维); L2→Monster(color+rotation多维) |
| P1-3 | **多触发规划** | changer需N次触发时，预计算最优leave-and-return路径 | L1: rotation 0→3需3次触发，自动计算最优路径 |
| P1-4 | **ReplayEngine零拷贝集成** | 新架构复用已有delta_state.py的ReplayEngine | 路径验证时间≤1s/验证 |
| P1-5 | **CR压缩比量化** | CR=IC(SA)/IC(SM)度量智能水平 | CR≥10³ (TB→KB压缩) |

#### P2 — Nice to Have (优化锦上添花)

| # | 需求 | 描述 | 验收标准 |
|---|------|------|----------|
| P2-1 | **Ramanujan Conjecture Engine集成** | 对数列类关卡用PSLQ+连分数加速 | 仅当关卡含数列模式时启用 |
| P2-2 | **κ-Winding Phase可视化** | 八元体相位缠绕调试输出 | `_DEBUG_L2=True`时输出wκ轨迹 |
| P2-3 | **跨关卡记忆** | L0通关策略自动迁移到L1/L2 | L1首次尝试即利用L0的changer策略 |

### 4.2 模块依赖关系

已有理论模块 (10个, import已验证) → 新模块映射:

| 已有模块 | 新模块中的角色 |
|----------|---------------|
| `rg_flow.py` (478行) | **核心**: RG-Flow Preplan的β函数驱动器 |
| `coset_search.py` (688行) | **核心**: Coset-Prioritized Navigation引擎 |
| `ido_axioms.py` (550行) | **核心**: Anti-Monotonicity剪枝 + κ-Entropy校验 |
| `sporadic_group_filter.py` (718行) | **P1**: Level分类路由 |
| `kappa_winding.py` (522行) | **P2**: κ-Winding调试可视化 |
| `semantic_constants.py` (445行) | **支撑**: 语义常数定义 |
| `ramanujan_conjecture.py` (770行) | **P2**: 数列模式加速 |
| `octonion_phase.py` (598行) | **P1**: 八元体相位计算(多触发规划) |
| `eml_perceiver.py` (897行) | **P0**: UV特征提取(11维感知) |
| `hungarian_align.py` (344行) | **P1**: 状态匹配最优对齐 |

**新增模块** (需开发):

| 模块 | 行数估计 | 描述 |
|------|----------|------|
| `rg_flow_planner.py` | ~500 | RG-Flow预规划主入口+SA→SM四维度压缩+Re-Inflow回溯 |
| `state_path_solver.py` | ~350 | 状态路径序列生成+多触发路径计算 |
| `kappa_budget_controller.py` | ~250 | Anti-Monotonicity预算控制+CR压缩比量化 |

### 4.3 关键接口设计

```python
# rg_flow_planner.py — 核心接口
class RGFlowPlanner:
    """RG-Flow预规划器 — SA(Inflow)→SM(κ-Snap归约)求解LS20

    两阶段信息事件:
      Phase1: SA(Inflow) — 原始态势感知采样
      Phase2: SM(κ-Snap归约) — 四维度压缩决策
    """

    def plan(self, game, level_idx) -> list[tuple] | None:
        """SA→SM 3-phase管线: SA采样→SM压缩→IR执行"""
        # Phase1: SA(Inflow) — 保留原始流贯缓存
        sa_cache = self._sa_inflow(game)
        # Phase2: SM(κ-Snap归约) — 四维度压缩
        sm = self._sm_compression(sa_cache)
        # Phase3: IR执行 — 一次性执行
        plan = self._execute_ir(game, sm)
        # Dead-Zero检测 → Re-Inflow回溯
        if plan is None or self._dead_zero_detected(sm):
            return self._re_inflow(sa_cache)
        return plan

    def _sa_inflow(self, game) -> SACache:
        """SA(Inflow): 原始态势感知采样 — 保留完整观测供Re-Inflow"""
        # EML提取11维UV特征
        # wall/changer/coin/player/goal全量采样
        # 保留SA缓存(供Dead-Zero回溯)

    def _sm_compression(self, sa_cache) -> SMDecision:
        """SM(κ-Snap归约): 四维度压缩决策"""
        # 认知(知滤感): 先验注入(已知goal需rot=3)
        # 时间(势代态): 相位拟合(触发序列规划)
        # 价值(算计统计算): 效用剪枝(coin只在κ-reduction>cost时收集)
        # 交互(人机折叠): ψ-Anchor(置信度驱动)
        # Coset-Prioritized搜索330陪集
        # Anti-Monotonicity剪枝(κ增加即剪)
        # κ-Entropy校验(|κ_entropy-14.32|<0.5)

    def _dead_zero_detected(self, sm) -> bool:
        """Dead-Zero误剪枝检测: GaussEx残差超预期"""
        # residual > EPS or |κ_entropy - 14.32| > TOL

    def _re_inflow(self, sa_cache) -> list[tuple] | None:
        """Re-Inflow解压回溯: 回溯SA缓存→释放被剪枝→重跑κ-Snap"""
        # 1. 从SA缓存恢复被剪枝的候选
        # 2. 放宽Anti-Monotonicity tolerance
        # 3. 重跑κ-Snap with wider pruning window


# state_path_solver.py — 状态路径序列
class StatePath:
    """预计算的状态路径序列 — SM压缩产物"""
    start_state: tuple  # (pos, rot, color, shape)
    goal_state: tuple   # 目标状态
    changer_sequence: list[ChangerStep]  # changer触发序列(时间维度: 势代态)
    total_kappa: float  # 总κ-release (认知维度: 知滤感)
    utility_score: float  # 效用评分 (价值维度: 算计统计算)
    psi_anchor: float  # ψ-Anchor置信度 (交互维度: 人机折叠)
    plan_actions: list[tuple]  # 完整动作序列

class SACache:
    """SA(Inflow)态势感知缓存 — 保留原始流贯供Re-Inflow"""
    game_state: dict  # 完整游戏观测
    uv_features: list  # EML 11维特征
    all_positions: set  # 全量位置采样(wall/changer/coin/player/goal)
    pruning_log: list  # Anti-Monotonicity剪枝日志


# kappa_budget_controller.py — 预算控制
class KappaBudgetController:
    """Anti-Monotonicity预算控制 + CR压缩比量化"""
    def should_collect_coin(self, current_kappa, coin_cost, kappa_reduction) -> bool:
        """价值维度(算计统计算): 仅在κ-reduction > action_cost时收集"""
        return kappa_reduction > coin_cost

    def compression_ratio(self, sa_cache, sm) -> float:
        """CR = IC(SA)/IC(SM) — 智能水平量化"""
        # IC(SA) = 信息基数(原始观测维度)
        # IC(SM) = 信息基数(压缩决策维度)
        # CR ≥ 10³ = 有效压缩

    def anti_monotonicity_check(self, kappa_before, kappa_after, tolerance=0.05) -> bool:
        """κ必须单调递减(含tolerance噪声容忍)"""
        return kappa_after <= kappa_before + tolerance
```

---

## 5. L1 失败根因 → 解决方案映射

### 5.1 诊断数据 → RG-Flow解释

| 诊断数据 | UV高熵态解释 | IR低熵态解决方案 |
|----------|-------------|-----------------|
| iter0-4: 贪婪收集coin+导航changer，浪费≈40步 | SA(Inflow)无压缩 → 单维度贪心 | SM四维度压缩: 认知(注入先验goal需rot=3) → 价值(coin仅在κ-reduction>cost时收集) |
| iter5: pos=(14,15) state matched但时间已耗尽 | κ-Snap归约达到IR态但能量耗尽 | RG-Flow预规划: 先算路径再执行，5s SA采样 + 8s SM压缩 + 5步IR执行 |
| BFS路径(14,15)→(14,40)只需5步但无法执行 | 时间预算已用完 | 时间维度(势代态): 相位拟合预分配时间预算，关键节点锚定 |
| 总时间45.08s > 45s预算 | 贪婪迭代无全局时间分配 | RG-Flow β(κ)驱动时间分配: κ快降阶段多分配时间 |
| 贪婪迭代误剪必需路径无回溯 | 无Dead-Zero检测 | Re-Inflow: 保留SA缓存 → Dead-Zero检测 → 回溯 → 重跑κ-Snap |

### 5.2 L1预期预规划路径

```
SA(Inflow)特征: pos(29,40), state(rot=0,color=1,shape=5), goal requires rot=3
  → EML 11维: wall(88), changer(3), coin(2), player, goal

SM(κ-Snap归约)四维度压缩:
  认知(知滤感): 先验注入 → goal需rot=3 → 需changer3次触发
  时间(势代态): 相位拟合 → trigger序列: rot=0→1→2→3, 关键节点changer@49,45
  价值(算计统计算): coin评估 → κ-reduction=0, action-cost>0 → SKIP(不收集)
  交互(人机折叠): ψ-Anchor=0.95(高置信) → 直接执行无需验证

Coset搜索 → 最优路径:
1. (29,40) → BFS→(49,45) [changer rhsxkxzdjz]
2. 离开(49,45)→再返回×3 [rotation 0→1→2→3]
3. (49,45) → BFS→(14,40) [goal, state matched]

Anti-Monotonicity检查:
- Step 1→2: κ从4维→3维 (rotation确定性增加) → PASS
- Step 2→3: κ继续递减 → PASS
- Coin收集: κ-reduction=0, action-cost>0 → SKIP (不收集)

κ-Entropy GaussEx校验:
Δκ = log(IC_UV/IC_IR) = log(4!/1!) ≈ 4.32 nat
|κ_entropy - 14.32| < 0.5 → plan通过
CR = IC(SA)/IC(SM) ≈ 10³ → 有效压缩

Dead-Zero监测: None (plan有效) → 无需Re-Inflow
```

---

## 6. 待确认问题

| # | 问题 | 选项 | 影响 | 默认建议 |
|---|------|------|------|----------|
| Q1 | **RG-Flow预规划是否替代所有3个Stage，还是仅替代Stage2?** | A: 完全替代3-stage → 2-phase RG-Flow; B: 仅替代Stage2(delta_state_bfs)，保留Stage1/3作为fallback | A风险高但简洁；B渐进安全 | **B** (渐进替代，保留fallback) |
| Q2 | **L5/L6 baseline需192/186步，RG-Flow预规划能否处理超长路径?** | A: 预规划上限200步; B: 分段预规划(segmented RG-Flow) | L5/L6可能需要多次coin收集+多changer序列 | **B** (分段预规划，每段独立坍缩) |
| Q3 | **Anti-Monotonicity的tolerance值: 严格(0.0)还是允许微小κ增加?** | A: strict=0.0 (任何κ增加即剪枝); B: tolerance=0.05 (允许5%噪声) | 严格可能错杀有效路径，宽松可能放过无效路径 | **B** (tolerance=0.05，有噪声容忍) |
| Q4 | **Sporadic Group分类何时执行: Level开始时一次性，还是动态迭代?** | A: 一次性分类(L0→Möbius); B: 动态分类(每步重新检测) | L2可能有color+rotation双维度，需要Monster策略 | **A** (一次性，EML感知在plan前执行) |
| Q5 | **已有10个理论模块是直接调用还是需要适配层?** | A: 直接调用(coset_search等已有接口); B: 需要适配层(game→IDO数据转换) | 模块是为ARC-3通用设计的，LS20是具体游戏场景 | **B** (需要LS20→IDO适配层) |
| Q6 | **Re-Inflow回溯的最大次数?** | A: 1次(只回溯一次); B: 2次(最多两轮回溯); C: 无限制 | 回溯开销大，过多回溯会消耗时间预算 | **B** (最多2轮Re-Inflow，之后fallback到Stage3) |
| Q7 | **κ-entropy参考值14.32 nat是否适用于LS20?** | A: 通用值14.32(来自α⁻¹≈137); B: 每关卡独立计算(L0≈1.39, L1≈4.32) | LS20不是标准ARC网格任务，信息基数约不同 | **B** (每关卡独立计算Δκ，14.32仅作为物理层参考) |

---

## 7. 实施风险

| 风险 | 严重性 | 缓解措施 |
|------|--------|----------|
| RG-Flow预规划在L5/L6超长路径上可能超时 | 中 | 分段预规划 + coin收集策略 |
| 10个理论模块接口与LS20数据格式不匹配 | 中 | LS20→IDO适配层 (`rg_flow_planner.py`) |
| Anti-Monotonicity严格剪枝可能错杀有效coin收集路径 | 低 | tolerance=0.05噪声容忍 + **Re-Inflow回溯兜底** |
| Coset搜索330陪集×24对称=7920路径计算量可能过大 | 低 | IDO预剪枝(50-70%) → 实际搜索≈2600-4000路径 |
| Re-Inflow回溯2轮后仍Dead-Zero | 低 | fallback到Stage3(κ-gradient/κ-PS) |
| CR压缩比在简单关卡(L0)可能达不到10³ | 低 | 简单关卡CR阈值降低到10²，只要求CR>1 |

---

## 8. 验收标准总表

| 关卡 | 当前状态 | 目标状态 | RHAE目标 |
|------|----------|----------|----------|
| L0 | PASS (plan_len=14, RHAE=115.0) | PASS (保持或改善) | ≥ 80 (当前115，保持) |
| L1 | FAIL (plan=None, 55.5s timeout) | PASS (<30s) | ≥ 80 |
| L2 | FAIL (堵塞) | PASS (<45s) | ≥ 80 |
| L3-L6 | 未测试 | PASS | ≥ 80 |

**总体目标**: `ALL_PASS: YES`，7/7关卡RHAE ≥ 80

---

*PRD v6.0 | 许清楚 (Xu) | 2026-06-27 | 吸纳势态知感(SA→SM)+Re-Inflow+四维度压缩*
