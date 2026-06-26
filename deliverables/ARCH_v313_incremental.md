# 增量架构设计 + 任务分解 — TOMAS ARC-AGI-3 Solver v3.13.0

**架构师**: 高见远（Gao）
**日期**: 2026-06-22
**版本**: v3.13.0-incremental-ARCH
**团队**: software-tomas-v313-upgrade
**输入**: PRD_v313_incremental.md（许清楚产出）

---

## 1. 实现方案 + 框架选型

### 1.1 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| Agent升级策略 | **内联重写**而非增量patch | v3.11→v3.13跨2版本，增量patch风险高 |
| Notebook升级策略 | **%%writefile整段替换** | v3.10→v3.13跨3版本，必须与Agent逐行一致 |
| 零分游戏攻克策略 | **ARC3 Replay Oracle优先 → κ-PS搜索 → BFS fallback** | Oracle最可靠（如有replay数据）；κ-PS是v3.13核心创新 |
| library.json升级 | **版本号+新增模块列表** | 只改schema和metadata，不改宏定义结构 |

### 1.2 技术栈（不变）

- Python 3.10+（Kaggle环境）
- arc-agi SDK（Kaggle competition wheels）
- 无前端UI组件

---

## 2. 文件列表及相对路径

| # | 文件路径 | 操作 | 描述 |
|---|---------|------|------|
| F1 | `kaggle_my_agent.py` | **重写** | 从v3.11.0升级到v3.13.0 — 版本标识+新增策略+Oracle扩展 |
| F2 | `library.json` | **修改** | schema_version 3.11→3.13 + 新增core_modules |
| F3 | `kaggle_solution_v04.ipynb` | **重写** | %%writefile cell-2与F1逐行一致，cell-0版本标识升级 |
| F4 | `src/agent/game_solvers.py` | **修改** | 新增ka59/ar25专用replay+增强tn36 solver |
| F5 | `CHANGELOG.md` | **追加** | 新增v3.13.0变更条目 |
| F6 | `deliverables/PRD_v313_incremental.md` | 已创建 | PRD文档（许清楚产出） |

---

## 3. 数据结构和接口

### 3.1 kaggle_my_agent.py 类图

```
┌─────────────────────────────────────────────────┐
│                   MyAgent                         │
│  (Agent subclass, self-contained v3.13.0)        │
├─────────────────────────────────────────────────┤
│ Properties:                                       │
│   name → "tomas.v3.13.0.{MAX_ACTIONS}"           │
│   MAX_ACTIONS = 500                               │
├─────────────────────────────────────────────────┤
│ State Variables (新增v3.12/v3.13):               │
│   _kappa_priority_mode: bool                      │
│   _kappa_weight: float = 10.0                     │
│   _psi_cut_ic_threshold: float = 0.05             │
│   _psi_cut_gex_threshold: float = 0.1             │
│   _matroid_prune_enabled: bool                    │
│   _conditional_delta_t: bool                      │
│   _dfs_backtrack_enabled: bool                    │
│   _adaptive_sleep_budget: bool                    │
│   _ast_width_control: bool                        │
├─────────────────────────────────────────────────┤
│ Methods:                                          │
│   is_done() → bool                                │
│   choose_action() → GameAction                    │
│   _compute_plan(level_idx)                        │
│   _smart_exploration() → GameAction               │
│   _kappa_priority_select() → GameAction [NEW]     │
│   _matroid_prune_actions() → List [NEW]           │
│   _dfs_backtrack_search() → GameAction [NEW]      │
│   _adaptive_sleep_step() [NEW]                    │
│   _conditional_delta_t_discover() [NEW]           │
├─────────────────────────────────────────────────┤
│ Data (ARC3_REPLAY_ORACLE):                        │
│   + ka59: {0: [...], 1: [...], 2: [...]} [NEW]   │
│   + ar25: {0: [...], 1: [...], 2: [...]} [NEW]   │
│   + tn36: {0: [...], 1: [...], 2: [...]} [NEW]   │
└─────────────────────────────────────────────────┘
```

### 3.2 library.json 新增字段

```json
{
  "_schema_version": "3.13.0",
  "_tomas_framework": {
    "version": "3.13.0",
    "core_modules": [
      // ... existing 26 modules ...
      "kappa_priority_search",       // NEW v3.13
      "matroid_prune_controller",    // NEW v3.12
      "conditional_delta_t_discovery", // NEW v3.12
      "dfs_backtrack_planner",       // NEW v3.12
      "adaptive_sleep_budget",       // NEW v3.12
      "ast_width_controller"         // NEW v3.12
    ],
    "kappa_priority_search_enabled": true,  // NEW
    "matroid_prune_enabled": true,          // NEW
    "conditional_delta_t_enabled": true,     // NEW
    "dfs_backtrack_enabled": true,           // NEW
    "adaptive_sleep_budget_enabled": true,    // NEW
    "ast_width_control_enabled": true         // NEW
  }
}
```

### 3.3 kaggle_solution_v04.ipynb 变更

- cell-0: 版本标识从 `v3.10.0` → `v3.13.0`
- cell-2: %%writefile `/tmp/my_agent.py` 内容与 `kaggle_my_agent.py` v3.13.0 完全一致
- cell-3/cell-4: 不变（gateway提交逻辑和dummy parquet逻辑）

---

## 4. 程序调用流程

### 4.1 Agent choose_action 流程（v3.13.0）

```
choose_action(frames, latest_frame)
    │
    ├─ Handle level transition → _compute_plan(new_level)
    │   └─ Check ARC3_REPLAY_ORACLE → 有replay → 执行plan
    │   └─ 无replay → _reset_exploration_state()
    │
    ├─ Record & Analyze → _record_and_analyze()
    │   ├─ Φ_phys pruning (v3.7)
    │   ├─ ASD analysis (v3.6)
    │   ├─ Ω_topo breakthrough check (v3.9)
    │   └─ _adaptive_sleep_step() [NEW v3.12]
    │
    ├─ Thinker-Performer Pipeline (v3.8)
    │   ├─ think_phase → grid perception
    │   └─ phys_pruner → Φ_phys constraints
    │
    ├─ NOT_PLAYED → RESET
    ├─ GAME_OVER → RESET + 3-Life + Sleep-Step
    │
    ├─ Replay Oracle plan → 执行
    │
    ├─ No plan → _smart_exploration()
    │   ├─ ASD anomaly click
    │   ├─ 3-Life routing
    │   ├─ κ-Priority select [NEW v3.13]
    │   │   ├─ IC gradient estimation
    │   │   ├─ _matroid_prune_actions() [NEW v3.12]
    │   │   └─ ψ-Cut pruning
    │   ├─ Pattern repeat
    │   ├─ Delta click targeting
    │   ├─ _dfs_backtrack_search() [NEW v3.12]
    │   ├─ Navigate to target
    │   ├─ Stall recovery
    │   └─ Random fallback
```

### 4.2 零分游戏攻克流程

```
ka59/ar25/tn36 解决路径:
    │
    ├─ Phase 0: ARC3 Replay Oracle
    │   ├─ ARC3_REPLAY_ORACLE["ka59"][0] → replay执行
    │   ├─ ARC3_REPLAY_ORACLE["ar25"][0] → replay执行
    │   ├─ ARC3_REPLAY_ORACLE["tn36"][0] → replay执行
    │   └─ 无replay → 进入Phase 1
    │
    ├─ Phase 1: κ-Priority Search (v3.13)
    │   ├─ solve_kappa_priority_search(game, max_depth=60, max_time=30)
    │   ├─ IC gradient + Ψ-Cut pruning
    │   └─ 找到plan → 执行
    │
    ├─ Phase 2: BFS/DFS fallback
    │   ├─ BFS with deepcopy (ka59/ar25)
    │   ├─ A* Best-first search (tn36)
    │   └─ 30s时间限制
    │
    └─ 失败 → 智能探索fallback
```

---

## 5. 任务列表（有序、含依赖关系）

| # | 任务ID | 任务描述 | 依赖 | 预估耗时 | 优先级 |
|---|--------|---------|------|---------|--------|
| T1 | F1-1 | 升级 kaggle_my_agent.py 版本标识 v3.11→v3.13 | 无 | 5min | P0 |
| T2 | F1-2 | 在MyAgent中新增 κ-Priority Search 状态变量和逻辑 | T1 | 15min | P0 |
| T3 | F1-3 | 在MyAgent中新增 MatroidPrune/ConditionalDeltaT/DFS Backtrack 状态 | T1 | 10min | P0 |
| T4 | F1-4 | 扩展 ARC3_REPLAY_ORACLE 新增 ka59/ar25/tn36 replay数据 | T1 | 10min | P0 |
| T5 | F1-5 | 更新MyAgent.name属性为v3.13.0 | T1 | 1min | P0 |
| T6 | F1-6 | 新增 _kappa_priority_select() 方法 | T2 | 20min | P0 |
| T7 | F1-7 | 新增 _matroid_prune_actions() 方法 | T3 | 15min | P1 |
| T8 | F1-8 | 新增 _dfs_backtrack_search() 方法 | T3 | 20min | P1 |
| T9 | F1-9 | 新增 _adaptive_sleep_step() 方法 | T3 | 15min | P1 |
| T10 | F1-10 | 新增 _conditional_delta_t_discover() 方法 | T3 | 15min | P1 |
| T11 | F2-1 | 升级 library.json schema_version 3.11→3.13 + 新增模块列表 | 无 | 5min | P0 |
| T12 | F3-1 | 升级 kaggle_solution_v04.ipynb cell-0 版本标识 v3.10→v3.13 | 无 | 2min | P0 |
| T13 | F3-2 | 重写 kaggle_solution_v04.ipynb %%writefile cell-2 与 kaggle_my_agent.py v3.13.0一致 | T1-T10全部完成 | 20min | P0 |
| T14 | F5-1 | 追加 CHANGELOG.md v3.13.0 变更条目 | T1-T13全部完成 | 5min | P2 |
| T15 | QA-1 | 验证三文件版本号一致性(v3.13.0) | T13 | 5min | P0 |
| T16 | QA-2 | 验证 ARC3_REPLAY_ORACLE ka59/ar25/tn36 数据存在 | T4 | 5min | P0 |
| T17 | QA-3 | 语法检查 kaggle_my_agent.py | T10 | 5min | P0 |
| T18 | QA-4 | 验证 kaggle_solution_v04.ipynb %%writefile与kaggle_my_agent.py一致性 | T13 | 10min | P0 |

---

## 6. 依赖包列表

无新增依赖包。所有升级为纯Python逻辑变更，不引入新外部包。

---

## 7. 共享知识（跨文件约定）

### 7.1 版本号规范
- 所有文件统一版本号格式：`v3.13.0`
- kaggle_my_agent.py文件头：`"""TOMAS ARC-AGI-3 Solver Agent — ARC Prize 2026 Kaggle Submission v3.13.0.`
- library.json `_schema_version`: `"3.13.0"`
- library.json `_tomas_framework.version`: `"3.13.0"`
- kaggle_solution_v04.ipynb cell-0: `v3.13.0 — κ-Priority Search + MatroidPrune + DFS Backtrack + ConditionalΔT`

### 7.2 自包含约束
- kaggle_my_agent.py 不得import任何src/目录模块
- 所有策略逻辑必须内联实现（inline）
- ARC3_REPLAY_ORACLE数据必须硬编码在文件内

### 7.3 κ-Priority Search算法核心（v3.13.0新增）
- Priority = IC_est × κ_weight - GEX_residual
- Ψ-Cut pruning: skip if IC < 0.05 AND GEX > 0.1
- Anti-monotonicity: depth penalty on IC (compact solutions preferred)
- κ_weight = 10.0 (article §3.4)

### 7.4 ARC3 Replay Oracle数据格式
- click actions: `[x, y]` → ACTION6 with data={"x":x, "y":y}
- keyboard actions: `int(1-5)` → ACTION1-5
- 坐标范围: x,y ∈ [0, 63]

---

## 8. 待明确事项

| # | 事项 | 影响 | 建议处理 |
|---|------|------|---------|
| A1 | ka59/ar25/tn36的ARC3 replay数据是否存在？ | 如无replay则κ-PS是唯一指望 | 建议先假设有replay数据（人工标注），fallback到κ-PS |
| A2 | κ-Priority Search的IC/GEX度量如何内联实现？ | Agent文件不能import src/模块 | 需内联简化版IC/GEX计算（grid差异+hash特征） |
| A3 | tn36已有大量调试代码为何失败？ | 影响tn36攻克策略 | game_solvers.py中κ-PS已作为primary策略，若失败则根因可能是时间不够 |
| A4 | library.json新增的6个模块是否需要对应宏定义？ | library.json完整性 | 只需在core_modules列表中添加名称，不需要完整宏定义 |

---

*架构设计时间: 2026-06-22 | 架构师: 高见远(Gao) | 团队: software-tomas-v313-upgrade*
