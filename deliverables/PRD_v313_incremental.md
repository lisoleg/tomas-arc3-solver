# 增量PRD — TOMAS ARC-AGI-3 Solver v3.13.0 升级

**产品经理**: 许清楚（Xu）
**日期**: 2026-06-22
**版本**: v3.13.0-incremental-PRD
**团队**: software-tomas-v313-upgrade

---

## 1. 产品目标

将TOMAS ARC-AGI-3 Solver的三个核心Kaggle提交文件同步升级到v3.13.0版本，同时攻克3个优先零分游戏(ka59/ar25/tn36)的L0关卡，提升整体RHAE评分从当前4.6到目标≥15。

## 2. 用户故事

### US-1: 版本一致性
> 作为Kaggle提交者，我需要三个核心文件(kaggle_my_agent.py/library.json/kaggle_solution_v04.ipynb)版本号统一为v3.13.0，以便提交时版本标识清晰且代码逻辑对齐。

### US-2: 零分游戏攻克
> 作为竞赛选手，我需要ka59/ar25/tn36三个游戏的L0 RHAE>0，以便从18个零分游戏中至少拿下3个，提升总分从801→≥1134(假设3个×115=345新增)。

### US-3: Notebook升级
> 作为提交管道维护者，我需要kaggle_solution_v04.ipynb从v3.10.0升级到v3.13.0，以便notebook中的Agent代码与kaggle_my_agent.py完全一致，避免版本漂移。

## 3. 需求池（优先级排序）

### P0 — 必须完成（阻塞提交）

| # | 需求 | 描述 | 验收标准 |
|---|------|------|---------|
| P0-1 | kaggle_my_agent.py v3.11→v3.13 | 升级Agent文件版本标识+新增策略代码 | 文件头版本标识为v3.13.0；新增v3.12/v3.13策略集成；ARC3_REPLAY_ORACLE新增ka59/ar25/tn36数据 |
| P0-2 | library.json v3.11→v3.13 | 升级DSL宏库版本+新增宏定义 | `_schema_version`="3.13.0"；新增模块列表与v3.13策略对齐 |
| P0-3 | kaggle_solution_v04.ipynb v3.10→v3.13 | 升级Kaggle Notebook提交文件 | %%writefile内容与kaggle_my_agent.py v3.13.0完全一致；cell-0版本标识为v3.13.0 |
| P0-4 | 攻克ka59零分游戏 | ka59 L0 RHAE>0 | ka59 L0至少完成1次level transition，RHAE≥1（理想≥80） |
| P0-5 | 攻克ar25零分游戏 | ar25 L0 RHAE>0 | ar25 L0至少完成1次level transition，RHAE≥1（理想≥80） |
| P0-6 | 攻克tn36零分游戏 | tn36 L0 RHAE>0 | tn36 L0至少完成1次level transition，RHAE≥1（理想≥80） |

### P1 — 应当完成（显著提升质量）

| # | 需求 | 描述 | 验收标准 |
|---|------|------|---------|
| P1-1 | ka59/ar25/tn36 L0 RHAE≥80 | 三个目标游戏L0满分 | 三个游戏L0均达到RHAE=115.0 |
| P1-2 | ARC3_REPLAY_ORACLE扩展 | 新增ka59/ar25/tn36的L0-L3 replay数据 | Oracle中有≥3个游戏的完整replay序列 |
| P1-3 | DFS通用求解器增强 | 针对ka59/ar25/tn36的solve_generic_dfs调优 | DFS在这3个游戏上12s内找到L0解 |
| P1-4 | 18个零分游戏批量分析 | 诊断剩余15个零分游戏的根因 | 输出根因报告（placeholder/逻辑错误/机制复杂） |

### P2 — 可以延后（锦上添花）

| # | 需求 | 描述 |
|---|------|------|
| P2-1 | 全量25游戏RHAE基准测试 | 验证v3.13.0在所有25游戏上的RHAE |
| P2-2 | Git提交+推送 | v3.13.0正式版本Git提交和推送 |
| P2-3 | CHANGELOG更新 | 新增v3.13.0变更条目 |
| P2-4 | 论文更新 | v3.13.0策略描述添加到论文 |

---

## 4. 现状分析

### 4.1 版本差距矩阵

| 文件 | 当前版本 | 目标版本 | 版本跳跃 | 差距评估 |
|------|---------|---------|---------|---------|
| kaggle_my_agent.py | v3.11.0 | v3.13.0 | +2 minor | 中等 — 需集成v3.12/v3.13策略 |
| library.json | v3.11.0-dev | v3.13.0 | +2 minor | 中等 — 需新增宏模块 |
| kaggle_solution_v04.ipynb | v3.10.0 | v3.13.0 | +3 minor | 较大 — 需重写%%writefile为v3.13.0 Agent代码 |

### 4.2 已有策略层级（v3.11.0）

kaggle_my_agent.py当前包含26条策略：
- ARC3 Replay Oracle (Phase 0)
- PhysicalCompactificationReduction (Φ_phys)
- ASD Anomaly Detection
- 3-Life Strategy
- Sleep-Step Trigger + Algorithm 1
- Thinker-Performer Pipeline
- Interactive ZKP Loop
- PersistentKVCache
- Clean Latent Writeback
- Ω_topo Breakthrough Trigger
- MoonshineProver
- GibbsEnsemble
- IDOFramework
- QuantumContextualEvaluator
- CHLIsomorphism
- UniverseZKP
- 等

### 4.3 v3.12→v3.13 新增策略（需确认）

可能的新增策略（基于CHANGELOG和代码模块推断）：
- **v3.12**: 拟阵剪枝(MatroidPruning)增强、条件ΔT发现(ConditionalDeltaTDiscovery)、自适应Sleep-Step预算
- **v3.13**: DFS回溯求解器增强(solve_generic_dfs)、零分游戏专用Adapter、ARC3 Replay扩展

### 4.4 零分游戏ka59/ar25/tn36状态

| 游戏 | 游戏类型 | Baseline步数 | 超时时间 | 当前Phase | 已有调试代码 | 根因推断 |
|------|---------|------------|---------|----------|------------|---------|
| ka59 | 未知(60s+超时) | 28 | 60s+ | Phase 1-7全部失败 | 无 | 可能是复杂机制/缺少专用Adapter |
| ar25 | 未知(60s+超时) | 32 | 60s+ | Phase 1-7全部失败 | 无 | 可能是复杂机制/缺少专用Adapter |
| tn36 | click游戏 | 32 | 60.1s | Phase 1-7全部失败 | debug_tn36_pipeline.py/debug_tn36_state.py/debug_tn36_v2.py/explore_tn36_orderings.py | 已有大量调试但未攻克，可能是solver逻辑缺陷 |

---

## 5. 版本兼容性约束

1. **Kaggle提交环境**: arc-agi包版本必须与Kaggle competition wheels兼容（python-dotenv + arc-agi --no-deps）
2. **自包含约束**: kaggle_my_agent.py必须完全自包含，不引用任何本地项目文件（src/下模块不能直接import）
3. **Notebook一致性**: kaggle_solution_v04.ipynb的%%writefile内容必须与kaggle_my_agent.py逐行一致
4. **ARC3框架合约**: Agent必须继承`agents.agent.Agent`，类名`MyAgent`，实现`is_done()`和`choose_action()`
5. **library.json schema**: `_schema_version`必须与Agent版本号对齐，`core_modules`列表必须与Agent中引用的模块一致

---

## 6. UI设计稿

> 本项目为纯代码升级+求解器优化，无前端UI组件，不需要UI设计稿。

---

## 7. 待确认问题

| # | 问题 | 影响 | 建议 |
|---|------|------|------|
| Q1 | v3.12/v3.13具体新增了哪些策略？ | 决定kaggle_my_agent.py升级范围 | 需扫描src/目录代码确认最新模块 |
| Q2 | ka59/ar25的游戏机制是什么？ | 决定专用Adapter设计 | 需运行环境_files/下的游戏源码分析 |
| Q3 | tn36的已有调试代码为何失败？ | 影响tn36攻克策略 | 需review debug_tn36系列代码的失败原因 |
| Q4 | ARC3_REPLAY_ORACLE能否获取ka59/ar25/tn36的replay数据？ | 决定是否用Oracle路线 | 如有replay数据则Phase 0直解，无需Adapter/DFS |
| Q5 | kaggle_solution_v04.ipynb升级策略：重写%%writefile还是增量patch？ | 影响工程工作量 | 建议重写——因为v3.10→v3.13跨越3个版本，增量patch过于复杂 |

---

## 8. 风险评估

| 风险 | 严重性 | 概率 | 缓解措施 |
|------|--------|------|---------|
| v3.12/v3.13策略代码丢失或不完整 | 高 | 低 | 从src/源码反向提取策略逻辑 |
| ka59/ar25/tn36游戏机制过于复杂无法攻克 | 高 | 中 | fallback：仅实现ARC3 Replay Oracle数据 |
| Notebook升级引入语法错误 | 中 | 中 | 对比kaggle_my_agent.py逐行验证 |
| 版本号不一致导致Kaggle评分异常 | 中 | 低 | 三文件版本号统一检查 |
| DFS求解器在60s内无法找到解 | 高 | 高 | 优先使用Oracle replay数据，DFS作为fallback |

---

*PRD创建时间: 2026-06-22 | 产品经理: 许清楚(Xu) | 团队: software-tomas-v313-upgrade*
