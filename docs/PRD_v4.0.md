# TOMAS-ARC3 Solver v4.0 — 产品需求文档 (PRD)

**版本**: v4.0  
**日期**: 2026-06-22  
**产品经理**: 许清楚 (Xu)  
**语言**: 中文  
**项目名**: `tomas_arc3_solver_v4`  
**技术栈**: Python 3.13 / arc_agi SDK / ReplayEngine / Z3-SMT / Nauty  

---

## 1. 原始需求复述

> "吸纳4篇微信公众号文章观点升级代码；代码太乱层累历史代码需重构优化；破解ARC-3必须摒弃梯度下降等连续优化方法，引入SAT/SMT约束求解、次模优化、分支定界搜索、EML图同构剪枝、群论轨道化简；deepcopy务必都改为Δ-状态重放；LS20 7关必须全部通关（pipeline模式）。"

---

## 2. 产品定义

### 2.1 产品目标

| # | 目标 | 度量标准 |
|---|------|----------|
| G1 | **零拷贝流贯架构** — 消除全部83处`copy.deepcopy`，改为Δ-State Replay | 0处deepcopy剩余；BFS节点内存从~2KB→~64B |
| G2 | **组合优化五阶段求解管线** — 摒弃连续优化，引入SAT/SMT+次模+群论轨道+EML同构+分支定界 | 五阶段全部可运行；κ-Snap搜索时间降50%+ |
| G3 | **LS20七关全通** — L0~L6在pipeline模式下全部通关 | LS20 score = 7/7 (RHAE≥115/关卡) |

### 2.2 用户故事

| # | 角色 | 故事 |
|---|------|------|
| US1 | ARC竞赛开发者 | 作为ARC-3竞赛开发者，我想用组合优化管线替代梯度搜索，以便在离散约束空间内高效找到正确程序 |
| US2 | 架构维护者 | 作为架构维护者，我想消除所有deepcopy改为Δ-State Replay，以便BFS能探索10K+节点而不耗尽内存 |
| US3 | 竞赛评测者 | 作为竞赛评测者，我想LS20全部7关通关，以便验证求解器在pipeline模式下具备完整的游戏求解能力 |
| US4 | 代码审计者 | 作为代码审计者，我想将src/agent/从40个文件精简到<15个核心模块，以便消除历史层累、降低维护成本 |
| US5 | 理论研究者 | 作为理论研究者，我想在κ-Snap搜索中注入刘机制S_rel优先级和三层共识验证，以便候选搜索按信息动力学自洽性排序 |

---

## 3. 需求池

### P0 — Must Have（必须实现，阻塞交付）

| ID | 需求 | 关联目标 | 验收标准 |
|----|------|----------|----------|
| P0-01 | **消除全部deepcopy** — 83处`copy.deepcopy`全部改为`Node(parent_id, action)` + `ReplayEngine`延迟物化 | G1 | `grep -r "copy.deepcopy" src/` 返回 0 |
| P0-02 | **五阶段求解管线骨架** — 群论轨道→SAT/SMT→次模贪心→分支定界→EML同构重放 | G2 | 五阶段类均可import；Stage1~5单元测试通过 |
| P0-03 | **LS20 L3~L6通关** — 在pipeline模式下补全L3~L6的求解逻辑 | G3 | LS20 7/7通关；每关RHAE≥115 |
| P0-04 | **刘机制S_rel优先级注入κ-Snap** — 用`S_rel = αM + βH + γP_noself`替代传统启发式排序候选 | G2,G5 | κ-Snap搜索按S_rel排序；对比基线搜索步数降≥30% |
| P0-05 | **三层共识验证** — 微视界(GaussEx≤1/6) + 中视界(ℐ加权) + 宏视界(叙事熵) | G2,G5 | 候选通过三层验证后才确认为解 |

### P1 — Should Have（重要但非阻塞）

| ID | 需求 | 关联目标 | 验收标准 |
|----|------|----------|----------|
| P1-01 | **Symbolic BitNet / Ternary Mask DSL** — +1=Keep, 0=Ψ-Cut, -1=Invert 三值掩码替代浮点权重 | G2 | `TernaryMaskDSL`类可运行；搜索空间压缩验证 |
| P1-02 | **EML AbsMean稀疏化** — 1/6阈值剪枝弱耦合边 | G2 | 边稀疏度≥83.3%；剪枝后求解精度不降 |
| P1-03 | **κ-Snap IC Early Stopping** — 达到目标信息基数即停止扩展 | G2 | 搜索平均步数降≥40% |
| P1-04 | **反单调性公理约束** — 信息丰富度不随表示基数单调增长 | G2 | IC(A∪B) ≤ IC(A)+IC(B) 校验通过 |
| P1-05 | **代码精简** — src/agent/从40文件→≤15核心模块 | G4 | 删除octonion_layers.py, nar_net_pytorch.py, gpu_backend.py等遗留文件；功能不退化 |
| P1-06 | **ψ-Anchor四条件检查** — IC足够大、GaussEx可校验、自指闭环、拓扑孤子稳定性 | G2,G5 | ψ-Anchor候选通过四条件后才锚定 |
| P1-07 | **动态因果EML** + κ-Snap反事实验证 | G2 | 反事实候选通过GaussEx安全边界校验 |
| P1-08 | **元认知四通路** — 证据累积/运动规划/运动推断/统计学习 | G2 | 四通路接口定义；至少2通路可运行 |

### P2 — Nice to Have（锦上添花）

| ID | 需求 | 关联目标 | 验收标准 |
|----|------|----------|----------|
| P2-01 | **DL+κ-Snap双螺旋架构** — 联想(右脑)+因果κ-Snap(左脑)交替演化 | G2,G5 | 双螺旋模式可选开启 |
| P2-02 | **Nauty签名EML图同构** — 用Nauty库计算canonical签名替代MD5布局哈希 | G2 | 同构检测时间降≥50% |
| P2-03 | **Karma Cache** — EML签名→结果缓存，不重算已求解子问题 | G2 | 缓存命中率≥30% |
| P2-04 | **precision_gamma自适应调节** — 根据任务难度动态调节GaussEx精度阈值 | G2 | gamma值随RHAE预算自适应 |
| P2-05 | **SyGuS + Z3/CVC5最小程序归纳** — 完整SyGuS语法前端 | G2 | 至少3个ARC任务通过SyGuS求解 |

---

## 4. 技术方案概要

### 4.1 组合优化五阶段管线架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TOMAS-ARC3 Solver v4.0 Pipeline                   │
│                                                                      │
│  Stage1: 群论轨道化简                                                │
│  ├── D_n / S_n 等价类检测                                            │
│  ├── 轨道代表元筛选（仅保留canonical form）                          │
│  └── 约束：rotation/mirror/reflection group → orbit reduction       │
│                                                                      │
│  Stage2: SAT/SMT约束求解                                             │
│  ├── SyGuS语法定义 → Z3/CVC5约束编码                                │
│  ├── 最小程序归纳（MDL优先）                                         │
│  └── 反单调性公理约束嵌入                                            │
│                                                                      │
│  Stage3: 次模优化贪心剪枝                                            │
│  ├── 边际效用递减 → 原语子集贪心选择                                 │
│  ├── EML AbsMean 1/6阈值稀疏化                                       │
│  ├── κ-Snap IC Early Stopping                                       │
│  └── S_rel = αM + βH + γP_noself 刘机制优先级排序                   │
│                                                                      │
│  Stage4: 分支定界 + A* Beam Search                                   │
│  ├── 容许启发式：h(n) = IC_remain                                    │
│  ├── GaussEx剪枝：GEX_FAIL_THRESHOLD 不可容分支直接丢弃              │
│  ├── Beam width = κ_budget / S_rel_min                               │
│  └── 三层共识验证：微视界→中视界→宏视界递进                          │
│                                                                      │
│  Stage5: EML图同构剪枝 + ReplayEngine                                │
│  ├── Nauty/MD5 canonical签名 → 同构检测                              │
│  ├── Karma Cache → 不重算已求解子结构                                │
│  ├── Δ-State Replay → 零拷贝状态重放                                 │
│  └── ψ-Anchor四条件锚定确认                                          │
│                                                                      │
│  ── 验证层 ─────────────────────────────────────────────────────────│
│  GaussExVerifier (1/6阈值) + LayoutHasher + 三层ConsensusGate       │
│                                                                      │
│  ── 预算层 ─────────────────────────────────────────────────────────│
│  RHAEBudgetController (RHAE = (H/A)²) + CoinCollector              │
│                                                                      │
│  ── 状态层 ─────────────────────────────────────────────────────────│
│  Node(parent_id, action) + ReplayEngine + TernaryMaskDSL            │
│                                                                      │
│  ── 双螺旋（可选）─────────────────────────────────────────────────│
│  DL联想通路(右脑) ⇄ κ-Snap因果通路(左脑)                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Symbolic BitNet — Ternary Mask DSL

| 掩码值 | 符号 | 语义 | IDO对应 |
|--------|------|------|---------|
| +1 | Keep | 保留当前变换 | 流贯直通通道 |
| 0 | Ψ-Cut | Dead-Z零切断（熔断） | LSTM forget gate → 信息丢弃 |
| -1 | Invert | 反转当前变换 | 流贯反转通道 |

搜索空间压缩：从 `2^N`（浮点权重组合）→ `3^N`（三值掩码）→ 轨道化简后仅 `3^K`（K=轨道代表元数）。

### 4.3 三层共识验证

| 视界 | 验证方法 | 阈值 | 语义 |
|------|----------|------|------|
| 微视界 | GaussEx硬校验 | ≤ 1/6 ≈ 0.167 | 卞氏5/6饱和度：5/6维度必须自洽 |
| 中视界 | ℐ加权信息基数 | IC ≥ IC_target | 信息基数是否达标 |
| 宏视界 | 叙事熵 | H_narrative ≤ H_max | 全局叙事是否自洽稳定 |

候选解必须**依次通过三层**才被确认为最终答案。任一层失败即回溯。

### 4.4 Δ-State Replay 替代 deepcopy

当前问题：
- `game_solvers.py`: 58处deepcopy
- `universal_solver_pipeline.py`: 20处deepcopy  
- `neural_dsl.py`: 3处deepcopy
- `delta_state.py`: 1处deepcopy（自身需清理）
- 总计: **83处**

替换方案：
- `Node(parent_id, action)` — 只存储父节点引用+动作，内存~64B/node
- `ReplayEngine` — 从root_state开始逐动作replay物化，配合LRU缓存(MAX_REPLAY_CACHE=128)
- `LayoutHasher` — MD5哈希做O(1)去重，替代O(n)状态比较
- **性能预期**: BFS可探索10K+节点（当前55节点就耗尽）；内存占用降低30倍

### 4.5 代码精简计划

当前 `src/agent/` 有 **40个.py文件**，包含大量历史遗留：

| 类别 | 待删文件 | 理由 |
|------|----------|------|
| 八元数遗留 | octonion_layers.py, octonion_layers_optimized.py, octonion_layers_simple.py, octonion_tensor.py, octonion_resnet.py | v3.7时期的八元数编码层，已被neural_dsl取代 |
| NAR遗留 | nar_net_pytorch.py, nar_net_core.py, nar_bridge.py, nar_oracle_adapter_mvp.py | 神经联想推理网络原型，未实际使用 |
| GPU遗留 | gpu_backend.py | CUDA后端，离线模式不需要 |
| 旧规划 | planner_agent.py, planner_agent_v6.py, dopamine_explorer.py | 早期规划器，已被Δ-State BFS取代 |
| 旧求解 | enhanced_architecture.py, deep_architecture.py, self_learning.py | 历史架构，功能已合并 |
| 旧适配 | universal_oracle_adapter.py | 已被game_solvers中的oracle逻辑取代 |
| 测试文件 | test_octonion_optimized.py, test_e2e_full.py, test_real_game.py | 应移至tests/目录 |

**目标**: src/agent/ 精简至 ≤15个核心模块：
- `tomas_core.py` (核心框架)
- `delta_state.py` (Δ-State引擎)
- `game_solvers.py` (精简后≤3000行)
- `tomas_learner.py` (精简后≤2000行)
- `neural_dsl.py` (保留并扩展TernaryMask)
- `rhae_controller.py` (RHAE预算)
- `game_configs.py` (游戏配置)
- `game_profiles.py` (游戏profile)
- `oracle_adapters.py` (Oracle适配)
- `combo_pipeline.py` (新增：五阶段组合优化管线)
- `group_orbit.py` (新增：群论轨道化简)
- `sat_smt_solver.py` (新增：SAT/SMT约束求解)
- `submodular_optimizer.py` (新增：次模优化)
- `branch_bound_search.py` (新增：分支定界+A*)
- `eml_iso_pruner.py` (新增：EML图同构剪枝+Nauty)

### 4.6 核心文件行数精简目标

| 文件 | 当前行数 | 目标行数 | 精简策略 |
|------|----------|----------|----------|
| game_solvers.py | 9458 | ≤2500 | 去除历史求解器；统一到Δ-State BFS + combo_pipeline |
| tomas_learner.py | 10670 | ≤2000 | 去除RL层累代码；保留Sleep-Step+EpisodeTrace |
| neural_dsl.py | 1346 | ≤1500 | 扩展TernaryMask；保留§10-§13 |
| tomas_core.py | 837 | ≤600 | 精简接口；去除旧agent逻辑 |
| universal_solver_pipeline.py | 2069 | ≤800 | 精简为combo_pipeline入口 |

---

## 5. UI设计稿

N/A — 纯代码项目，无UI界面。交互通过CLI `main.py` 和 Kaggle notebook接口。

---

## 6. 待确认问题

| # | 问题 | 影响范围 | 建议 |
|----|------|----------|------|
| Q1 | Z3/CVC5依赖是否纳入requirements.txt？ | P1-05(P2-05) | 建议Z3-Python作为P0依赖；CVC5作为P2可选 |
| Q2 | Nauty图同构库的Python绑定选型？ | P2-02 | 建议先用networkx.is_isomorphic()作为基线；Nauty(nauty-python)作为P2优化 |
| Q3 | LS20 L3~L6的游戏结构是否与L0~L2一致（click vs keyboard）？ | P0-03 | 需实际运行arc_agi SDK探查L3~L6的game结构 |
| Q4 | SyGuS前端需要自定义语法还是用标准SyGuS-IF？ | P2-05 | 建议先自定义轻量SyGuS子集，标准IF作为P2扩展 |
| Q5 | κ-Snap S_rel的α/β/γ系数如何初始化？ | P0-04 | 建议从信息基数估计自动推导；或固定α=0.5,β=0.3,γ=0.2作为默认 |
| Q6 | game_solvers.py精简后，现有38+个game-specific solver函数如何归并？ | P1-05 | 建议按游戏类型归入combo_pipeline的Stage路由，每个game_id映射到对应Stage组合 |
| Q7 | 八元数编码(octonion_hyperedge)是否保留？ | P1-05 | 建议：核心模块(src/core/octonion_hyperedge.py)保留，但agent层的octonion_layers系列删除 |
| Q8 | Δ-State Replay在grid mode下的物化如何保证不引入deepcopy？ | P0-01 | grid mode用numpy view + transformation sequence，避免拷贝原始grid |
| Q9 | Karma Cache的持久化策略？ | P2-03 | 建议默认session-level内存缓存；可选JSON文件持久化 |
| Q10 | 双螺旋DL+κ-Snap的交替频率如何控制？ | P2-01 | 建议默认每3步κ-Snap穿插1步DL联想；频率可配置 |

---

## 7. 理论映射总结

| 微信文章 | 核心理论 | PRD需求映射 |
|----------|----------|-------------|
| 文章1: IDO + 刘机制 | S_rel优先级、三层共识、ψ-Anchor | P0-04, P0-05, P1-06 |
| 文章2: Symbolic BitNet | Ternary Mask DSL、EML稀疏化、IC Early Stopping、反单调性 | P1-01, P1-02, P1-03, P1-04 |
| 文章3: 具身主动推理 | 动态因果EML、反事实验证、元认知四通路、双螺旋 | P1-07, P1-08, P2-01 |
| 文章4: 组合优化范式 | 群论轨道→SAT/SMT→次模→分支定界→EML同构 | P0-02, P2-02, P2-03, P2-05 |

---

*文档结束 — 许清楚 (Xu) · 产品经理*
