# PRD: 四层混合搜索架构增量重构 (Hybrid Search Architecture)

> Author: 许清楚 (Xu, PM) | Date: 2026-06-22 | Version: 1.0
> Based on: 微信公众号文章《面向 ARC-AGI-3 的混合搜索架构：从 κ-优选到 Wall-BFS 的协同设计与形式化分析》(章锋)
> Target: 175关卡 (25游戏×7关卡) 全部 RHAE=115.0 满分通关

---

## 1. 项目信息

- **Language**: 中文
- **Programming Language**: Python (Vite+React 不适用，本项目是纯 Python ARC-AGI-3 solver)
- **Project Name**: `hybrid_search_architecture`
- **原始需求复述**: 按章锋文章的四层混合搜索方法论（L1状态枚举→L2结构剪枝→L3残差评估→L4决策融合），重构 `game_solvers.py` 中所有25个游戏的搜索架构，从当前碎片化的单体solver模式迁移到统一管线，最终实现175关卡全部满分通关。

---

## 2. 产品目标

**一句话目标**: 将25个游戏的搜索从碎片化单体solver统一为四层混合搜索管线（L1→L2→L3→L4），用 Δ-State Replay 替代 deepcopy瓶颈，以 κ-优选决策融合统一选择策略，攻克5个零分游戏并提升LS20等高分潜力游戏至满分。

**三大正交目标**:

1. **架构统一**: 所有25游戏共享同一 L1→L2→L3→L4 管线框架，消除当前每个solver独立实现的碎片化局面
2. **性能突破**: Δ-State Replay 全局替代 deepcopy，搜索效率提升 10× 以上（当前 deepcopy ~50ms/次 vs Replay ~1ms/次）
3. **满分通关**: 5个零分游戏（ka59/tn36/cn04/ar25/sb26）攻克 + LS20 L1-L6通关 + 其他游戏提升至RHAE=115.0

---

## 3. 用户故事

- **US1**: 作为ARC求解器开发者，我需要统一的四层搜索架构，以便所有25个游戏共享同一管线（L1→L2→L3→L4），而非各自独立实现搜索策略
- **US2**: 作为求解器运维者，我需要 Δ-State Replay 替代 deepcopy，以便搜索效率提升10倍（1ms vs 50ms），在45秒时间预算内探索更多状态
- **US3**: 作为Kaggle参赛者，我需要175关卡全部满分（RHAE=115.0），以便获得竞赛第一名
- **US4**: 作为ka59开发者，我需要Wall-BFS+Dead-Zero熔断+Wall-Ride豁免的专用L1配置，以便推箱子关卡能从RHAE=0提升至700+
- **US5**: 作为ls20开发者，我需要BFS路径规划+陷阱规避残差+κ-优选安全着陆的配置，以便从RHAE=115提升至805

---

## 4. 需求池

### P0 — 必须 (Must Have)

| ID | 需求 | 验收标准 |
|----|------|----------|
| P0-1 | **四层混合搜索引擎框架** | 新模块 `hybrid_search_engine.py`，提供 `HybridSearchPipeline` 类，接口 `solve(game, game_id, level_idx) → plan`，内部按 L1→L2→L3→L4 管线执行 |
| P0-2 | **Δ-State Replay 全局统一** | `ReplayEngine`（已存在于 `delta_state.py`）从仅 `solve_ls20` 使用扩展到所有BFS/DFS/kappa搜索节点扩展，消除 `solve_generic_bfs` 中每节点 `copy.deepcopy(g)` |
| P0-3 | **κ-优选决策融合 (L4)** | 所有L1候选经L2剪枝+L3评估后，统一通过 `confidence = 1 - η/δ_K` 升序选择最优宏动作，取代当前各solver的独立启发式选择 |
| P0-4 | **5个零分游戏攻克** | ka59(RHAE 0→700+), tn36(0→通关), cn04(0→通关), ar25(0→通关), sb26(0→通关)，每个游戏至少L0关卡通过 |
| P0-5 | **solve_game() 路由重构** | `solve_game()` 入口从8-phase cascade简化为: Phase 0→HybridSearchPipeline (统一四层), Phase 1→通用fallback（仅对无专用配置的游戏） |

### P1 — 重要 (Should Have)

| ID | 需求 | 验收标准 |
|----|------|----------|
| P1-1 | **Wall-BFS 推箱子专用** | ka59 使用 Wall-BFS（贴墙宽度优先）替代当前 Newton-push BFS，含物理检查、Dead-Zero熔断、Wall-Ride豁免 |
| P1-2 | **Prime-Signature 快速拒识** | tr87/ft09 等规则发现类游戏使用数论指纹(Prime-Signature)快速排除伪规则候选 |
| P1-3 | **拟阵约束 (Union-Find)** | sb26/sk48 等排序类游戏使用 Union-Find 回路检测替代 `is_valid_poset_order()` |
| P1-4 | **LS20 L1-L6 通关** | solve_ls20 的7关卡全部通过，RHAE 从115→805+ |
| P1-5 | **组合对称剪枝** | ka59/cn04/sp80 等使用八元数编码旋转/镜像归一化剪枝等价状态 |
| P1-6 | **Dead-Zero 熔断全局化** | 所有BFS/kappa搜索节点扩展时统一检查 GaussEx 残差 η，η ≥ δ_K 时熔断（不扩展该分支） |

### P2 — 可选 (Nice to Have)

| ID | 需求 | 验收标准 |
|----|------|----------|
| P2-1 | **八元数对称归一化** | 所有状态哈希 `_game_state_hash()` 加入八元数编码旋转/镜像归一化，减少对称重复探索 |
| P2-2 | **Tsirelson bound 检查** | tr87 规则发现中使用 CHSH≤2√2 检查，>2√2 的伪规则判为 PR-Box 代数非法，Dead-Zero熔断 |
| P2-3 | **Asym Index 非结合残差** | tr87 使用八元数非结合性指标（Associator Index η）作为L3残差评估 |
| P2-4 | **陪集空间投影 κ-Snap (L3)** | C(11,4)陪集归约用于网格模式类游戏（cn04/sp80）的状态质量评估 |
| P2-5 | **游戏特定L1配置热插拔** | YAML/JSON 配置文件驱动L1-L4配置，无需修改代码即可调整游戏策略 |

---

## 5. 游戏 L1-L4 配置表（25游戏全覆盖）

基于代码现状分析，以下是每个游戏的四层配置方案：

### 5.1 文章明确配置的4个游戏

| 游戏 | 游戏类型 | L1 状态枚举 | L2 结构剪枝 | L3 残差评估 | L4 决策融合 | 当前RHAE | 目标RHAE |
|------|----------|-------------|-------------|-------------|-------------|----------|----------|
| **ka59** | 推箱子(Sokoban) | Wall-BFS (贴墙BFS+物理检查) | 组合对称剪枝 (八元数编码) | CHK_DL+Wall-Ride豁免 (Dead-Zero熔断) | κ-优选优先推箱至目标 | 0 | 700+ |
| **ls20** | 平台跳跃 | BFS路径规划 (位置网格寻路) | 时间轴剪枝 (碰撞图+changer序列归一化) | 陷阱规避残差 (GaussEx安全着陆评估) | κ-优选优先安全着陆 | 115(L0 only) | 805 |
| **tr87** | 双次翻译(cipher) | DFS枚举 (短序列宏动作：导航+变体调整) | Prime-Signature (数论指纹快速拒识伪规则) | 八元数非结合残差 (Asym Index η) | κ-优选选择残差最小翻译顺序 | ~115 | 575+ |
| **ft09** | 点击模式(color rotation) | DFS枚举点击 (约束满足→点击序列) | 视觉特征聚类剪枝 (等价目标合并) | 点击反馈残差 (点击后颜色匹配度GaussEx) | κ-优选模式匹配 (最少点击优先) | ~115 | 645+ |

### 5.2 剩余21个游戏推断配置

> **推断原则**: L1按游戏动作空间选择（BFS=移动类, DFS=枚举类, Wall-BFS=推箱子类, Δ-State Replay=所有BFS/DFS）；L2按游戏对称性选择；L3按游戏约束类型选择；L4统一κ-优选。

| 游戏 | 游戏类型 | L1 状态枚举 | L2 结构剪枝 | L3 残差评估 | L4 决策融合 | 当前状态 | 备注 |
|------|----------|-------------|-------------|-------------|-------------|----------|------|
| **dc22** | 移动到目标 | BFS (位置网格寻路, step=2) | walkable集合去重 | 到达残差 (距离η) | κ-优选最短路径 | 已通关 | 简单BFS, 适配Δ-Replay |
| **tu93** | 键盘操作 | DFS枚举 (短序列宏动作) | 对称剪枝 (动作序列归一化) | 执行残差 (GaussEx) | κ-优选最优序列 | 已通关 | DFS适配Δ-Replay |
| **bp35** | 键盘操作 | DFS枚举 | 动作序列去重 | 执行残差 | κ-优选 | 已通关 | 类tu93 |
| **wa30** | 键盘导航 | BFS (位置寻路) | walkable去重 | 到达残差 | κ-优选最短路径 | 已通关 | 类dc22 |
| **g50t** | 键盘+点击 | BFS+DFS混合 | 对称剪枝 | GaussEx残差 | κ-优选 | 已通关 | 独立g50t_solver.py |
| **lf52** | 键盘操作 | DFS枚举 | 动作序列去重 | 执行残差 | κ-优选 | 已通关 | 类tu93 |
| **sk48** | 排序类 | DFS枚举 (swap序列) | 拟阵约束 (Union-Find回路检测) | swap质量残差 | κ-优选最优排序 | 已通关 | 类sb26 |
| **m0r0** | 键盘导航 | BFS (位置寻路) | walkable去重 | 到达残差 | κ-优选 | 已通关 | 类dc22 |
| **cn04** | 仿射变换 | DFS枚举 (旋转+移动宏动作) | 组合对称剪枝 (旋转归一化: 4×旋转等价) | κ-Phase一致性残差 (kappa_phase_consistency) | κ-优选最佳变换 | 0(未通关) | 当前用affine_transform，需增强L3 |
| **r11l** | 点击选择 | DFS枚举点击 (颜色匹配) | 颜色聚类去重 | 点击反馈残差 | κ-优选最少点击 | 已通关 | 类ft09 |
| **lp85** | 键盘导航 | BFS | walkable去重 | 到达残差 | κ-优选 | 已通关 | 类dc22 |
| **s5i5** | 键盘+点击 | BFS+DFS混合 | 对称剪枝 | GaussEx残差 | κ-优选 | 已通关 | 类g50t |
| **tn36** | 点击编程(状态机) | DFS枚举 (opcode序列直接计算) | 无(无对称性) | opcode正确性残差 (直接计算无搜索) | 直接计算返回(κ-优选不适用) | 0(未通关) | deepcopy-free, 需补全opcode映射 |
| **su15** | 点击推进(ring expansion) | DFS枚举点击 | ring对称剪枝 | 推进效果残差 | κ-优选推进顺序 | 已通关 | 纯点击类 |
| **vc33** | 键盘操作 | DFS枚举 | 动作序列去重 | 执行残差 | κ-优选 | 已通关 | 类tu93 |
| **re86** | 移动+绘画 | BFS (位置寻路+绘画) | canvas区域去重 | 绘画覆盖残差 | κ-优选最优覆盖路径 | 已通关(部分) | 需Δ-Replay优化 |
| **ar25** | 镜像反射 | BFS (反射路径计算+移动) | 组合对称剪枝 (镜像归一化) | κ-Phase一致性残差 (mirror+coverage) | κ-优选覆盖最大化 | 0(未通关) | 当前有AR25Adapter但未通关 |
| **sc25** | 键盘+点击 | BFS+DFS混合 | 对称剪枝 | GaussEx残差 | κ-优选 | 已通关 | 类g50t |
| **sb26** | 颜色排序(Poset) | DFS枚举 (swap序列) | 拟阵约束 (Union-Find回路检测+poset验证) | swap质量残差 (is_valid_poset_order) | κ-优选最优排序 | 0(未通关) | 当前有poset primitives |
| **cd82** | 导航+印章 | BFS (位置寻路) | canvas区域去重 | 印章覆盖残差 | κ-优选覆盖顺序 | 已通关(L0) | 简单 |
| **sp80** | 选择+溢出(liquid) | BFS+DFS混合 | 区域对称剪枝 | 液体覆盖残差 | κ-优选覆盖最大化 | 已通关(部分) | 需增强 |

---

## 6. 当前代码现状与差距分析

### 6.1 现有基础设施（可复用）

| 模块 | 文件 | 已有能力 | 缺口 |
|------|------|----------|------|
| Δ-State Engine | `delta_state.py` | Node(parent_id+action), ReplayEngine, LayoutHasher, GaussExVerifier | **仅ls20使用**，需扩展到所有BFS/DFS/kappa搜索 |
| κ-Transformation ISA | `t_processor_isa.py` | KappaCausalReductionSolver, KSnapEngine(C(11,4)陪集), Tsirelson bound | **未被任何solver实际调用**，仅定义 |
| Physics Primitives | `physics_primitives.py` | can_push_box, is_deadlock_corner, mirror_point, find_affine_transform, κ-Phase consistency | **已被ka59/ar25/cn04/sb26使用** |
| Oracle Adapters | `oracle_adapters.py` | KA59Adapter, AR25Adapter, TN36Adapter | **3个游戏有适配器**，其余22个无 |
| Neural DSL | `neural_dsl.py` | KappaTsirelsonVerifier, KappaCausalReductionDSL, confidence_from_eta | **未被实际调用** |
| Octonion Modules | `octonion_tensor.py`, `octonion_layers.py`, `octonion_resnet.py` | 八元数乘法、内积、陪集投影 | **未被实际调用** |

### 6.2 关键差距

1. **deepcopy 瓶颈全局存在**: `solve_generic_bfs` (line 7175) 每节点 `copy.deepcopy(g)`；`solve_kappa_priority_search` (line 7007/7014) 每节点 deepcopy；`solve_game()` 入口 (line 12707/12796) 多次 deepcopy
2. **碎片化solver模式**: 25个独立solver函数，每个有自己的搜索策略、剪枝逻辑、验证方式——无统一管线
3. **已有模块未集成**: κ-Snap, Tsirelson bound, Octonion陪集投影等高级模块存在于代码中但未被任何solver的执行路径调用
4. **5个零分游戏根因**: ka59(push-mechanic未正确处理), tn36(opcode映射不完整), cn04(affine-transform验证失败), ar25(coverage BFS失败), sb26(poset排序失败)

### 6.3 性能对比

| 操作 | 当前(deepcopy) | 目标(Δ-Replay) | 提升倍数 |
|------|----------------|----------------|----------|
| 单节点扩展 | ~50ms (deepcopy) | ~1ms (perform_action) | **50×** |
| ls20 300步验证 | ~15s (300×deepcopy) | ~0.3s (300×perform_action) | **50×** |
| BFS 500000节点 | ~25000s (超预算) | ~500s (可探索) | **50×** |

---

## 7. UI/交互设计（架构层级）

> 注：本项目是纯后端solver架构，无用户界面。以下为内部架构设计。

```
┌──────────────────────────────────────────────────────────┐
│                  solve_game() [路由入口]                    │
│  Phase 0: HybridSearchPipeline(game, game_id, level_idx) │
│  Phase 1: Generic Fallback (仅对无L1-L4配置的游戏)         │
└──────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│              HybridSearchPipeline (四层管线)                │
│                                                            │
│  ┌─ L1 状态枚举层 ──────────────────────────────────┐     │
│  │  L1Config(game_id) → 选择枚举策略                 │     │
│  │  Wall-BFS | BFS | DFS | Δ-State Replay           │     │
│  │  → 候选动作列表 + 状态节点(Node)                 │     │
│  └───────────────────────────────────────────────────┘     │
│                           │                                │
│                           ▼                                │
│  ┌─ L2 结构剪枝层 ──────────────────────────────────┐     │
│  │  L2Config(game_id) → 选择剪枝策略                 │     │
│  │  组合对称 | Prime-Signature | 拟阵约束 | 无剪枝    │     │
│  │  → 剪枝后候选列表                                │     │
│  └───────────────────────────────────────────────────┘     │
│                           │                                │
│                           ▼                                │
│  ┌─ L3 残差评估层 ──────────────────────────────────┐     │
│  │  L3Config(game_id) → 选择评估策略                 │     │
│  │  κ-Snap | Dead-Zero Fuse | GaussEx | Asym Index  │     │
│  │  → 每候选残差 η + confidence = 1 - η/δ_K         │     │
│  └───────────────────────────────────────────────────┘     │
│                           │                                │
│                           ▼                                │
│  ┌─ L4 决策融合层 ──────────────────────────────────┐     │
│  │  κ-优选(κ-Selection): η升序选择最小残差候选       │     │
│  │  → 最终动作计划                                  │     │
│  └───────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
```

---

## 8. 待确认问题

| # | 问题 | 影响范围 | 建议默认 | 需确认人 |
|---|------|----------|----------|----------|
| Q1 | **游戏特定solver保留策略**: 25个现有单体solver函数是否全部删除替换为HybridSearchPipeline，还是保留作为Phase 0 fallback？ | 全架构 | 保留作为fallback（Phase 0优先专用solver，HybridPipeline作为Phase 1统一化） | 用户+架构师 |
| Q2 | **Δ-State Replay 与 deepcopy-unsafe游戏的兼容**: tn36等游戏deepcopy会破坏lambda闭包，Δ-State Replay是否也受影响？ReplayEngine用perform_action而非deepcopy，但tn36的plan是直接计算无需搜索 | tn36/类似游戏 | tn36绕过搜索管线（直接计算），Δ-Replay仅用于需要搜索的游戏 | 架构师 |
| Q3 | **κ-优选的η阈值 δ_K**: 文章提到δ_K≈0.036（来自t_processor_isa.py的KSnapEngine），但当前GEX_PASS_THRESHOLD=1/6≈0.167。两个阈值如何统一？ | L3+L4全局 | 统一使用 δ_K=0.036 作为κ-优选判定阈值，1/6仅用于RHAE预算控制 | 用户+架构师 |
| Q4 | **Wall-BFS 的具体实现**: 文章描述Wall-BFS为"贴墙宽度优先搜索"，当前ka59使用Newton-push BFS(Sokoban状态=(player_pos, frozenset(box_pos)))。Wall-BFS是否是在当前Sokoban BFS基础上增加贴墙优先级（push box toward wall-side goal），还是全新的搜索模式？ | ka59 | 在当前Sokoban BFS基础上增加贴墙优先队列排序（push box toward nearest wall-side goal） | 用户 |
| Q5 | **25个游戏是否全部需要L2剪枝**: 某些简单游戏（dc22/wa30/lp85）只需简单BFS寻路，L2剪枝可能增加开销而非减少 | 性能 | L2默认为"pass-through"(无剪枝)，仅对组合爆炸类游戏启用 | 架构师 |
| Q6 | **Wall-Ride豁免的具体物理含义**: 文章提到KA59的Wall-Ride豁免——当前代码中push-through-wall机制(line 1651-1686)是否就是Wall-Ride的实现？ | ka59 L3 | push-through-wall = Wall-Ride豁免，保持当前实现 | 用户 |
| Q7 | **文章报告的RHAE数字可信度**: 文章声称Ka59 0→700+, Ls20 115→805等。这些数字是否基于实际运行验证？是否有可复现的测试脚本？ | 目标设定 | 假设数字可信，但以实际测试结果为准。如达不到，以"5个零分游戏攻克"为P0底线 | 用户 |
| Q8 | **deepcopy消除范围**: `_verify_plan()` (line 12741) 和 `solve_game()`入口(line 12707) 的 deepcopy 是否也需要消除？验证阶段是否可以用 Δ-Replay？ | 全架构 | 验证阶段保留1次deepcopy（最终验证），搜索节点扩展全部改用Δ-Replay | 架构师 |

---

## 9. 风险评估（遵循哲学工程手册约束）

| 风险 | 等级 | 说明 | 缓解 |
|------|------|------|------|
| **架构迁移破坏现有通关** | 高 | 25个游戏中20+已通关，统一重构可能引入回归 | 增量迁移：先在5个零分游戏验证，再逐游戏迁移；保留原solver作为fallback |
| **deepcopy-unsafe游戏兼容** | 中 | tn36等lambda闭包游戏无法deepcopy也无法Δ-Replay | tn36走"直接计算"路径，不经过搜索管线 |
| **κ-Snap/陪集投影性能** | 中 | C(11,4)=330陪集投影可能增加计算开销 | 仅在L3实际需要时启用，简单游戏跳过L3 |
| **文章RHAE数字不可复现** | 低 | 文章实验数字可能基于不同环境/配置 | 以实际测试为准，P0底线是5个零分游戏攻克 |
| **45秒时间预算不足** | 低 | 统一管线可能增加初始化开销 | Δ-Replay节省的时间远大于管线初始化开销 |

---

## 10. 实施优先级与里程碑

| 里程碑 | 内容 | 预期时间 |
|--------|------|----------|
| **M1** | HybridSearchPipeline 框架 + Δ-Replay全局化 + 5个零分游戏配置 | 1-2天 |
| **M2** | ka59 Wall-BFS + tn36直接计算完善 + cn04/sb26/ar25 L1-L4配置 | 2-3天 |
| **M3** | ls20 L1-L6通关 + tr87/ft09 Prime-Signature + κ-优选统一 | 1-2天 |
| **M4** | 剩余21游戏迁移到统一管线 + 回归测试 + 性能优化 | 2-3天 |
| **M5** | P2功能(八元数归一化/Tsirelson/Asym Index) + 175关卡满分冲刺 | 2-3天 |

---

## 附录A: 关键文件清单

| 文件 | 行数 | 角色 | 重构方向 |
|------|------|------|----------|
| `game_solvers.py` | 13420 | 25个solver + 通用solver + solve_game()路由 | 重构为 HybridSearchPipeline 调用 |
| `delta_state.py` | ~500 | Node, ReplayEngine, GaussExVerifier | 扩展为全局搜索基础设施 |
| `t_processor_isa.py` | ~300 | κ-ISA, KappaCausalReductionSolver, KSnapEngine | 集成到 L3/L4 管线 |
| `neural_dsl.py` | ~2000 | κ-Tsirelson, YinLong, Tianxing | 集成到 L3 评估 |
| `oracle_adapters.py` | ~200 | KA59/AR25/TN36 适配器 | 扩展到25个游戏 |
| `physics_primitives.py` | ~300 | 推箱子/镜像/仿射物理原语 | 保持，作为L1物理检查 |
| `octonion_tensor.py` + `octonion_layers.py` + `octonion_resnet.py` | ~800 | 八元数运算 | 集成到 L2 对称剪枝 + L3 Asym Index |

## 附录B: SOLVERS字典当前映射（25游戏全覆盖）

```python
SOLVERS = {
    "ls20": solve_ls20,    "tr87": solve_tr87,    "ft09": solve_ft09,
    "tu93": solve_tu93,    "wa30": solve_wa30,    "dc22": solve_dc22,
    "m0r0": solve_m0r0,    "re86": solve_re86,    "lp85": solve_lp85,
    "cd82": solve_cd82,    "g50t": solve_g50t,    "sb26": solve_sb26,
    "r11l": solve_r11l,    "cn04": solve_cn04,    "sp80": solve_sp80,
    "ar25": solve_ar25,    "ka59": solve_ka59,    "sc25": solve_sc25,
    "sk48": solve_sk48,    "su15": solve_su15,    "vc33": solve_vc33,
    "s5i5": solve_s5i5,    "tn36": solve_tn36,    "bp35": solve_bp35,
    "lf52": solve_lf52,
}
```
