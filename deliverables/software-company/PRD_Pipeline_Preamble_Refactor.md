# PRD: Pipeline + SB Preamble Injector 架构重构

## 项目信息

| 字段 | 值 |
|------|-----|
| Language | 中文 |
| Programming Language | Python 3.10+ |
| Project Name | tomas-arc3-solver |
| 基准 | ARC-AGI-3 竞赛求解器，基于太乙互搏(TOMAS)框架 |
| 当前状态 | 21/25 游戏 L0 RHAE 满分，4 个零分（tn36, ka59, ar25, sb26） |

## 原始需求复述

将当前"每 SB 写独立 solver 函数 + SOLVERS dict 分发"的架构，重构为"SB 注册 preamble 约束注入 + 统一 pipeline 求解"架构。核心动因：当前架构违反 IDO 反单调性（IC(代码base)↑，IC(原理)↓），无法泛化到未见 SB 游戏。

---

## 1. 产品目标

1. **消除反单调性违反**：将 25 个独立 solver 函数收敛为统一 pipeline + preamble 注入模式，使 IC(代码base)↓、IC(原理)↑，满足 IDO 反单调性公理。
2. **提升零分游戏覆盖率**：通过 preamble 注入的陪集偏好、对称先验、数列检测等能力，将当前 4 个零分游戏（tn36, ka59, ar25, sb26）至少提升至非零分。
3. **支持未见 SB 泛化**：新架构下，遇到未注册 preamble 的游戏时，pipeline 以默认参数自动运行，无需手写新 solver。

---

## 2. 用户故事

| # | 用户故事 |
|---|---------|
| US1 | As a **竞赛开发者**，I want to **为新 SB 游戏只需写 10 行 preamble 注入而非完整 solver 函数**，so that 我可以快速适配新游戏而不违反 IDO 反单调性。 |
| US2 | As a **TOMAS 框架维护者**，I want to **统一 pipeline 按序驱动 EML→Coset→RM→κ-Snap→GaussEx**，so that 每个求解阶段遵循一致的 IDO 公理约束而非各 SB 各自实现。 |
| US3 | As a **调试/刷榜工程师**，I want to **通过 SOLVERS dict 的 shortcut/fallback 机制保留旧 solver 作为兜底**，so that 重构期间不丢失已验证的求解能力。 |
| US4 | As a **ARC-AGI 竞赛参赛者**，I want to **pipeline 在 preamble 缺失时仍以默认参数运行**，so that 未见 SB 游戏也能自动求解而非直接零分。 |

---

## 3. 需求池

### P0 — Must Have（必须完成）

| ID | 需求 | 描述 |
|----|------|------|
| P0-1 | SBInjector dataclass | 创建 `src/agent/injectors.py`，定义 SBInjector 数据类，包含 name, coset_filter, symmetry_hint, enable_rm, time_window, eps_factor, sporadic_pref 六个字段 |
| P0-2 | 25 游戏 preamble 注册表 | 在 `injectors.py` 中为 25 个游戏各写 preamble 函数（10 行以内），返回 SBInjector 实例；建立 `PREAMBLES: Dict[str, Callable]` 注册表 |
| P0-3 | solve_via_pipeline 入口 | 修改 `universal_solver_pipeline.py`，新增 `solve_via_pipeline(input, train_pairs, injector: SBInjector)` 函数，按序驱动 EML→Coset→RM→κ-Snap→GaussEx |
| P0-4 | SOLVERS dict 改造 | 修改 `game_solvers.py`，将 SOLVERS dict 改造为三层架构：dispatcher（pipeline+injector）/ shortcut（旧 solver 快捷路径）/ fallback（旧 solver 兜底） |
| P0-5 | GaussEx + κ-entropy 校验模块 | 创建 `src/agent/verify.py`，提取并整合 ido_axioms.py 中的 κ-entropy 校验和 neural_dsl.py 中的 GaussEx 宽松度验证 |

### P1 — Should Have（推荐完成）

| ID | 需求 | 描述 |
|----|------|------|
| P1-1 | Coset 剪枝 | pipeline 中 Coset_Search 阶段读取 injector.coset_filter，优先搜索指定陪集，跳过无关联陪集 |
| P1-2 | 对称先验注入 | κ-Snap 阶段读取 injector.symmetry_hint，优先 D4/Monster 对称群而非全量枚举 |
| P1-3 | RM 开关 | Ramanujan_Machine 阶段依据 injector.enable_rm 决定是否启用 PSLQ+连分数数列检测 |
| P1-4 | time_window 多帧 | EML_Perceive 阶段依据 injector.time_window 保留多帧因果边，适配交互任务 |
| P1-5 | eps_factor 宽松度调节 | GaussEx 校验阶段依据 injector.eps_factor 调节阈值乘数 |
| P1-6 | sporadic_pref 过滤 | Coset_Search 阶段依据 injector.sporadic_pref 优先/排除散在群 |

### P2 — Nice to Have（锦上添花）

| ID | 需求 | 描述 |
|----|------|------|
| P2-1 | GameProfile 注入字段 | 在 `game_profiles.py` 的 GameProfile 中添加 injector_name 字段，与 PREAMBLES 注册表联动 |
| P2-2 | preamble 热加载 | 支持运行时动态注册新 preamble，无需重启 |
| P2-3 | pipeline 阶段耗时追踪 | 各阶段记录耗时，输出 IDO 反单调性验证日志 |

---

## 4. 25 游戏 ID → Preamble 参数映射表

> 注：以下映射基于游戏类型、已知求解难度和现有 game_profiles.py 配置推断。具体 preamble 值需在实现阶段根据调试结果微调。

### 键盘游戏 (keyboard, 6 个)

| 游戏 ID | game_type | coset_filter | symmetry_hint | enable_rm | time_window | eps_factor | sporadic_pref | 备注 |
|---------|-----------|-------------|---------------|-----------|-------------|------------|---------------|------|
| ls20 | keyboard | None | "D4" | True | 1 | 1.0 | None | 键盘映射+D4旋转常见 |
| tu93 | keyboard | None | "D4" | True | 1 | 1.0 | None | 类似ls20 |
| tr87 | keyboard | [0,1] | None | True | 1 | 1.0 | None | 可能涉及陪集变换 |
| re86 | keyboard | None | "D4" | True | 1 | 1.0 | None | 键盘+对称 |
| g50t | keyboard | None | None | True | 1 | 1.0 | None | 默认键盘处理 |
| wa30 | keyboard | None | None | True | 1 | 1.5 | None | 可能需宽松校验 |

### 点击游戏 (click, 7 个)

| 游戏 ID | game_type | coset_filter | symmetry_hint | enable_rm | time_window | eps_factor | sporadic_pref | 备注 |
|---------|-----------|-------------|---------------|-----------|-------------|------------|---------------|------|
| ft09 | click | None | "D4" | True | 1 | 1.0 | None | 点击+D4对称 |
| vc33 | click | None | "D4" | True | 1 | 1.0 | None | 类似ft09 |
| s5i5 | click | [2,3] | "D4" | True | 1 | 1.0 | None | 可能涉及陪集 |
| su15 | click | None | None | True | 1 | 1.0 | None | 默认点击处理 |
| lp85 | click | None | None | True | 1 | 1.0 | None | 默认点击处理 |
| r11l | click | None | None | True | 1 | 1.0 | None | 默认点击处理 |
| tn36 | click | [0,1,2] | "Monster" | True | 1 | 1.5 | "M11" | ⚠️ 当前零分，需特殊preamble：Monster对称+散在群M11+宽松校验 |

### 键盘+点击游戏 (keyboard+click, 12 个)

| 游戏 ID | game_type | coset_filter | symmetry_hint | enable_rm | time_window | eps_factor | sporadic_pref | 备注 |
|---------|-----------|-------------|---------------|-----------|-------------|------------|---------------|------|
| ar25 | kb+click | [0,2,4] | "Monster" | True | 2 | 2.0 | "M24" | ⚠️ 当前零分，需特殊preamble：多陪集+Monster+多帧+宽松+M24 |
| cn04 | kb+click | None | "D4" | True | 1 | 1.0 | None | 混合型默认 |
| dc22 | kb+click | None | "D4" | True | 1 | 1.0 | None | 混合型默认 |
| ka59 | kb+click | [1,3] | "Monster" | True | 1 | 2.0 | "M12" | ⚠️ 当前零分，需特殊preamble：陪集+Monster+宽松+M12 |
| m0r0 | kb+click | None | "D4" | True | 1 | 1.0 | None | 混合型默认 |
| sp80 | kb+click | None | None | True | 1 | 1.0 | None | 混合型默认 |
| sb26 | kb+click | [0,3] | "D4" | True | 2 | 1.5 | None | ⚠️ 当前零分，需特殊preamble：陪集+多帧+宽松 |
| sc25 | kb+click | None | "D4" | True | 1 | 1.0 | None | 混合型默认 |
| sk48 | kb+click | None | "D4" | True | 1 | 1.0 | None | 混合型默认 |
| bp35 | kb+click | None | None | True | 1 | 1.0 | None | 混合型默认 |
| cd82 | kb+click | None | None | True | 1 | 1.0 | None | 混合型默认 |
| lf52 | kb+click | None | "D4" | True | 1 | 1.0 | None | 混合型默认 |

---

## 5. 待确认问题

| # | 问题 | 影响范围 |
|---|------|---------|
| Q1 | 4 个零分游戏（tn36, ka59, ar25, sb26）的 preamble 参数是否需要更激进的设置（如 eps_factor>2.0、time_window>2）？ | P0-2, P1-4, P1-5 |
| Q2 | SOLVERS dict 的 shortcut/fallback 层是否保留所有 25 个旧 solver 函数，还是仅保留当前零分游戏的旧函数作为兜底？ | P0-4 |
| Q3 | `verify.py` 中 GaussEx 校验的提取边界——是仅提取阈值逻辑，还是同时提取 κ-entropy 的完整计算流程？ | P0-5 |
| Q4 | preamble 函数是否需要支持运行时参数覆盖（如调试时临时修改 eps_factor）？ | P0-2, P2-2 |
| Q5 | 统一 pipeline 的阶段失败回退策略——某阶段（如 RM）超时或无结果时，是否跳过继续下一阶段，还是直接 fallback 到旧 solver？ | P0-3, P0-4 |

---

## 6. 架构概览（信息性，不作为需求）

```
                    ┌─────────────────┐
                    │  game_solvers.py │
                    │  SOLVERS dict    │
                    │  (3层dispatch)   │
                    └─────────┬───────┘
                              │
              ┌───────────────┼───────────────┐
              │ dispatcher    │ shortcut       │ fallback
              │ (pipeline)    │ (旧solver快径) │ (旧solver兜底)
              └───────────────┘               │
                      │                       │
              ┌───────▼───────┐       ┌───────▼───────┐
              │ injectors.py  │       │ solve_sbXX()  │
              │ PREAMBLES注册 │       │ (原函数保留)  │
              └───────┬───────┘       └───────────────┘
                      │
              ┌───────▼───────────────────────────────┐
              │ solve_via_pipeline(input, pairs, inj) │
              │                                        │
              │  EML_Perceive(inj.time_window)         │
              │    → Coset_Search(inj.coset_pref)      │
              │    → Ramanujan_Machine(inj.enable_rm)  │
              │    → κ-Snap(inj.symmetry_hint)         │
              │    → GaussEx(inj.eps_factor)           │
              └────────────────────────────────────────┘
```

---

## 7. 新建/修改模块清单

| 模块 | 操作 | 说明 |
|------|------|------|
| `src/agent/injectors.py` | **新建** | SBInjector dataclass + 25 个 preamble 函数 + PREAMBLES 注册表 |
| `src/agent/verify.py` | **新建** | GaussEx 校验 + κ-entropy 验证（提取自 ido_axioms + neural_dsl） |
| `src/agent/universal_solver_pipeline.py` | **修改** | 添加 SBInjector 参数，驱动各阶段策略选择 |
| `src/agent/game_solvers.py` | **修改** | SOLVERS dict 三层改造 |
| `src/agent/game_profiles.py` | **可选修改** | GameProfile 添加 injector_name 字段 |
