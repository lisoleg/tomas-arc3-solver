# ARC-AGI-3 Solver 双轨模式状态报告

## 版本: v3.4.2-dev | 日期: 2026-06-26

## 双轨架构概览

| 轨道 | 入口 | 目标RHAE | 适用范围 |
|------|------|---------|---------|
| Track 1 (硬编码Adapter) | SOLVERS dict, Phase 0 | ≥ 80 | Public Set 25已知游戏 |
| Track 2 (通用Pipeline) | UniversalSolverPipeline, Phase 1 | ≥ 60 | Private Set未知游戏 |

---

## Track 1: 硬编码Adapter (25游戏SOLVERS dict)

### L0 通过情况 (实测数据)

| 游戏 | L0 RHAE | L0 步数 | Baseline | 耗时 | Phase | 状态 |
|------|---------|---------|----------|------|-------|------|
| ls20 | 115.0 | 13 | 22 | 25.2s | Phase 0→1 (Oracle+Pipeline) | ✅ |
| vc33 | 115.0 | 3 | 7 | 0.2s | Phase 0 (Adapter) | ✅ |
| tr87 | 115.0 | 14 | 54 | 0.2s | Phase 0 (Cipher) | ✅ |
| tu93 | 111.4 | 18 | 19 | 0.0s | Phase 0 (Tu93) | ✅ |
| ft09 | 115.0 | 4 | 43 | 0.0s | Phase 0 (Constraint) | ✅ |
| s5i5 | 115.0 | 13 | 20 | 0.0s | Phase 0 (Selector) | ✅ |
| lp85 | 115.0 | 5 | 17 | 0.1s | Phase 0 (Lp85) | ✅ |
| re86 | 0 | 999 | 26 | 68.0s | Phase 1-7 (全部失败) | ❌ |
| g50t | 0 | 999 | 78 | 68.0s | Phase 1-7 (全部失败) | ❌ |
| wa30 | 0 | 999 | 71 | 68.3s | Phase 1-7 (全部失败) | ❌ |
| tn36 | 0 | 999 | 32 | 60.1s | Phase 1-7 (全部失败) | ❌ |
| su15 | 0 | 999 | 22 | 45.5s | Phase 1-7 (全部失败) | ❌ |
| r11l | 0 | 999 | 22 | 4.7s | Phase 1-7 (全部失败) | ❌ |
| bp35 | 0 | 999 | 21 | 64.4s | Phase 1-7 (全部失败) | ❌ |
| dc22 | 0 | 999 | 59 | 68.3s | Phase 1-7 (全部失败) | ❌ |
| sk48 | 0 | 999 | 61 | 68.1s | Phase 1-7 (全部失败) | ❌ |
| lf52 | 0 | 999 | 32 | 62.5s | Phase 1-7 (全部失败) | ❌ |
| sc25 | 0 | 999 | 36 | 61.2s | Phase 1-7 | ❌ |
| m0r0 | 0 | 999 | 30 | 68.2s | Phase 1-7 | ❌ |
| cn04 | 115.0 | 13 | 29 | 34.3s | Phase 0 (Cn04) | ✅ |
| cd82 | 115.0 | 5 | 55 | 2.2s | Phase 0 (Cd82) | ✅ |
| sp80 | 115.0 | 4 | 39 | 13.4s | Phase 0 (Sp80) | ✅ |
| ka59 | ? | ? | 28 | ? | 运行中(60s+) | ⏳→❌ |
| ar25 | ? | ? | 32 | ? | 运行中(60s+) | ⏳→❌ |
| sb26 | ? | ? | 18 | ? | 运行中(60s+) | ⏳→❌ |

**L0 已确认**: 10/22 ✅ (45.5% 通过率), ka59/ar25/sb26预计❌
**L0 已确认 RHAE**: avg=115.0 (9个满分), tu93=111.4

### L1 通过情况 (前一轮诊断数据)

| 游戏 | L1 RHAE | 根因分析 |
|------|---------|---------|
| vc33 | 115.0 ✅ | Phase 0 一步到位 |
| tu93 | 115.0 ✅ | Phase 0 正常 |
| ls20 | 0 ❌ | 所有Phase失败, L0靠Pipeline22s解决, L1需要更复杂BFS |
| s5i5 | 0 ❌ | 14步计划但verify不通过 — selector尺寸限制(最多8次grow) |
| lp85 | 0 ❌ | 30步计划但verify不通过 — 按钮坐标错误(y偏移9倍) |

---

## Track 2: 通用Pipeline (UniversalSolverPipeline)

### 修复状态
- ✅ NameError修复: `_solve_map_pixel_bfs` 和 `_simulation_bfs_click_replay` 中缺失的 `ActionInput` import 已补上
- Pipeline在s5i5/lp85 L1测试时仍无法独立解决（需要游戏知识）

### Pipeline Strategy 覆盖
| Strategy | 适用场景 | 效果 |
|----------|---------|------|
| Strategy 0 | Click games with map/pixel sprites | BFS点击搜索 |
| Strategy 1 | Keyboard games with pathfinding | 路径规划+移动 |
| Strategy 2 | Mixed keyboard+click | 键盘优先+点击辅助 |
| Strategy 3 | Puzzle games with state hashing | 状态空间搜索 |
| Strategy 4 | Fallback random+heuristic | 混合随机 |

---

## 关键问题与优先级

### 🔥 P0 — 最高优先级 (影响最多游戏)
1. **18个游戏L0=0**: re86/g50t/wa30/tn36/su15/r11l/bp35/dc22/sk48/lf52/sc25/m0r0/cn04/cd82/sp80/ka59/ar25/sb26 的Adapter全部失败，pipeline也60s超时
2. **根因**: 这些游戏的Phase 0 solver可能是空壳(placeholder)或逻辑错误，Phase 1-7的BFS/Beam/IDFS/DFS/Keyboard/Random都无法在60s内找到解

### 🔧 P1 — 中等优先级 (已有L0但L1失败)
1. **s5i5 L1**: selector最多8次grow但solver分配12次 → 需仿真式验证
2. **lp85 L1**: 按钮坐标y偏移错误 → 需用 `_get_valid_action_inputs()` 正确坐标
3. **ls20 L1**: 所有Phase 60s超时 → Oracle replay完全无效

### 📋 P2 — 低优先级
1. **ft09 L4/L5**: NTi菱形交互未实现
2. **tr87 L5**: alter_rules + tree_trans 未实现
3. **tu93 L0**: RHAE=111.4 (差3.6分) → 步数18 > baseline 19

---

## RHAE 总分估算

### 已测17游戏L0贡献
- 7 solved × avg RHAE ≈ 7 × 114.6 ≈ 801.0
- 10 failed × 0 = 0
- **L0 subtotal ≈ 801.0**

### 全量估算(25游戏×7-9关卡)
- 已解决关卡: 7个L0 + vc33/tu93 L1 ≈ 9关卡
- 总关卡数: 25游戏 × avg 7关卡 ≈ 175关卡
- **通过率 ≈ 5.1%** (9/175)
- **总分 RHAE ≈ 801.0 / 175 ≈ 4.6** (有效平均)

### 目标差距
- 目标: Adapter RHAE ≥ 80 → 需要 ~140关卡中 avg RHAE=80
- 当前: 只有9关卡 × avg 114.6
- **差距巨大**: 需从 18个0分游戏至少解决L0

---

## 下一步建议

1. **诊断18个0分游戏的根因**: 为什么Adapter和Pipeline都失败？是solver逻辑错误还是游戏机制复杂？
2. **优先修最简单的5个**: r11l(22步baseline, 4.7s超时) / su15(22步) / tn36(32步) / cn04(29步) / ar25(32步)
3. **修s5i5/lp85 L1**: 快速修复已有L0游戏的L1通过率
4. **优化Pipeline Strategy**: 让更多游戏在Phase 1解决而非60s超时

---

*生成时间: 2026-06-26 07:12 | 数据来源: benchmark实测 + 前轮诊断*
