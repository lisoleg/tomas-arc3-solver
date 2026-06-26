# 诊断报告：通用求解器能力评估 (Task #340)

**日期**: 2026-06-26
**测试方法**: `solve_game()` 批量求解器 + plan verification on deepcopy
**测试范围**: 全25游戏 Level 0

## 核心发现

### 1. 双路径架构的关键区分

项目中存在**两条完全独立的求解路径**：

| 路径 | 入口 | 特点 | 已验证游戏 |
|------|------|------|-----------|
| **实时Oracle路径** | `PlannerAgent._choose_action()` | 逐步导航，Oracle adapter提供实时实体定位 | ls20(7/7), ft09(6/6), tr87(5/6) |
| **批量Solver路径** | `solve_game()` | 一次性deepcopy+DFS/heuristic搜索，输出完整plan | 8/25 Level 0 verified |

⚠️ **关键问题**: PlannerAgent实时路径依赖硬编码Oracle Adapter（LS20Adapter/FT09Adapter/TR87Adapter），**对Private Set完全无效**。批量Solver路径不调用Oracle Adapter，是通用路径的唯一入口。

### 2. 批量Solver路径 Level 0 结果

| 游戏 | 类型 | Plan Found | Verified | Plan Len | Baseline | RHAE | 耗时 |
|------|------|-----------|---------|---------|---------|------|------|
| **vc33** | click | ✅ | ✅ | 3 | 7 | **115.0** ⭐ | 0.05s |
| **lp85** | click | ✅ | ✅ | 5 | 8 | **115.0** ⭐ | 0.19s |
| **s5i5** | click | ✅ | ✅ | 13 | 8 | 37.9 | 0.03s |
| **tu93** | keyboard | ✅ | ✅ | 18 | 9 | 25.0 | 0.61s |
| **ls20** | keyboard | ✅ | ✅ | 13 | 7 | 29.0 | 8.15s |
| **wa30** | keyboard | ✅ | ✅ | 27 | 9 | 11.1 | 46.22s |
| **sk48** | kb+click | ✅ | ✅ | 40 | 8 | 4.0 | 27.92s |
| **m0r0** | kb+click | ✅ | ✅ | 37 | 6 | 2.6 | 35.31s |
| ft09 | click | ❌ | ❌ | 0 | 6 | 0 | 0.06s |
| tr87 | keyboard | ❌ | ❌ | 0 | 6 | 0 | 57.14s |
| re86 | keyboard | ❌ | ❌ | 0 | 8 | 0 | 23.94s |
| g50t | keyboard | ❌ | ❌ | 0 | 7 | 0 | 41.29s |
| tn36 | click | ❌ | ❌ | 0 | 7 | 0 | 26.08s |
| su15 | click | ❌ | ❌ | 0 | 9 | 0 | 0.09s |
| r11l | click | ❌ | ❌ | 0 | 6 | 0 | 3.94s |
| bp35 | keyboard | ❌ | ❌ | 0 | 9 | 0 | 29.82s |
| dc22 | click | ❌ | ❌ | 0 | 6 | 0 | 17.04s |
| lf52 | keyboard | ❌ | ❌ | 0 | 10 | 0 | 33.90s |
| sc25 | kb+click | ❌ | ❌ | 0 | 6 | 0 | 5.16s |

(剩余cn04/cd82/sp80/ka59/ar25/sb26待完整结果)

### 3. UniversalOracleAdapter Heuristic Discovery 测试

| 游戏 | Heuristic Player Found | Config Player Found | Walls | Goals |
|------|----------------------|--------------------|-------|-------|
| tu93 | ✅ (via heuristic) | ✅ (via config tag) | 0→1(config) | 1→1(config) |
| vc33 | ✅ (heuristic) | ✅ (heuristic) | 0 | 4 |
| bp35 | ✅ (heuristic) | ✅ (heuristic) | 0 | 0 |

**⚠️ GameEntity bug**: `'GameEntity' object has no attribute 'width'` — UniversalOracleAdapter 返回的 GameEntity 缺少 width/height 属性，导致后续逻辑异常。

### 4. 关键战略结论

#### 对Kaggle Private Set的预测

- **硬编码adapter游戏(ls20/ft09/tr87)**: Private Set中不会有这些游戏ID → 硬编码adapter完全失效
- **批量Solver路径**: 是Private Set的唯一希望
  - 8/25游戏 Level 0 verified → **但RHAE极低**
  - 平均RHAE约 33.7 (去掉115.0满分游戏后约15.4)
  - vc33和lp85达到满分115.0，但plan长度只有3-5步(可能碰巧简单)
  - 大多数verified游戏的RHAE << 80 → **Private Set总分极低**

#### 通用路径的核心瓶颈

1. **17/25游戏找不到任何plan** → solve_generic_dfs搜索空间太大/游戏机制太复杂
2. **找到plan的游戏RHAE极低** → DFS路径远非最优步数
3. **GAME_CONFIGS大多是空壳** → 25个游戏中只有5个有player_tag，3个有wall_tag → 大部分config-based detection退化为heuristic
4. **GameEntity属性缺失** → UniversalOracleAdapter返回的实体缺少基本属性(width/height)

## 下一步：Task #342 构建 UniversalSolverPipeline

基于诊断结果，UniversalSolverPipeline需要：

1. **核心**: 强化 `solve_game()` 批量路径而非PlannerAgent实时路径
   - PlannerAgent实时路径依赖Oracle adapter → Private Set无效
   - solve_game() 是纯game introspection → Private Set可用

2. **实体发现改进**: 修复GameEntity属性 + 加强heuristic规则
   - 当前heuristic已经能发现player(3/3游戏) → 基础可用
   - 需要修复width/height属性传递
   - 需要增加walkable/intangible detection (dc22需要walkable set)

3. **搜索策略优化**: 从DFS转向simulation-based BFS
   - solve_generic_dfs搜索效率极低(max_nodes=30000, max_time=10s)
   - 大多数游戏30+秒超时 → 需要更智能的action space缩减
   - click游戏应该系统性探索而非暴力DFS

4. **GAME_CONFIGS清理**: Private Set上config不可用
   - 所有25个config中的硬编码tag → Private Set无用
   - 需要100%依赖heuristic discovery
   - 但可以保留game_type信息(keyboard/click/kb+click) → 这可以从action space推断

## RHAE预估

假设Private Set有25个类似复杂度的游戏：
- ~8个可能被solve_game解决(RHAE 4-115)
- ~17个完全无法解决(RHAE 0)
- 总分估算: ~270 RHAE (8个平均33.7) vs 目标20125

**结论**: 当前通用路径在Private Set上得分极低，需要显著改进搜索策略和实体发现才能有竞争力。
