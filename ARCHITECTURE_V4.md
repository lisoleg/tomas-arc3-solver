# DopamineExplorer V4 — 深度架构重构

## 综合理论来源

### 网络调研（10+篇文章）
1. **图基探索论文** (arXiv:2512.24156): 帧分割→状态哈希→frontier导航
2. **ARC-AGI-3官方**: RHAE=min(1,human/agent)², 25游戏, Milestone 1截止2026-06-30
3. **竞赛排名**: #1 StochasticGoose 12.58%(CNN+RL), #3 dolphin 30/52关(图基探索)

### 理论文章（10篇微信公众号）
4. **CCG**: 复数i^k表示旋转状态，L1复结构作为归纳偏置
5. **GPCT**: 边界层降解——识别高耦合变量BL_c(2^k枚举)，外区退化为2-SAT/UP
6. **De Sitter排斥**: 最小信息归约尺度——防止推理链无限归约导致逻辑崩溃
7. **EML超图**: 认知记忆晶格——Append-only拓扑增广，MUS双存(矛盾不覆盖)
8. **κ-Snap**: 不可逆显影算子——从潜在态投影到显影态，Merkle审计链
9. **NASGA代数**: 非阿贝尔半群吸收律——路径依赖+逻辑闭环
10. **查表即算力**: 记忆-算力等价——预计算查表优于实时推导
11. **流贯调度**: 上下文切换=最优路由，Cache继承+跨进程启发
12. **过拟合免疫**: EML超图拓扑增广 vs LLM参数优化，无灾难性遗忘

### GitHub仓库（3个）
13. **tomas-agi**: κ-Snap调度器+八元数Fano平面+Dead-Zone熔断+MUS双存+φ-Gate语义门控
14. **jingguang-agi**: 认知递归动力学(CRD)+闭包捕获+Continuation回溯
15. **mnq-golden-spirit-ball-simulator**: 金灵球本征态+存算一体

### 用户额外要求
16. **RL/IRL**: 多巴胺激励+惩罚，加快收敛
17. **游戏业算法**: Dijkstra流场、行为树、MCTS
18. **计算机图形学**: 空间推理、占据栅格
19. **AlphaGo**: 蒙特卡洛树搜索+UCB+alpha-beta剪枝

## V4 架构：DopamineExplorer

### 核心思想
**多巴胺驱动的MCTS探索 + Dijkstra流场导航 + 创伤记忆免疫 + EML超图记忆 + GPCT边界层规划**

```
DopamineExplorer (V4)
├── 感知层 (Perception)
│   ├── FrameProcessor — 帧分割(单色连通组件)+状态栏遮蔽+状态哈希
│   ├── OccupancyGrid — 空间占据栅格(碰撞反馈学习墙壁) ← CG/机器人
│   └── CCGTracker — 复数i^k旋转状态跟踪 ← CCG理论
│
├── 记忆层 (Memory) ← EML超图+κ-Snap
│   ├── EMLHyperGraph — Append-only拓扑增广(无覆写)+MUS双存(矛盾共存)
│   ├── DangerMemory — 创伤记忆(GAME_OVER→惩罚近期动作) ← IRL
│   └── MacroLibrary — 成功宏库(关卡完成→记录动作序列→跨关卡复用)
│
├── 规划层 (Planning)
│   ├── FlowField — Dijkstra流场(目标→全图距离场) ← 游戏业
│   ├── MCTSPlanner — 蒙特卡洛树搜索(UCB+alpha-beta剪枝) ← AlphaGo
│   └── GPCTPlanner — 边界层降解(关键格子枚举+外区简化) ← GPCT
│
├── 决策层 (Decision)
│   ├── RewardEngine — 多巴胺引擎(好奇+进步-危险-步耗) ← RL
│   ├── QLearner — Q学习(ε-greedy+TD更新) ← RL
│   └── BehaviorTree — 行为树(层次化决策) ← 游戏业
│
└── 验证层 (Verification) ← Dead-Zone+MUS
    ├── DeadZoneBreaker — 阈值熔断(低置信→HOLD)
    └── MUSDetector — 矛盾检测(互斥稳态双存)
```

### 工作流（每一步）
```
1. 处理帧 → 状态哈希 + 组件 + 玩家位置
2. 更新占据栅格 ← 碰撞反馈
3. 更新EML超图 ← Append-only + MUS双存
4. 更新创伤记忆 ← GAME_OVER/Reset惩罚
5. 计算流场 ← Dijkstra(目标→全图)
6. MCTS搜索 ← UCB选择+模拟+回传
7. Q学习更新 ← TD(S,A,R,S')
8. 行为树决策:
   a. 宏库匹配? → 重放成功序列
   b. MCTS置信度高? → 执行MCTS最优动作
   c. 流场方向明确? → 跟随流场梯度
   d. 有未测试动作? → ε-greedy探索
   e. BFS frontier导航
   f. 随机安全动作
9. Dead-Zone熔断检查 ← 置信度<阈值→HOLD
```

### 与V3对比
| 维度 | V3 (ExplorerAgent) | V4 (DopamineExplorer) |
|------|---------------------|------------------------|
| 驱动信号 | 无(纯探索) | 多巴胺奖励/惩罚(RL) |
| 空间推理 | 无 | Dijkstra流场+占据栅格 |
| 规划 | 无 | MCTS+GPCT边界层 |
| 失败学习 | 无 | 创伤记忆(IRL) |
| 记忆 | 状态图(覆写) | EML超图(无覆写+MUS双存) |
| 跨关卡 | MechanismMemory | MacroLibrary(成功序列复用) |
| 旋转跟踪 | 无 | CCG复数i^k |
| 决策 | 层次化选择 | 行为树+Q学习 |
| 验证 | 无 | Dead-Zone熔断+MUS检测 |
