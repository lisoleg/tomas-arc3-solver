# ARC-AGI-3 Solver 架构 V3 — 图基探索 + 机制迁移 + 精确规划

## 调研总结 (2026-06-24)

### 竞赛概况
- **截止日期**: Milestone 1 = 2026-06-30 (6天!)
- **奖金**: $850K (M1第一名$25K)
- **评分**: RHAE = min(1, human/agent)², 关卡权重递增, 总分=所有游戏平均
- **提交**: Kaggle Notebook, CPU/GPU ≤ 9h, 无网络, 25个游戏

### 当前排名
| 名次 | 方案 | 得分 | 方法 |
|------|------|------|------|
| 1 | StochasticGoose | 12.58% | CNN + RL, 预测帧变化 |
| 2 | Blind Squirrel | 6.71% | 有向状态图 |
| 3 | dolphin-in-a-coma | 30/52关 | 图基探索(训练无关) |
| - | 前沿AI (GPT-5.4等) | <1% | LLM |
| - | 人类 | 100% | - |

### 关键发现
1. **纯探索 > LLM**: 图基探索(无训练)显著优于LLM方法, LLM甚至不如随机
2. **第3名方法**: 帧分割(单色连通组件) + 状态栏遮蔽 + 5层优先级 + 有向状态图 + frontier导航
3. **第3名短板**: 无跨关卡迁移, 状态空间线性增长, 确定性假设
4. **官方核心能力**: 探索 + 建模 + 目标设定 + 规划执行
5. **RHAE平方惩罚**: 2×步数→25%得分, 效率是王道

## V3 架构设计

### 核心思想
**图基探索为基座 + 机制检测为增强 + 精确规划为加速器**

```
ExplorerAgent (新核心)
├── FrameProcessor (帧处理引擎)
│   ├── connected_components() — 单色连通组件分割
│   ├── mask_status_bar() — 状态栏检测与遮蔽
│   ├── state_hash() — 遮蔽后网格哈希
│   └── priority_groups() — 5层优先级分组
│
├── StateGraph (有向状态图)
│   ├── StateNode — 节点(状态哈希, 动作表, frontier距离)
│   ├── edges — 边(源→目, 动作, 转移信息)
│   ├── frontier_nodes — 含未测试动作的节点
│   ├── bfs_to_frontier — 最短路径到frontier
│   └── mark_reset_action — 标记reset动作(修复第3名bug)
│
├── ActionSelector (层次化动作选择)
│   ├── Level 1: 当前状态未测试的高优先级动作
│   ├── Level 2: BFS导航到最近frontier
│   └── Level 3: 降低优先级递归
│
├── MechanismDetector (跨关卡机制检测) ← V3新增
│   ├── detect_goal — 检测目标(消失/变化的组件)
│   ├── detect_switcher — 检测属性切换器
│   ├── detect_wall — 检测不可通过区域
│   ├── detect_refill — 检测步数补充
│   └── memory — 跨关卡持久化记忆
│
└── StrategicPlanner (精确规划) ← 机制明确时激活
    ├── bfs_pathfinding — BFS寻路到目标
    ├── ccg_rotation — 复数i^k旋转跟踪
    └── bounce_strategy — switcher bounce优化
```

### 工作流
```
每一步:
1. 处理帧 → 状态哈希 + 组件列表
2. 更新状态图 → 添加节点/边, 标记reset
3. 检测机制 → 更新MechanismMemory
4. 选择动作:
   if 机制明确(MechanismMemory置信度高):
     → StrategicPlanner (BFS + CCG)
   else:
     → ActionSelector (层次化探索)
5. 执行动作 → 观察结果
6. 跨关卡: 关卡完成时保留MechanismMemory, 重置StateGraph
```

### 与V2对比
| 维度 | V2 (TomasAgent) | V3 (ExplorerAgent) |
|------|-----------------|---------------------|
| 感知 | 5×5 sprite检测 | 单色连通组件分割 |
| 状态跟踪 | 手动位置追踪 | 状态哈希自动跟踪 |
| 探索策略 | 无(直接规划) | 层次化图探索 |
| 跨关卡 | 无 | MechanismMemory |
| reset处理 | 无 | 标记reset动作 |
| 适用范围 | 仅ls20 | 所有25个游戏 |
| 代码量 | ~5700行 | ~1500行(目标) |
