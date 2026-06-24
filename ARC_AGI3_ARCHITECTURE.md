# TOMAS ARC-AGI-3 Agent 架构设计

## 一、核心发现

### 1.1 ARC-AGI-3 ≠ ARC-AGI v1
- **ARC-AGI v1**: 静态网格变换（input→output），我们之前的50个任务来自这个
- **ARC-AGI-3**: 交互式游戏环境，Agent 在64×64网格中通过动作序列完成任务

### 1.2 25个游戏环境
| # | 游戏 | 类型 | Baseline动作数 | 描述 |
|---|------|------|----------------|------|
| 1 | LS20 | keyboard | 7 | 网格移动解谜 |
| 2 | TR87 | keyboard | 6 | |
| 3 | TN36 | click | 7 | |
| 4 | WA30 | keyboard | 9 | |
| 5 | LP85 | click | 8 | |
| 6 | BP35 | keyboard_click | 9 | |
| 7 | SB26 | keyboard_click | 8 | |
| 8 | TU93 | keyboard_click | 9 | |
| 9 | SP80 | keyboard_click | 6 | |
| 10 | VC33 | click | 7 | |
| 11 | DC22 | keyboard_click | 6 | |
| 12 | FT09 | (none) | 6 | |
| 13 | AR25 | keyboard_click | 8 | |
| 14 | R11L | click | 6 | |
| 15 | RE86 | keyboard_click | 8 | |
| 16 | KA59 | keyboard_click | 7 | |
| 17 | LF52 | click | 10 | |
| 18 | SC25 | keyboard_click | 6 | |
| 19 | CN04 | keyboard_click | 6 | |
| 20 | S5I5 | click | 8 | |
| 21 | SU15 | click | 9 | |
| 22 | CD82 | keyboard_click | 6 | |
| 23 | M0R0 | keyboard_click | 6 | |
| 24 | G50T | keyboard | 7 | |
| 25 | SK48 | keyboard_click | 8 | |

### 1.3 动作空间
- `GameAction.RESET` (0): 重置当前关卡
- `GameAction.ACTION1-7` (1-7): 游戏特定动作
  - keyboard游戏: ACTION1=上, ACTION2=下, ACTION3=左, ACTION4=右
  - click游戏: 需要设置 `action.set_data({"x": N, "y": N})` 坐标
  - keyboard_click游戏: 两种都可用

### 1.4 帧数据结构 (FrameData)
```python
class FrameData:
    frame: list[list[list[int]]]  # [1, 64, 64] int8 网格
    state: GameState              # NOT_FINISHED | GAME_OVER | WIN | NOT_PLAYED
    levels_completed: int         # 已完成关卡数
    win_levels: int              # 总关卡数
    available_actions: list[int] # 可用动作列表
    action_input: ActionInput    # 上次执行的动作
```

### 1.5 评分公式（关键！）

**每关评分:**
```
level_score = ((baseline_actions / agent_actions) ** 2) * 100
level_score = min(level_score, 115)  # 上限115
```

**每游戏评分:**
```
game_score = Σ(level_score[i] × level_index[i]) / Σ(level_index[i])
# level_index 1-indexed，后面的关卡权重更大
```

**总评分:**
```
overall_score = average(all_25_game_scores)
```

**策略含义:**
- 用与人类相同的步数完成 → 100分
- 用一半步数完成 → 115分（上限）
- 用两倍步数完成 → 25分
- 未完成 → 0分
- 后面的关卡权重更大（第7关权重=7，第1关权重=1）

### 1.6 竞赛模式约束
- 必须通过API交互（不能本地运行）
- 每个游戏只能make()一次
- 只能Level Reset（不能Game Reset）
- 运行中不能查看分数
- Kaggle提交强制此模式

## 二、Agent 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    TOMAS Interactive Agent                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Perception   │  │  World Model │  │  Planner     │      │
│  │  Layer        │→ │              │→ │              │      │
│  │               │  │               │  │               │      │
│  │ • Grid Parse  │  │ • State Track │  │ • BFS/A*     │      │
│  │ • Sprite ID   │  │ • Transition  │  │ • MCTS       │      │
│  │ • Object Extract│ │   Model      │  │ • Heuristic  │      │
│  │ • Player Find │  │ • Goal Infer  │  │              │      │
│  └──────────────┘  └──────────────┘  └──────┬───────┘      │
│                                            │                │
│                            ┌───────────────▼──────────┐     │
│                            │    Action Executor       │     │
│                            │                          │     │
│                            │ • Keyboard actions       │     │
│                            │ • Click coordinates      │     │
│                            │ • Level Reset            │     │
│                            └───────────────┬──────────┘     │
│                                            │                │
│  ┌──────────────┐  ┌──────────────┐  ┌────▼───────────┐    │
│  │  Library     │  │  Memory      │  │  Environment   │    │
│  │  Learning    │← │              │← │  Interface     │    │
│  │              │  │               │  │                │    │
│  │ • Pattern    │  │ • Episode    │  │ • arc_agi API  │    │
│  │   Extract    │  │   Replay     │  │ • Frame Parse  │    │
│  │ • Strategy   │  │ • Baseline   │  │ • Score Track  │    │
│  │   Library    │  │   Compare    │  │                │    │
│  └──────────────┘  └──────────────┘  └────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心模块

#### Perception Layer (感知层)
- **GridParser**: 将64×64 int8网格解析为结构化对象
  - 颜色聚类 → 识别精灵/物体
  - 连通区域标记 → 分离独立物体
  - 玩家定位 → 通过帧间差异找出可移动实体
  - 目标识别 → 通过颜色/形状模式匹配

#### World Model (世界模型)
- **StateTracker**: 跟踪游戏状态变化
  - 记录 (state, action, next_state) 三元组
  - 识别哪些精灵是可移动的、哪些是静态的
  - 检测碰撞、触发事件
  
- **TransitionModel**: 学习动作→状态转移
  - keyboard: ACTION1-4 → 方向移动
  - click: (x,y) → 点击效果
  - 构建局部世界模型用于规划

- **GoalInferer**: 从观察中推断目标
  - levels_completed增加 → 正确行为
  - GAME_OVER → 错误行为
  - 通过帧间差异识别关键物体

#### Planner (规划器)
- **BFSPlanner**: 广度优先搜索最优动作序列
  - 在世界模型上模拟动作
  - 找到达到目标状态的最短路径
  - 支持 Level Reset 回退

- **HeuristicPlanner**: 基于启发式规则的快速决策
  - 朝目标方向移动
  - 避开障碍物
  - 收集关键物品

- **MCTSPlanner**: 蒙特卡洛树搜索（可选）
  - 对不确定环境进行rollout
  - UCB1选择最优分支

#### Library Learning (库学习)
- **PatternExtractor**: 从成功 episode 中提取模式
  - 识别重复出现的子序列
  - 计算MDL增益
  - 注册为新策略原语

- **StrategyLibrary**: 可复用的游戏策略
  - "移动到目标" 策略
  - "避开障碍" 策略
  - "收集所有物品" 策略

### 2.3 Agent 接口实现

```python
class TomasAgent(Agent):
    """TOMAS Interactive Agent for ARC-AGI-3."""
    
    def __init__(self):
        self.perception = GridParser()
        self.world_model = WorldModel()
        self.planner = Planner()
        self.memory = EpisodeMemory()
        self.library = StrategyLibrary()
        self.current_strategy = None
        
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if latest_frame.state is GameState.GAME_OVER:
            return True  # Will reset
        # Check if stuck (no progress for N steps)
        if self.memory.is_stuck(threshold=50):
            return True
        return False
    
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # 1. Parse current frame
        grid = np.array(latest_frame.frame[0])  # 64x64
        objects = self.perception.parse(grid)
        
        # 2. Update world model
        self.world_model.update(latest_frame, objects)
        
        # 3. If GAME_OVER, reset level
        if latest_frame.state is GameState.GAME_OVER:
            return GameAction.RESET
        
        # 4. If first step, initialize strategy
        if len(frames) <= 1:
            self.current_strategy = self.library.select_strategy(
                objects, latest_frame.available_actions
            )
        
        # 5. Plan next action
        action = self.planner.plan(
            self.world_model, 
            self.current_strategy,
            latest_frame.available_actions
        )
        
        # 6. Record action
        self.memory.record(latest_frame, action)
        
        return action
```

### 2.4 文件结构

```
tomas-arc3-solver/
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── tomas_agent.py        # 主Agent类
│       ├── perception/
│       │   ├── __init__.py
│       │   ├── grid_parser.py    # 64x64网格解析
│       │   ├── object_detector.py # 物体检测
│       │   └── player_tracker.py  # 玩家位置追踪
│       ├── world_model/
│       │   ├── __init__.py
│       │   ├── state_tracker.py   # 状态跟踪
│       │   ├── transition_model.py # 转移模型
│       │   └── goal_inferrer.py   # 目标推断
│       ├── planner/
│       │   ├── __init__.py
│       │   ├── bfs_planner.py     # BFS搜索
│       │   ├── heuristic_planner.py # 启发式
│       │   └── mcts_planner.py    # MCTS
│       ├── learning/
│       │   ├── __init__.py
│       │   ├── pattern_extractor.py # 模式提取
│       │   └── strategy_library.py  # 策略库
│       └── memory/
│           ├── __init__.py
│           └── episode_memory.py  # Episode记忆
├── kaggle/
│   ├── notebook_template.ipynb   # Kaggle提交notebook
│   └── agent_harness.py          # Agent运行框架
└── tests/
    └── test_agent.py
```

## 三、实现优先级

### Phase 1: MVP (最小可行产品) — 今天
1. GridParser: 基本网格解析（颜色聚类、连通区域）
2. StateTracker: 帧间差异跟踪
3. HeuristicPlanner: 基于规则的简单策略
4. TomasAgent: 基本Agent框架
5. 在LS20上测试

### Phase 2: 核心能力 — 明天
1. BFSPlanner: 在已知世界模型上搜索
2. TransitionModel: 学习动作效果
3. GoalInferer: 从奖励信号推断目标
4. Level Reset策略: 卡住时回退重试
5. 在5个游戏上测试

### Phase 3: 优化 — 后天
1. MCTSPlanner: 对不确定环境进行搜索
2. PatternExtractor: 从成功episode学习
3. StrategyLibrary: 跨游戏策略迁移
4. 全部25个游戏测试
5. 效率优化（减少步数）

### Phase 4: Kaggle提交 — 6月30日前
1. Competition mode适配
2. Kaggle notebook打包
3. 提交验证
