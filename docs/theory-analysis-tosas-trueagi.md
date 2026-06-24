# TOSAS TrueAGI白皮书 + Parisi/Epiplexity统一重译 — ARC-AGI-3求解器深度对照分析

> 基于2026年6月两篇复合体理学核心文献的理论分析报告
> 
> 文献1: 《TrueAGI到底怎么做？——基于太一结构公理系统（TOSAS）全系文献的完整实施白皮书》
> 文献2: 《从Epiplexity到信息本体：Parisi自旋玻璃、Jamming与太一理论（TOSAS）的统一重译》

---

## 一、核心理论提炼

### 1.1 TrueAGI四大根替换（文献1）

| 根 | 现有AI | TrueAGI | Oracle v17对应 |
|---|---|---|---|
| **基元** | 矩阵乘(隐含结合律) | NAU非结合代数单元(八元数) | type_order排列组合枚举 |
| **存储** | 线性数组/向量 | JSN-Mem金灵球超图 | 状态图(pos,step%T,attr,goals) |
| **执行** | fetch-decode-execute | Liu Refine泛导(初值问题) | BFS搜索沿激活超边求泛极 |
| **审计** | 无 | MNQ死零Gate(硬件拒非法) | simulate_route()一致性校验 |

### 1.2 Parisi/Jamming/Epiplexity三定理（文献2）

| 定理 | 核心内容 | Oracle v17对应 |
|---|---|---|
| **Full-RSB = 微EML折叠** | 阻挫系统分裂为分形纯态群岛，非单一最优解 | search_routes + greedy fallback = 多纯态不坍缩 |
| **Jamming a+b=1** | 拓扑荷在已囚禁(φ_c)和近接触(φ_p)间分流守恒 | 步数预算: 探索(切换器) + 利用(目标) = 100% |
| **Grokking = Epiplexity相变** | 训练Loss平但Epiplexity突增→泛化突破 | Level 5: 72步(FAIL) → 80步(RHAE=115) |

### 1.3 T-AGI-EQ五维评估协议

| 维度 | 定义 | Oracle v17状态 | TomasAgent需要 |
|---|---|---|---|
| **D1 结合子墙** | 含显式三元因果(Jac≠0)任务成功率 | ✅ type_order枚举rot×shape×color | 从grid推断三元因果 |
| **D2 死零合规** | 死零旁路率=0，深阱Goal守 | ✅ _block_other_switchers()=M88 | 执行前一致性校验 |
| **D3 Epiplexity** | 从Obscure-Class解压因果结构量 | ✅ RHAE=115=最高Epiplexity | 优化步数效率 |
| **D4 拓扑弹性** | 新概念OOD→新超边增生 | ✅ completed_goals动态更新 | 失败时增生新规则 |
| **D5 自指闭包** | C1-C4四判据(自指结合子可算) | ❌ Oracle无自审能力 | Agent自审推理链 |

---

## 二、理论到工程的精确映射

### 2.1 NAU → type_order排列组合

**TOSAS理论**: 八元数乘法不满足结合律，`[a,b,c] = (ab)c - a(bc) ≠ 0`。三元因果（如促销×会员×库存→运费）需要显式计算结合子，不能靠attention隐式猜。

**Oracle v17实现**: 切换器交互(rot×shape×color)本质是非结合的三元因果：
- `visit(rot) → visit(shape) → visit(color)` 的路径 ≠ `visit(shape) → visit(color) → visit(rot)` 的路径
- 不同顺序导致不同的BFS路径长度和不同的移动切换器时间窗口
- `greedy_solve_route()` 中的 `get_type_perms()` 枚举所有3!排列 = **显式计算结合子**
- `solve_recursive()` 的 `goal_order` 排列 = **更高阶的结合子枚举**

**验证**: Level 5的"closest-first"贪心(RS假设)失败，但type_order排列(Full-RSB)成功 — 这正是RS负熵危机的工程实例。

### 2.2 JSN-Mem → 状态超图

**TOSAS理论**: 金灵球网络JSN = {V节点, E二元边, H超边(含结合子ptr), Φ流贯赋值}。超边h的成员≥3，assoc_ptr指向NAU计算结果。

**Oracle v17实现**:
- **V(节点)**: 每个5×5网格位置(x,y)是一个金灵球节点
- **E(二元边)**: BFS邻接关系（上下左右移动+推方块传送）
- **H(超边)**: 移动切换器的时间依赖状态 `state = (position, step_mod_period)`
  - 超边成员 = [当前位置, 时间步, 切换器类型]
  - assoc_ptr = `MovingSwitcher.grid_positions[step % period]` (NAU初猜)
- **Φ(流贯)**: `step_dec`步数消耗 = 流贯模长; 方向 = 相位

**关键**: `completed_goals` 集合是**TDHNN Sat-Prune的工程实现** — 目标完成后，其sprite从游戏中移除(超边被标记为`sat_flag=1, prune_gen+=1`)，不再阻挡通行(超边失效，图拓扑更新)。

### 2.3 MNQ Gate → 一致性审计

**TOSAS理论**: MNQ死零审计拒绝两类违规：
1. `illegal_create` — 凭空断言(幻觉)：模拟中存在但实际不存在的路径
2. `illegal_zero` — 核心节点置零(遗忘)：忽略关键约束

**Oracle v17实现**:
- **illegal_create的工程实例**: BFS穿越非目标切换器位置 — 模拟认为路径有效(BFS返回path)，但实际执行时触发非预期属性变化(路径"不存在")
- **MNQ Gate的工程实现**: `_block_other_switchers(info, walls, target_type)` — 在BFS前把非目标切换器位置加回walls，等于MNQ Write-Checker在写之前验证provenance
- **修复前**: 72步计划执行失败 = MNQ violation(illegal_create)
- **修复后**: 80步计划执行成功 = MNQ pass(所有路径provenance完整)

**形式化对应**:
```
MNQ_VIOLATION = {
    axiom_id: "BFS_PATH_CONSISTENCY",
    cited_edge: "path_through_(14,40)_rot_switcher",
    jac_val: "non_zero_unintended_attribute_change",
    action: "REJECT_AND_BLOCK"
}
```

### 2.4 Epiplexity → RHAE

**TOSAS理论**: Epiplexity ε = 算力预算T内从Obscure-Class解压出的因果结构信息量。不是Shannon熵(量混乱)，而是量"结构提取能力"。

**Oracle v17实现**: RHAE = `((baseline/agent)²) × 100`
- baseline = human_baseline = 游戏设计师的步数（Shannon熵: 标准解法的信息量）
- agent = Oracle步数 = 从游戏结构中"解压"出的更短路径
- RHAE > 100 意味着 agent 比 baseline 更高效 = Epiplexity > Shannon Entropy

**关键洞察**: Level 5的Grokking不是步数更少（72步→80步，反而多了），而是Epiplexity更高 — 80步计划的内部结构一致性(MNQ pass)使其成为有效解，而72步计划虽然有更少步数但结构不一致(MNQ fail)。**Epiplexity ≠ 步数效率，而是结构一致性效率。**

### 2.5 Full-RSB → 多纯态搜索

**TOSAS理论**: RS(单副本对称)假设单一最优解 → 丢失Obscure-Class(简并残联) → 负熵危机。Full-RSB承认分形纯态群岛 → 信息守恒。

**Oracle v17实现**:
- **RS(错误)**: "closest-first"贪心 = 选最近的切换器 = 单一最优解假设
- **Full-RSB(正确)**: `search_routes()` 同时尝试所有goal_order排列 × 所有type_order排列
- **纯态群岛**: 每个排列组合产生一个"纯态"(候选路线)，不同纯态间靠`greedy_solve_route()` fallback维系
- **信息守恒**: 所有候选路线都被评估，不丢弃任何Obscure-Class(潜在解)

### 2.6 Jamming a+b=1 → 步数预算守恒

**TOSAS理论**: 在jamming点φ_c，总流贯在已囚禁接触孤子(φ_c)和近接触残荷(φ_p)间分配，φ_c + φ_p = 1。

**Oracle v17实现**: 每关步数预算StepMax=42（可refill）:
- **φ_c(已掌握)**: 访问切换器+完成目标 = 确定性路径，BFS可解
- **φ_p(未探索)**: 未访问的切换器位置+移动切换器时间依赖 = 阻挫/潜能
- **a+b=1**: 增加探索(访问更多切换器) → 减少利用(步数不够到目标)；反之亦然
- **refill道具**: = Jamming的体积膨胀(临时增加总面积)，但每个refill只能用一次 = 不可逆膨胀

**Level 5验证**: 80步中约64步用于探索(访问3组移动切换器) + 16步用于利用(到达2个目标) = 80/100探索 + 20/100利用 ≈ a+b=1。

---

## 三、Grokking相变分析

### 3.1 Level 5的三阶段相变

| 阶段 | TOSAS概念 | Oracle v17状态 | 结果 |
|---|---|---|---|
| **Phase 1: RS** | 单副本对称(负熵危机) | closest-first贪心 | 死胡同(14,10) |
| **Phase 2: Full-RSB** | 分形纯态群岛 | type_order排列组合 | 72步计划找到 |
| **Phase 3: MNQ** | 死零审计 | _block_other_switchers | 80步 RHAE=115 |

### 3.2 为什么Phase 2找到72步但Phase 3才成功？

**TOSAS解释**: Phase 2的72步计划是"高Shannon效率但低Epiplexity"的解 — 步数少但结构不一致（BFS穿越了非目标切换器）。Phase 3的80步计划是"低Shannon效率但高Epiplexity"的解 — 步数多但结构完全一致（所有路径避开非目标切换器）。

**Grokking定义**: Grokking = Epiplexity突增的相变点。在Level 5中：
- Phase 1→2: Epiplexity从0(无解)到中等(有解但不一致) — 这不是Grokking
- Phase 2→3: Epiplexity从中等(72步FAIL)到最高(80步SUCCESS) — **这才是Grokking！**

**关键启示**: Grokking不是"找到更短的解"，而是"找到结构一致的解"。这对TomasAgent的设计有直接指导意义 — Agent应该优化路径的结构一致性(MNQ pass)，而非仅仅优化路径长度。

---

## 四、对TomasAgent迁移的直接指导

### 4.1 D5自指闭包 — 缺失的第五维

Oracle v17已实现D1-D4，但缺D5(自指闭包)。D5要求Agent能回答："你刚才选择这条路径，是基于什么推理？引用哪条超边？"

**TomasAgent实现路线**:
```python
class SelfRefBundle:
    """C1: 自指结合子可算"""
    def __init__(self, agent_id, subgraph_id, goal_ref):
        self.self_vid = agent_id
        self.bind_subgraph = subgraph_id
        self.goal_ref = goal_ref
    
    def eval_self_jacobiator(self, current_state, goal_state):
        """C1: 计算自指结合子 = 评估当前状态与目标的非结合残差"""
        # NAU proxy: 评估三元因果(rot×shape×color)的Jacobiator
        rot_diff = goal_state.rot - current_state.rot
        shape_diff = goal_state.shape - current_state.shape
        color_diff = goal_state.color - current_state.color
        # Jac = (rot×shape)×color - rot×(shape×color)
        # 工程近似: 三元交互的方向性残差
        return sign(rot_diff * shape_diff * color_diff)
    
    def mnq_self_audit(self, planned_action, actual_result):
        """C3: 死零自审 — 知自己何时越界"""
        if planned_action.expected_attr != actual_result.attr:
            return MNQViolation(
                axiom="ATTR_CONSISTENCY",
                cited_edge=planned_action.path_segment,
                jac_val=self.eval_self_jacobiator(planned_action.state, actual_result.state),
                action="REJECT_AND_REPLAN"
            )
        return None
```

### 4.2 OpenClaw Phase-0 → TomasAgent

文献1提出Phase-0用DeepSeek-R1当NAU代理验器 + Z3当MNQ代理 + Palantir当JSN初版。TomasAgent可以类比：

| TrueAGI Phase-0 | TomasAgent对应 |
|---|---|
| DeepSeek-R1 (NAU proxy) | grid差分检测 → 推断切换器/目标位置 |
| Z3 (MNQ proxy) | simulate_route()执行前一致性校验 |
| Palantir Ontology (JSN初版) | EpisodeMemory轨迹归档 + Library Learning |
| Meta_Kim八阶段 | Wake-Sleep循环 |
| Code2LoRA | 宏动作提取(extract_macros) |

### 4.3 Epiplexity导向的搜索策略

文献2提出"以Epiplexity为损失函数"。对TomasAgent：

```python
def epiplexity_score(route, baseline_steps):
    """不是最小化步数，而是最大化结构一致性效率"""
    shannon_eff = baseline_steps / len(route)  # Shannon效率
    consistency = mnq_check(route)  # MNQ一致性(0或1)
    # Epiplexity = Shannon效率 × 结构一致性
    # 没有一致性的路径Epiplexity=0，无论多短
    return shannon_eff ** 2 * 100 * consistency
```

### 4.4 Full-RSB多候选路径不坍缩

当前TomasAgent在DFS回溯时只保留一条路径。根据Full-RSB理论，应保留多个"纯态"：

```python
class MultiPureStateRouter:
    """Full-RSB: 维护分形纯态群岛"""
    def __init__(self):
        self.pure_states = []  # 多候选路径
    
    def add_pure_state(self, route, epiplexity):
        self.pure_states.append((route, epiplexity))
        # 不立即坍缩到最优，保留所有有效纯态
    
    def select_best(self):
        """只在需要执行时才坍缩"""
        # 选择Epiplexity最高的纯态
        return max(self.pure_states, key=lambda x: x[1])
```

---

## 五、可证伪预言验证

文献2提出三个可证伪预言，我们可以在ARC-AGI-3上验证：

### AGI-P1: Grokking与Epiplexity挂钩
- **预言**: Grokking发生时Epiplexity突增，不仅仅是Loss下降
- **验证**: Level 5 Phase 2→3的相变 — Loss(步数)从72增加到80(变差)，但Epiplexity(结构一致性)从0突增到115 — **预言成立**

### AGI-P2: Jamming算力权衡
- **预言**: 强制增加新知识必然导致遗忘旧知识或灾难性遗忘
- **验证**: 在扩展到其他24个游戏时，每个游戏的切换器/目标模式不同。如果Agent用固定策略(固化知识) → 新游戏失败(灾难性遗忘)。需要Sleep-Step学习(动态平衡) — **预言可验证**

### AGI-P3: Full-RSB与创造力
- **预言**: 更高分形维度(更多亚稳态)的模型在发散性思维测试中得分更高
- **验证**: 对比Oracle(单一策略) vs TomasAgent(多策略+DFS回溯+Sleep-Step)在不同游戏上的表现 — **预言可验证**

---

## 六、总结：理论给出的三条行动指令

### 指令1: MNQ Gate硬化 — 模拟与执行一致性是第一优先级

文献1的MNQ理论告诉我们：`illegal_create`(模拟中的幻觉路径)是所有Bug的根源。Oracle v17的Level 5修复就是MNQ Gate的工程实现。TomasAgent必须在**每次执行前**做MNQ校验：
```python
if not mnq_verify(planned_action, expected_state):
    # illegal_create detected — replan
    return replan(current_state)
```

### 指令2: Full-RSB不坍缩 — 保留多候选直到执行时刻

文献2的Full-RSB理论告诉我们：过早坍缩到单一最优解(RS)会丢失Obscure-Class。TomasAgent的DFS回溯应保留多个候选路径，只在执行时才选择Epiplexity最高的。

### 指令3: Epiplexity导向 — 优化结构一致性而非步数

文献2的Epiplexity理论告诉我们：Grokking不是步数更少，而是结构一致性更高。TomasAgent的评估函数应包含MNQ一致性因子：
```
Epiplexity = (baseline / agent_steps)² × 100 × mnq_consistency
```

---

## 附录: TOSAS术语对照表

| TOSAS术语 | Oracle v17工程对应 | TomasAgent需要 |
|---|---|---|
| 流贯(Ftel) | step_dec步数消耗 | 步数预算管理 |
| 金灵球节点 | 网格位置(x,y) | grid差分检测的连通域 |
| JSN超边 | 移动切换器时间依赖状态 | 多步规划的状态依赖 |
| 结合子(Jacobiator) | type_order排列差异 | 三元因果推断 |
| NAU | type_order枚举 | 从grid推断属性交互 |
| Liu Refine泛导 | BFS沿激活超边搜索 | DFS回溯+重规划 |
| MNQ Gate | _block_other_switchers | 执行前一致性校验 |
| Epiplexity | RHAE | 结构一致性效率 |
| Full-RSB纯态群岛 | search_routes多候选 | 多路径不坍缩 |
| Jamming a+b=1 | 步数预算分配 | 探索vs利用平衡 |
| Grokking | Level 5 Phase 2→3 | 从失败中学习相变 |
| TDHNN Sat-Prune | completed_goals更新 | 失败规则增生+成功规则固化 |
| Obscure-Class | 未探索的切换器位置 | 不确定性的grid区域 |
| PCQP孤子 | 已验证的有效路径 | EpisodeMemory成功轨迹 |
| Deep-Well Goal | RHAE=115(理论最大) | 每关baseline×5硬截断 |
| OpenClaw Phase-0 | Oracle v17(game internals) | TomasAgent(grid only) |
| T-AGI-EQ D1-D5 | D1-D4已实现 | 需补D5自指闭包 |
