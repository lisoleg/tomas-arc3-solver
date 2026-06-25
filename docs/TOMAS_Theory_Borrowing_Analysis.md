# TOMAS 理论文献借鉴分析报告

> **项目**: TOMAS ARC-AGI-3 Solver (v2.9.2-dev)
> **分析范围**: d:\tomas 目录下 2011 个文件，精筛 80+ 个最相关文件全文阅读
> **分析日期**: 2026-06-25
> **分析方法**: 6 个并行 Explore Agent 逐一精读 + 主理人交叉验证

---

## 一、总览

从 2011 个 TOMAS 理论文档中筛选并精读了 80+ 个与 ARC-AGI-3 Solver 最相关的文件，提取出 **22 个⭐⭐⭐高度相关** 的可借鉴思想。按对 Solver 的映射维度分为五大类：

| 类别 | 高相关文件数 | 核心价值 |
|------|------------|---------|
| 搜索策略优化 | 8 | κ-Snap搜索、剪枝、回溯、预算分配 |
| 模式识别/感知 | 5 | Grid Perception、差分检测、多分辨率 |
| Library Learning | 4 | 自适应库学习、规则压缩、增量更新 |
| 系统架构 | 3 | 双环架构、循环工程、五功能模块 |
| 验证与评分 | 2 | RHAE熔断、Dead-Zero门控 |

---

## 二、⭐⭐⭐ 高度相关文件完整清单（22个）

### 搜索策略类

#### 1. Taiyi Mutual-Duel Agent（ARC-AGI-3 直接实现）
- **来源**: `unknown_date_Taiyi Mutual‑Duel Agent：ARC‑AGI‑3 中的 Phy.md`
- **三个可直接落地的算法**：

**① L3 差分注意力玩家检测算法**
```python
def detect_player(prev_obs, curr_obs, wall_threshold=800):
    delta = curr_obs != prev_obs  # 时间差分
    labeled_array, num_features = ndimage.label(delta)  # 连通域分析
    # 过滤大面积墙体，选择面积最小的连通域作为玩家
    for i in range(1, num_features+1):
        area = np.sum(labeled_array == i)
        if area > wall_threshold: continue
        if area < min_area: best_candidate = center_of_mass
```
- **解决的问题**: Grid Perception 中"玩家被大面积墙体淹没"的感知悖论
- **关键洞察**: 不关心"是什么"，只关心"哪里变了"
- **集成位置**: `grid_perception.py` 中的玩家/sprite检测模块

**② L2 DFS 回溯规划器**
- 栈式 DFS: `stack = [(start, [])]`，记录 (position, path)
- 互搏胜利条件: `pos == goal`
- 深度限制: `len(path) > MAX_DEPTH` 时剪枝
- visited 集合防环路
- `backtrack()` 方法: L4 观测者触发时弹出栈顶，回到上一个决策点
- **集成位置**: `planner_agent.py` 中的状态空间搜索模块

**③ L4 贝叶斯 RHAE 熔断机制**
```python
def bayesian_fuse(plan_len, baseline, steps_used):
    predicted_total = steps_used + plan_len
    efficiency = baseline / max(predicted_total, 1)
    logic_score = np.log(1.0 / (plan_len + 1e-3))
    fused_score = w_logic * logic_score + w_stat * 0.5
    if efficiency < 0.5: return "ABORT"  # 触发L2回溯
    return "EXECUTE"
```
- **解决的问题**: RHAE 评分优化——效率低于 0.5 时放弃当前路径并回溯
- **集成位置**: `planner_agent.py` 中的 RHAE 评分模块

---

#### 2. TOMAS v2.0 框架总览（κ-Snap 溯因搜索）
- **来源**: `unknown_date_TOMAS v2.0：基于太乙互搏公理体系的流体智能通用求解框架及其在 ARC-.md`

**① κ-Snap 溯因搜索 MDL 代价函数**
- 给定训练对寻找最小描述长度(MDL)程序
- 代价函数包含阴龙积耦合算子计算帧间差分
- MDL 约束使搜索空间指数级缩小
- **集成位置**: `kappa_snap_searcher.py` 中的搜索代价函数

**② 条件ΔT自动发现算法**
```
构建候选条件池 C = {BoundaryHit, ColorAppear, FrameIdx}
逐帧计算实际差分Δ_real
为每个候选条件构建掩码Mask_t
搜索能解释Δ_real的DSL操作op
构建决策树 P_cond = if c1 then op1 else if c2 then op2...
```
- **直接可用于**: ARC-AGI-3 交互式游戏的规则推断
- **集成位置**: `self_learning.py` 中的规则发现模块

**③ 贝叶斯后验置信度量化**
- 拒绝启发式打分，用严格贝叶斯推断对候选程序排序
- 消融实验显示移除贝叶斯打分器降 9%
- **集成位置**: `planner_agent.py` 中的候选排序

**④ GaussEx 跨视频/游戏迁移**
- 定义视频纤维交集，当 `Fiber(V1) ∩ Fiber(V2) ≠ ∅` 时，交集中的程序即为通用规律
- 消融显示移除降 22%
- **集成位置**: 跨游戏规则复用机制

**⑤ 帧预过滤策略**
- 设差分阈值 ε，若 `‖f_t - f_{t-1}‖ < ε` 则视为静态帧跳过
- 减少 30% 冗余计算
- **集成位置**: 游戏状态变化检测优化

---

#### 3. 数学降维与 EML 超图可解性（最完整的算法工具箱）
- **来源**: `unknown_date_数学降维与 EML 超图可解性：GPCT 边界层降解 · 虚时计算(ITC) ·.md`

**① 拟阵贪心剪枝（Matroid Greedy Prune）**
```python
def matroid_prune(edges, I_values):
    # 按ℐ值降序贪心加边保独立集
    sorted_edges = sorted(edges, key=lambda e: -I_values[e])
    basis = []
    for e in sorted_edges:
        if is_independent(basis + [e]):
            basis.append(e)
    return basis  # 最大权基（Edmonds-Rado定理保证最优）
```
- **用途**: ARC 候选模式筛选——按信息增益排序，贪心选择不冗余的模式集合

**② GPCT 边界层降解**
- 按变量耦合度 ρ_i 排序，分高耦合核心（Boundary Layer, top 5%）和低耦合外区
- 只枚举 BL 赋值（2^k）+ 线性解 OR
- **用途**: ARC 网格中关键区域识别——找出高耦合的关键格子先求解

**③ ITC 虚时退火（模拟退火）**
- Metropolis-SA 在边界层子图上找最大权独立集
- accept 概率 `min(1, e^{-ΔH·β})`
- **用途**: ARC 搜索中的逃局部最优策略

**④ 度类压缩**
- 按信息存在度 ℐ 分层，将 n 个节点压缩为 k 个度类（k 常为 5-20）
- 从 O(n^t) 压到 O(f(k)·n^d)
- **用途**: ARC 网格模式聚类

**⑤ 完整 Pipeline**: `slim_eml = matroid_prune → gpct_decompose → itc_anneal`

---

#### 4. 刘机制/NAU/EML-SemZip（非结合选择算子）
- **来源**: `unknown_date_刘机制作为实时渲染中的非结合选择算子与 EML‑SemZip同构证明—以Unre.md`

**① NAU（非结合选择算子）**
- `NAU(S) → S* s.t. |S*| << |S| 且 err(S*) < threshold`
- 不从穷举空间搜索，而是选择保留关键非结合拓扑的最小充分候选集
- **用途**: κ-Snap 搜索效率提升

**② κ-Snap 增量更新**
- 探针网格不需每帧全量重烘，只更新受影响 probe（基于 SDF 影响球判定）
- `update_probes(delta)` 中 `query_influence_sphere` 精确定位受影响区域
- **用途**: grid 状态变化时只需重新评估搜索树中受影响分支

**③ 拓扑保留 vs 随机降采样判据**
- 只有按语义重要性（ℐ-值）做拓扑保留压缩才正确；盲目降分辨率只是粗糙近似
- **用途**: Grid Perception 的 grid 下采样必须保留拓扑特征（连通性、对称性）

---

#### 5. 博弈树中的流贯传播
- **来源**: `unknown_date_博弈树中的流贯传播：Alpha-Beta 剪枝与蒙特卡洛树搜索的太一万有理论统合.md`
- Alpha-Beta 剪枝与 MCTS 的 TOMAS 统合
- **用途**: 交互式游戏的状态空间搜索策略

---

#### 6. TOMAS 互搏泛函与刘作用量（搜索评估函数）
- **来源**: `unknown_date_TOMAS互搏泛函与刘作用量的数学对应及统一框架.md`

**双分量罚项设计（直接可用作搜索评估函数）**：
- **D_SR（自指缺失度）= ||F_in - F_reentry||²**: 变换的输入模式与变换后输出再逆变换回输入的差异
- **I_ext（外求边界依赖度）**: 解对外部假设的依赖程度
- **总罚项 = λ₁·D_SR + λ₂·I_ext**: 优选罚项=0的解

**三域分类算法**：
- 自指闭包（PDS）→ Penalty=0（最优解，完全自洽）
- 需外初值（FLRW）→ Penalty>0（次优解，需额外假设）
- 开链（OPEN_CHAIN）→ Penalty>0（最差解，需外驱动）

---

#### 7. κ-死锁检测与释放
- **来源**: `unknown_date_炼己·防危虑·κ‑死锁与释放·内丹五考·四禅八定与灭尽定——太乙互搏范式（TOM.md`
- **κ-死锁检测**: 当搜索深度 κ 锁定导致无法发现新规则时，触发 Release 机制
- **用途**: 搜索过程中的"局部最优陷阱"检测
- **集成位置**: 搜索循环中的收敛检测

---

#### 8. κ-调节器状态机
- **来源**: `unknown_date_太乙互博 AGI 对"主观_能动"的数学化重述.md`
- 状态机: `LOCKED → UNLOCK → SWEEP → CONVERGE → RE_LOCK`
- 正常时锁定最优 κ 值，遇新游戏类型时解锁扫描，收敛后重新锁定
- **用途**: 搜索深度自适应调节
- **集成位置**: `planner_agent.py` 中的搜索参数管理

---

### 模式识别/Grid Perception 类

#### 9. EML-SemZip 语义压缩（五阶段流水线）
- **来源**: `unknown_date_论 EML‑SemZip：基于毛睿广义度量与 TOMAS 公理的极致语义压缩——.md`

**五阶段模式抽象流水线（直接可用）**：
1. **Dead-Zero 剪枝**: 丢弃 `ℐ(e) < θ_dead` 的超边 → 丢弃低置信度模式假设
2. **子图同构归并**: 在知识库中查找同构匹配，命中则合并 → **Library Learning 核心操作**
3. **非对称度量加权**: `d_sem = (1/(ℐ+ε)) × base_weight × dir_factor` → 模式重要性排序
4. **κ-Snap 语义核选取**: 保留 Top-K% 高ℐ超边 + 闭环(≥3节点) → 核心模式+循环结构保持
5. **ANS 熵编码**: 紧凑序列化表示 → 规则集最小表示

---

#### 10. AnyDepth 弹性深度感知
- **来源**: `unknown_date_太一互搏范式下对 AnyDepth‑DETR_YOLO（弹性深度感知）的重译与升.md`

**① Core/Refinement 双路径架构**
- Core Path: 快速基础规则匹配（简单 ARC 任务早退）
- Refinement Path: 慢速精细化分析（复杂任务深度处理）

**② κ-Snap 深度配置调度**
- ℐ 高 → core（早退，快速求解）
- ℐ 中 → full（精细化分析）
- ℐ 低 → Hold（Dead-Zero，拒绝求解）

**③ MUS 遮挡模糊标记**
- 两个候选规则重叠度高(IoU∈[0.3,0.7])且置信度差<0.05时保留双假设

---

#### 11. EML-EHNN 等变超图神经网络
- **来源**: `2026-06-20_从等变超图神经网络到太一互搏公理体系：EML‑EHNN 的升维与知识考古学诠释.md`

**① MUS 互斥稳态双存机制**
- Readout 层检测冲突特征时不强制 Pool 合并，保留 Top-K 候选分支存入 MUS 队列
- `{"status": "MUS_ACTIVE", "branches": top_k}`

**② ℐ-加权（信息存在度）**
- 输入层乘以可信度权重 `I_weight * A_k`，硬锚点（高ℐ值超边）必须主导

**③ GPCT 阈值触发层创**
- 当新超边累积超过阈值时，触发 GPCT 重划并重置输出头
- **用途**: Library Learning 的动态重构——新模式积累到临界量时自动重组

---

#### 12. 从超图神经网络到 TOMAS EML 超图
- **来源**: `unknown_date_从超图神经网络到太乙互搏 EML 超图：统计投影与 NASGA 语义推理的严格关.md`

**① HGNN→EML 初始化脚本（完整 PyG 代码）**
- 将 HGNN 超边 raw_score 归一化为 ℐ 初值
- 用 HGNN 预训练初始化 EML 图比随机初始化减少 40% 迭代

**② κ-门控传播**
- 推理沿高ℐ超边 hop
- 多幸存则 NDTM fork 保存 Continuation
- 无幸存则回溯/κ微调

---

#### 13. 基元律动/临界初始化（M207 算法）
- **来源**: `unknown_date_基元律动深度剖析与太乙AGI的形式化认知架构：流贯动力学与临界初始化的统合.md`

**M207 临界金灵球初始化算法（含完整 Python 伪代码）**：
- 不从随机噪声起步，而是将网络初始化在"秩序与混沌的边缘"
- 特征值谱服从 Wigner 半圆律，E/I（兴奋/抑制）平衡
- 生成稀疏对称随机边（15%连接率），20%抑制性边，验证谱半径 `rho ≈ 2α`
- **用途**: Grid Perception 模型初始化，以极少样本快速适应

**刘机制变分原理**：
- 流贯选择使"关系作用量"取极小值的路径 `δS=0`
- **用途**: κ-Snap 搜索的优先级判据——优先探索"关系作用量"最小的路径

---

### Library Learning 类

#### 14. 自适应库学习（Adaptive Library Learning）
- **来源**: `unknown_date_TOMAS v2.0：基于自适应库学习的神经符号架构与ARC-AGI-3流体智能.md`

**① 自适应 Sleep-Step 算法（核心可借鉴）**
```python
# 动态预算公式
B = B_base + α * MDL(prog) + β * log2(freq(prog) + 1)
# 对每个已解决程序，按预算提取AST子树
# 统计子树频率freq_counter，按(MDL Gain, freq)排序
# 注册Top-K新原语到库
# α和β根据历史增益反馈在线调整
```
- 高增益时增加 α 以探索更复杂组合
- 高频时增加 β 以固化常用模式
- **集成位置**: `library_learning.py` 中的 sleep_step() 方法

**② AST 宽度控制**
- `W(d) = W_max * exp(-λ*d)`，确保高价值浅层组合子优先保留
- 防止单程序产生过多子树

**③ DSL 基集选择**
- 若 DSL 基集包含 copy, cond, recurse 原语，经自适应库学习收敛至图灵完备闭包

---

#### 15. TOMAS 算子学习中的过拟合免疫性
- **来源**: `unknown_date_TOMAS 算子学习中的过拟合免疫性与天然增量学习机制：基于 EML 超图、κ-.md`
- 过拟合免疫机制和天然增量学习
- **用途**: Library Learning 的防过拟合设计

---

#### 16. 数据结构即算法
- **来源**: `unknown_date_数据结构即算法：基于 EML 超图与 NASGA 拟群的太乙互搏范式.md`
- 将规则库设计为自指涉数据结构，使得对规则的存储、检索、组合操作本身就构成推理过程
- 在 ARC-AGI-3 中，可将已发现的 ARC 模式编码为超图节点，模式间的组合关系为超边
- "遍历超图"就等价于"生成新规则"

---

#### 17. 从混合架构到纯太乙互搏 AGI
- **来源**: `unknown_date_从混合架构到纯太乙互搏 AGI：技术演进路线与终极形态.md`

**① Hybrid Bootstrap 策略**
- 利用 LLM 为 EML 超图标注初始拓扑和 ℐ 初值
- 当密度达阈值后移除 LLM
- **用途**: "用 LLM 做初始模式标注/规则发现，再用符号推理精化"

**② 拓扑生长学习机制**
- `learn方法: evaluate_novelty → add_hyperedge 或 update_I`
- 而非权重调整（无反向传播）
- **用途**: Library Learning 增量更新——发现新模式时新增规则，验证时调整置信度

---

### 系统架构类

#### 18. 太极OS USCS（统一语义-计算状态）
- **来源**: `unknown_date_太极OS_ 统一语义-计算状态的页式管理系统.md`

**① Φ门控（语义一致性检查）**
- 用余弦相似度量化候选输出与当前 World Model 的一致性
- 低于阈值则拒绝 → 新发现的规则必须与已有知识库语义一致

**② 自适应 Φ阈值（CV-based）**
- 根据 Φ 值的变异系数动态调整阈值
- Φ 波动大时降低阈值避免过度拒绝，波动小时严格门控

**③ Continuation（检查点/恢复）**
- 将 Solver 完整状态（规则库快照+搜索历史+当前意图）持久化
- 可恢复 → 直接适用于回溯搜索

**④ δ-mem L1-L2 融合架构**
- 热缓存（最近 N 轮残差记忆）+ 冷存储（Episodic Memory）
- **用途**: Library Learning 的分层存储：常用规则热缓存，长尾规则冷存储

**⑤ 三态自适应衰减（STABLE/DRIFTING/RECOVERY）**
- 不同阶段使用不同遗忘率
- **用途**: Solver 搜索过程中的策略调整

---

#### 19. 智能体五功能最小完备架构
- **来源**: `unknown_date_太一互搏范式下对《智能体五功能最小完备架构》的同构与升维.md`

**五功能架构（I/O/M/G/C）**：
- **Input**: Grid Perception（网格感知）
- **Output**: Solution Generation（解生成）
- **Memory**: Rule Library（规则库，按 ℐ 排序检索）
- **Generation**: Rule Discovery（规则发现，低证据标记 [INFERRED_LOW_I]）
- **Control**: Arbitration（裁决核：Dead-Zero Gate + MUS Arbiter + ℐ-optimal Scheduler）

**G_ego_Core.step() 控制循环伪码**：
```
I_batch → M_state → G_candidates → Dead-Zero过滤 → MUS标记 
→ ℐ-optimal选择 → M更新 → O输出
```

---

#### 20. TOMAS-AGI v2.0 完整框架
- **来源**: `unknown_date_太乙互搏 AGI——基于互搏架构的非结合通用人工智能理论（v2.0）——基于非结.md`

**① 谱悖论耐受定理**
- 允许矛盾规则以双分歧态共存，更多证据出现后再坍缩
- **用途**: 处理"一个游戏有多种可能规则解释"

**② 双环正义架构**
- 认知环：规则发现和验证
- 行为环：游戏交互和副作用管理
- **用途**: Solver 整体架构

**③ TVDE v2.0 视觉谱压缩**
- 直接用于 Grid Perception 模块

---

#### 21. 从提示工程到循环工程
- **来源**: `unknown_date_从提示工程到循环工程：软件工程的谱空间互搏转向——基于太乙互搏（TOMAS）框架.md`

**四重断点设计**：
- **记忆断点**: 保存已发现规则库
- **验证断点**: Lean 式形式化验证规则正确性
- **触发断点**: 游戏状态变化触发规则匹配
- **信任断点**: 规则置信度评估

**生成-验证闭环**：比线性 pipeline 更鲁棒，特别适合规则发现这种需要反复试错的场景

---

#### 22. 多模型潜空间谐振腔
- **来源**: `unknown_date_基于非结合谱图代数的多模型潜空间谐振腔：理论、架构与实现路径——西格玛范式下第二.md`

**SpectralProjectionAdapter + ResonanceCavity**：
- 将不同模型（视觉感知/规则推理/策略生成）的输出投影到统一谱空间
- κ∈[0.45,0.65] 区间内多模型潜空间发生建设性干涉
- **用途**: 多模型集成决策——输出在谱空间发生建设性干涉时增强规则置信度

---

## 三、跨批次高频核心概念（出现≥3次）

| 概念 | 出现批次 | 对 Solver 的具体映射 |
|------|---------|---------------------|
| **MUS 双存机制** | 批次2/4/6 + 第二波1/6 | 多候选规则并行保留，不过早收敛 |
| **Dead-Zero 门控** | 批次2/4/6 + 第二波1 | 低置信度规则剪枝/拒绝写入 Library |
| **ℐ-加权排序** | 批次2/4/6 + 第二波1/6 | 规则按 grounding 强度排序检索 |
| **EML-SemZip 五阶段** | 批次2/6 | 模式发现→同构匹配→压缩→核心保持→最小表示 |
| **κ-Snap 增量更新** | 第二波6/批次2 | 只重搜受影响分支，非全量重算 |
| **双分量罚项(D_SR+I_ext)** | 批次2 | 搜索评估函数：自洽性+自含性 |
| **Core/Refinement 双路径** | 批次4 | 快速基础匹配+深度精细化分层搜索 |
| **M207 临界初始化** | 第二波6 | Grid Perception 模型在秩序-混沌边缘初始化 |
| **Φ门控语义一致性** | 批次4 | 新规则与已有知识库的一致性验证 |
| **M 码算子集** | 批次2 | 完整算法原语目录(收敛/同步/传播/歧义) |
| **拓扑生长学习** | 批次5 + 第二波6 | Library Learning 增量更新：新增规则而非调参 |
| **κ-调节器** | 第二波6 | 搜索深度自适应：LOCKED→UNLOCK→SWEEP→CONVERGE |

---

## 四、按集成优先级排序的行动建议

### P0 — 立即可集成（1-2天）

| # | 借鉴点 | 来源文件 | 集成位置 | 预期收益 |
|---|--------|---------|---------|---------|
| 1 | L3 差分注意力玩家检测 | Taiyi Mutual-Duel Agent | `grid_perception.py` | 解决墙体淹没问题 |
| 2 | 帧预过滤（差分阈值跳过静态帧） | TOMAS v2.0 框架总览 | `planner_agent.py` | 减少30%冗余计算 |
| 3 | 贝叶斯 RHAE 熔断（效率<0.5回溯） | Taiyi Mutual-Duel Agent | `planner_agent.py` | RHAE 评分优化 |
| 4 | DFS 回溯规划器 | Taiyi Mutual-Duel Agent | `planner_agent.py` | 状态空间搜索+回溯 |

### P1 — 近期集成（3-5天）

| # | 借鉴点 | 来源文件 | 集成位置 | 预期收益 |
|---|--------|---------|---------|---------|
| 5 | 自适应 Sleep-Step 动态预算 | 自适应库学习 | `library_learning.py` | 提升库学习效率 |
| 6 | AST 宽度控制 W(d)=W_max·exp(-λ·d) | 自适应库学习 | `library_learning.py` | 子树提取剪枝 |
| 7 | 拟阵贪心剪枝 | 数学降维 | `kappa_snap_searcher.py` | 候选模式筛选 |
| 8 | 条件ΔT自动发现 | TOMAS v2.0 框架 | `self_learning.py` | 游戏规则推断 |
| 9 | MUS 双存（Top-K候选保留） | EML-EHNN / 多个文件 | `planner_agent.py` | 多假设并行 |
| 10 | Dead-Zero 门控（低ℐ拒绝） | 多个文件 | `library_learning.py` | 规则准入控制 |

### P2 — 中期集成（1-2周）

| # | 借鉴点 | 来源文件 | 集成位置 | 预期收益 |
|---|--------|---------|---------|---------|
| 11 | Core/Refinement 双路径搜索 | AnyDepth 弹性深度 | 搜索架构 | 分层搜索效率 |
| 12 | κ-Snap 增量更新 | 刘机制/NAU | `kappa_snap_searcher.py` | 减少重复计算 |
| 13 | 双分量罚项评估函数 | TOMAS 互搏泛函 | 搜索评估 | 解质量分级 |
| 14 | GPCT 边界层降解 | 数学降维 | 搜索策略 | 关键区域优先 |
| 15 | Φ门控语义一致性 | 太极OS USCS | 规则验证 | 规则库一致性 |
| 16 | κ-调节器状态机 | 主观/能动数学化 | 搜索参数管理 | 深度自适应 |
| 17 | EML-SemZip 五阶段流水线 | EML-SemZip | `library_learning.py` | 规则压缩+复用 |
| 18 | GaussEx 跨游戏迁移 | TOMAS v2.0 框架 | 跨游戏复用 | 零样本迁移 |

### P3 — 架构级重构（长期）

| # | 借鉴点 | 来源文件 | 集成位置 | 预期收益 |
|---|--------|---------|---------|---------|
| 19 | 五功能架构(I/O/M/G/C) | 智能体五功能 | 整体架构 | 模块清晰化 |
| 20 | 双环正义架构 | TOMAS-AGI v2.0 | 整体架构 | 认知-行为分离 |
| 21 | 四重断点循环工程 | 循环工程 | 整体架构 | 生成-验证闭环 |
| 22 | M207 临界初始化 | 基元律动 | Grid Perception | 少样本快速适应 |

---

## 五、⭐⭐ 有启发的文件（按主题归类，共~30个）

### 搜索策略启发
- **刘机制分布式调度**（6G内生AI）: gossip-based 并行搜索，每个网格区域独立搜索后交换信息
- **κ-Snap事件驱动调度**（存储墙重译）: 只在积累足够信息势时才触发深度搜索
- **ITC虚时退火**（数学降维）: 模拟退火逃局部最优
- **经验隧穿**（星球级智慧潮汐）: 远距离模式在谱空间"准简并共振"时直接迁移规则
- **κ-门控传播**（超图神经网络→EML）: 沿高ℐ超边hop+fork+回溯

### 模式识别启发
- **ℐ-加权相似度**（MBSE本体论）: 低ℐ对象即使向量距离近，相似度也被压制
- **k-hop推理**（MBSE本体论）: 邻接矩阵幂运算实现多跳关系推理，发现多步变换规则
- **多尺度分析**（八元数统一基础）: 不同分辨率下网格模式不同，需多尺度分析
- **Penrose投影**（准晶拓扑序）: 高维网格模式投影到低维特征空间匹配

### Library Learning 启发
- **拒写入机制**（死零Dead-Zero）: 不仅拒绝输出，还拒绝将无据关系写入知识库
- **超图全局约束检查**（死零Dead-Zero）: 新关系分支写入前检查全图一致性
- **MUS互斥检查**（FDE本体构建器）: 防止Library中矛盾规则共存
- **Tier分级策略**（信息最优化视角）: Tier1(已证)→Tier2(开放)→Tier3(启发式)标记

### 验证启发
- **公理可导出性测试**（T-Shield）: 验证新发现规则能否从已有公理推导
- **Self-Consistency Loop**（太极OS）: 两层管道——语义矛盾检测+Φ门控
- **CKA特征对齐验证**（AnyDepth）: 验证不同分辨率下发现的规则一致性
- **莫比乌斯验证环**（认识论倒置）: 用规则解释训练样例，若自洽则接受

---

## 六、关键 M 码算子集速查表

来自文件30（TOMAS完备性终章），这些算子原语可直接映射为 Solver 操作：

| M码 | 名称 | 功能 | Solver 映射 |
|-----|------|------|------------|
| M106 | Φ-收敛 | 检测搜索是否达 ArgMin S_Rel | 搜索收敛检测 |
| M244 | Kuramoto 同步 | EML节点相位同步 | 模式对齐检测 |
| M245 | PG几何原型引擎 | Oloid/钢板网/三角钻头等 | Grid 变换模板 |
| M247 | CRD 不动点 | Banach 不动点求解 | 迭代应用规则直到稳定 |
| M253 | Epiplexity Ξ | ℐ-流梯度度量 | 多分辨率分析决策 |
| M256 | NASGA 传播 | 沿超边离散演化 | 图搜索传播算法 |
| M259 | 双环判断 | Asym≠0 ∧ ℐ≈ ⇒ MUS | 歧义检测 |
| M88 | 一致性 Hash | 多视角投影须得相同摘要 | 多视角一致性验证 |
| M207 | 临界初始化 | Wigner半圆谱+E/I平衡 | 模型初始化 |

---

## 七、结论

从 2011 个 TOMAS 理论文档中提取的 22 个高度相关可借鉴思想，覆盖了 ARC-AGI-3 Solver 的全部核心模块：

1. **搜索策略**（8个）: κ-Snap溯因搜索、DFS回溯、拟阵剪枝、NAU选择算子、双分量罚项、κ-调节器、κ-死锁检测、增量更新
2. **模式识别**（5个）: 差分注意力检测、EML-SemZip压缩、弹性深度感知、等变超图MUS双存、M207临界初始化
3. **Library Learning**（4个）: 自适应Sleep-Step、过拟合免疫、数据结构即算法、拓扑生长学习
4. **系统架构**（3个）: USCS页式管理、五功能架构、双环正义架构
5. **验证评分**（2个）: 贝叶斯RHAE熔断、Dead-Zero门控

**最高优先级的 4 个立即可集成项**（P0）均来自 `Taiyi Mutual-Duel Agent` 文件，该文件是全部 2011 个文档中与 ARC-AGI-3 交互式游戏 Solver **最直接相关**的文件，提供了可直接落地的代码级算法。
