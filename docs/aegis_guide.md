# AEGIS 演进引擎文档

## 概述

AEGIS (ExtendHypergraph on Config Space) 是 TOMAS v2.4 引入的程序演进搜索引擎，将 κ-Snap 从"单次搜索"升级为"进化搜索"。

**来源**：吸收自 `tomas-agi/tomas_agi/sim/harness_aegis.py`

## 核心理念

传统 κ-Snap 搜索是一次性的：生成候选 → 剪枝 → 验证 → 结束。AEGIS 在此基础上添加了**进化循环**：从搜索结果中学习，规划下一批候选，变异优质程序，再次验证。

```
传统搜索:  生成 → 剪枝 → 验证 → 结束
AEGIS搜索: 生成 → 剪枝 → 验证 → 消化 → 规划 → 变异 → 验证 → ... (多代进化)
```

## 四阶段流水线

### 1. Digester（消化器）

消化候选程序，提取特征向量：

```python
# 输入：κ-Snap 搜索结果的候选程序列表
# 输出：特征聚类

features = digester.digest(candidates)
# 每个候选提取:
#   - DSL 原语组合 (e.g., [resize, mirror_h])
#   - MDL 成本
#   - topo_hash 签名
#   - 置信度
#   - 验证通过率

# 聚类：相似特征的程序归为同一簇
clusters = digester.cluster(features, k=5)
```

### 2. Planner（规划器）

基于特征聚类规划下一批候选：

```python
# 输入：聚类结果 + 验证统计
# 输出：下一代变异计划

plan = planner.plan(clusters, verification_stats)
# 规划内容:
#   - 哪些簇有潜力 → 增加变异
#   - 哪些簇已穷尽 → 减少资源
#   - 哪些 DSL 组合未探索 → 新探索方向
```

### 3. Evolver（演化器）

基于规划变异 top-K 程序：

```python
# 变异操作
mutated = evolver.evolve(top_k_programs, plan)

# 变异类型:
#   1. 参数变异: 修改 DSL 原语参数 (e.g., rotate 90 → rotate 180)
#   2. 结构变异: 替换/插入/删除 DSL 原语
#   3. 组合变异: 合并两个程序的部分
#   4. 交叉: 两个程序的交叉重组
```

### 4. Critic + Gate（批判 + 门控）

验证变异后的程序：

```python
# 使用 GaussEx 验证 + ψ-Gate 门控
for mutated_program in mutated:
    # GaussEx 纤维验证
    verified = gaussex.verify(mutated_program, demo_pairs)
    
    if verified:
        # ψ-Gate 门控裁决
        verdict = psi_gate.adjudicate(mutated_program, demo_pairs)
        
        if verdict == GateVerdict.PASS:
            # 保留到下一代
            next_generation.append(mutated_program)
        elif verdict == GateVerdict.DEFER:
            # MUS 双存
            psi_gate.add_to_mus(mutated_program)
```

## MUS 变体隔离

AEGIS 维护 K≤5 个变体簇（MUS = Mutually Uncertain States）：

```python
# 当多个变体产生不同但都部分有效的结果时
# 不立即裁决，而是保持共存
# CRR (Conflict Resolution Rate) > 95% 时触发裁决

mus = MUSContainer(k=5)
mus.add(variant_1)  # 置信度 0.7
mus.add(variant_2)  # 置信度 0.65
mus.add(variant_3)  # 置信度 0.6

# 当新证据到来时，逐步消除不确定性
if mus.crr() > 0.95:
    best = mus.resolve()  # Bayesian 融合裁决
```

## κ-Gate 双轨

AEGIS 使用 κ-Gate 进行双轨协同进化：

```
harness_ver (执行验证) ←→ model_weight (模型权重)
       ↑                          ↑
       └── 协同进化 ──────────────┘
```

- `harness_ver`：验证执行路径的可靠性
- `model_weight`：模型对候选程序的置信权重
- 两者协同进化，互相校准

## 配置

### config/default.yaml

```yaml
aegis:
  enabled: false                  # 启用 AEGIS (默认 false，按需开启)
  max_generations: 3             # 最大进化代数
  population_size: 20            # 每代种群大小
  mutation_rate: 0.15            # 变异率
  crossover_rate: 0.1            # 交叉率
  top_k: 5                      # 每代保留 top-K
  mus_k: 5                      # MUS 变体簇大小
  crr_threshold: 0.95           # 冲突解决率阈值
  verbose: false                # 详细日志
```

### 代码使用

```python
from src.solver.aegis_evolver import AEGISEvolver

# 创建 AEGIS 演化器
aegis = AEGISEvolver(config['aegis'])

# 进化搜索
result = aegis.evolve_search(
    task_data=task,
    searcher=solver.searcher,  # κ-Snap 搜索器
    max_generations=3,
)

# result 包含:
#   - best_program: 最优程序
#   - evolution_history: 每代的候选和验证结果
#   - mus_state: MUS 最终状态
#   - statistics: 进化统计
```

## 与 Kaggle 集成

在 Kaggle 提交模板中，AEGIS 用于重试失败任务：

```python
# Kaggle notebook 中的使用
if config['aegis']['enabled'] and failed_tasks:
    aegis = AEGISEvolver(config['aegis'])
    
    for task_id in failed_tasks:
        result = aegis.evolve_search(
            task_data=task,
            searcher=solver.searcher,
            max_generations=3,
        )
        if result:
            submission[task_id] = result
```

## 性能影响

根据基准测试结果：

| 指标 | 普通搜索 | AEGIS 搜索 | 影响 |
|------|---------|-----------|------|
| 准确率 | 33.3% | 33.3% | 持平（简单任务） |
| 搜索时间 | 4.32s | 5.26s | +22%（进化开销） |
| 候选数 | 211 | 211 | 持平（首轮搜索相同） |

**注意**：
- AEGIS 的优势在**复杂任务**上体现：通过多代进化可以发现单次搜索遗漏的程序
- 简单任务不建议开启（增加不必要的开销）
- 建议 `max_generations: 3`，超过 5 代理边际收益递减

## 进化统计

AEGIS 运行后提供以下统计：

```python
stats = aegis.get_statistics()
# {
#   "generations": 3,
#   "total_candidates_evolved": 45,
#   "total_verified": 12,
#   "best_generation": 2,         # 最优程序出现在第2代
#   "improvement": 0.15,          # 相比第1代的提升
#   "mus_resolved": 3,            # MUS 解决的冲突数
#   "crr_final": 0.96,            # 最终冲突解决率
# }
```

## 最佳实践

1. **按需开启**：`enabled: false` 默认关闭，仅对困难任务开启
2. **控制代数**：`max_generations: 3` 足够，避免过度进化
3. **调整变异率**：`mutation_rate: 0.15` 平衡探索和利用
4. **与 ψ-Gate 配合**：AEGIS + ψ-Gate 组合效果最佳
5. **监控 MUS**：如果 MUS 长期不解决，说明任务过于模糊

## 文件位置

- 实现：`src/solver/aegis_evolver.py`
- 集成：`kaggle/notebook_template.ipynb` (Section 6)
- 配置：`config/default.yaml` (aegis 段)
- 来源：`tomas-agi/tomas_agi/sim/harness_aegis.py`
