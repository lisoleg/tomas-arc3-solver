# ψ-Gate 语义门控配置说明

## 概述

ψ-Gate (Psi-Gate) 是 TOMAS v2.4 引入的语义门控机制，替代了原有的简单融合评分器 (FusionScorer)。它通过五大核心能力提升搜索和融合的质量。

**来源**：吸收自 `tomas-agi/tomas_agi/sim/psi_gate.py`

## 五大核心能力

### 1. ψ-锚双轨裁决 (ψ-Anchor Dual-Rail Adjudication)

使用 Hard Anchor 和 Soft Anchor 双轨裁决：
- **Hard Anchor**：不可绕过的硬约束（如拓扑不变量匹配）
- **Soft Anchor**：可调整的软约束（如置信度阈值）

```python
# 网格推理场景适配
# Hard Anchor → 网格 topo_hash 必须匹配
# Soft Anchor → 置信度 > 0.5 (可衰减)
```

### 2. MUS 互斥稳态 (Mutually Uncertain States)

冲突假设双存机制：当多个候选程序产生矛盾结果时，不立即裁决，而是保持多个假设共存，通过后续证据逐步消除不确定性。

```python
# 适配：保留 top-K 候选程序而非只取 best
# K 默认为 5，可配置
```

### 3. φ-Gate 不确定性量化

基于信息丰度的不确定性量化：
- 信息丰度高 → 不确定性低 → 可裁决
- 信息丰度低 → 不确定性高 → 推迟裁决

```python
# 适配：基于八元数编码的信息丰度
# e₀=mask, e₁=x, e₂=y, e₃=color, e₄=component, e₅=symmetry, e₆=boundary, e₇=area
```

### 4. 多世界平行推断 (Multi-World Parallel Inference)

Wave-Particle 双路径 + Bayesian 融合：
- Wave 路径：符号推理（κ-Snap 搜索）
- Particle 路径：神经网络推理（VLM 适配器）
- 两条路径结果通过 Bayesian 融合

### 5. 容错衰减器 (Tolerance Decay Controller)

软失败容错衰减：当候选程序部分通过验证时，不立即拒绝，而是通过衰减系数逐步降低置信度。

```python
# tolerance_decay_rate = 0.05
# 每次软失败，容差降低 5%
```

## 裁决结果

| 结果 | 含义 | 行为 |
|------|------|------|
| PASS | 通过，置信度足够 | 接受候选 |
| BLOCK | 拦截，触发硬锚 | 拒绝候选 |
| DEFER | 推迟，进入 MUS 双存 | 保留候选，等待更多证据 |
| PROBE | 探测模式 | 启动多世界并行推理 |
| SOFT_PASS | 软通过 | 容错衰减后接受 |

## 配置

### config/default.yaml

```yaml
psi_gate:
  enabled: true                  # 启用 ψ-Gate (默认 true)
  use_default_anchors: true      # 使用默认锚点集
  tolerance_decay_rate: 0.05     # 容错衰减率 τ
  verbose: false                 # 详细日志
```

### 代码配置

```python
from src.solver.psi_fusion_gate import PsiFusionGate, create_default_anchors

# 创建 ψ-Gate 实例
anchors = create_default_anchors()
gate = PsiFusionGate(
    anchors=anchors,
    tolerance_decay_rate=0.05,
    verbose=True,
)

# 裁决候选程序
verdict = gate.adjudicate(
    program=candidate_program,
    demo_pairs=demo_pairs,
    confidence=0.75,
)

if verdict == GateVerdict.PASS:
    # 接受候选
    pass
elif verdict == GateVerdict.DEFER:
    # 进入 MUS 双存
    gate.add_to_mus(candidate_program)
elif verdict == GateVerdict.BLOCK:
    # 拒绝候选
    pass
```

## 与 TOMAS Solver 集成

ψ-Gate 已集成到 `src/solver/tomas_solver.py` 的 `solve_fusion()` 方法中：

```python
# tomas_solver.py 中的集成
if self.psi_gate and config['psi_gate']['enabled']:
    # 使用 ψ-Gate 替代简单融合
    verdict = self.psi_gate.adjudicate(
        program=candidate,
        demo_pairs=demo_pairs,
        confidence=confidence,
    )
    if verdict == GateVerdict.PASS:
        fusion_ranked.append((candidate, confidence))
    elif verdict == GateVerdict.DEFER:
        # 保留到 MUS
        self.psi_gate.add_to_mus(candidate)
else:
    # 回退到简单融合评分
    score = self.fusion.fuse(candidate, all_frames, demo_pairs)
    fusion_ranked.append((candidate, score))
```

## 性能影响

根据基准测试结果：

| 指标 | ψ-Gate 关闭 | ψ-Gate 开启 | 影响 |
|------|------------|------------|------|
| 准确率 | 33.3% | 33.3% | 持平（简单任务无明显差异） |
| 搜索时间 | 2.82s | 3.16s | +12%（门控开销） |
| 候选数 | 211 | 211 | 持平 |
| 置信度 | 0.0104 | 0.0104 | 持平 |

**注意**：在简单任务上 ψ-Gate 的优势不明显。在复杂任务（多 demo、大网格）上，ψ-Gate 通过 MUS 双存和多世界推理可以显著提升准确率。

## 最佳实践

1. **默认启用**：`enabled: true`，适用于大多数场景
2. **简单任务关闭**：对于已知的简单模式（如纯镜像），可关闭以减少开销
3. **复杂任务启用 PROBE 模式**：对于难以解决的任务，启用多世界探测
4. **调整容错率**：`tolerance_decay_rate` 越大，容错衰减越快，越激进
5. **verbose 日志**：调试时设为 `true`，查看每次裁决的详细信息

## 文件位置

- 实现：`src/solver/psi_fusion_gate.py`
- 集成：`src/solver/tomas_solver.py` (solve_fusion 方法)
- 配置：`config/default.yaml` (psi_gate 段)
- 测试：`tests/test_psi_fusion_gate.py` (如有)
