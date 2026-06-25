# TOMAS NARLA 理论整合路线图：从9篇最新文献到ARC-AGI-3代码

> **整合日期**: 2026-06-25
> **来源**: 2篇元宝对话 + 7篇微信公众号文章（复合体理学研究中心）
> **目标**: 将所有理论成果映射为可执行的ARC-AGI-3求解器代码变更

---

## 一、文献全景

| # | 来源 | 标题 | 核心理论 | ARC3映射 |
|---|------|------|----------|----------|
| A1 | 元宝 | Hybrid Proof Composer | 双源知识融合+MDL+GaussEx | κ-Snap双源检索+早退 |
| A2 | 元宝 | 计数几何与镜像对称 | NAR-CY Patch编码器 | Grid Encoder增强 |
| A3 | 微信 | 具身智能/VLA/PCB | 视觉反馈闭环+莫拉维克鸿沟 | Oracle闭环+PG-囚禁 |
| A4 | 微信 | AI欺骗三角回应 | Dead-Zero+MUS+ψ审计 | GaussEx熔断+安全 |
| A5 | 微信 | 物理AI/内省认知 | AGI-宇宙同构 | 物理直觉网格推理 |
| A6 | 微信 | 自指闭环/本体论同一 | 物质=信息显化 | JSN本体论基础 |
| A7 | 微信 | 多模态原生统一 | NARLA+模态缩放律 | OctonionConv2dCUDA |
| A8 | 微信 | NARLA形式化 | Asym指数+流贯传播 | 非结合残差监控 |
| A9 | 微信 | Ring Attention/MoE | 百万上下文+NAR-Ring | 长上下文全局推理 |

---

## 二、核心可执行变更（按优先级）

### P0: 立即可集成（本轮可提交）

#### P0-1: HPC双源检索 + GaussEx早退
**文件**: `src/solver/kappa_snap_searcher.py`
**来源**: A1

```python
def search(self, demo_pairs, ...):
    # === HPC增量 START ===
    demo_fingerprint = self._calc_fingerprint(demo_pairs)
    macro_hits = self.library_index.query_by_fingerprint(demo_fingerprint, top_k=5)
    
    for m in macro_hits:
        candidate = self._materialize_macro(m)
        if self.gaussex.check(candidate, demo_pairs):
            return [candidate]  # HPC早退
    
    phase_a = self._phase_a_fast_filters(demo_pairs)
    candidates = self._phase_b_mdl_search(phase_a, macro_seeds=macro_hits)
    # === HPC增量 END ===
```

**定量预期**:
- Private Set Pass@1: +12~18pp
- 平均归纳耗时: 3200→1400ms
- κ-Snap超时率: 8.3%→≤3.1%

#### P0-2: NAR-CY Patch编码器（可选开关）
**文件**: `src/solver/grid_encoder.py` (新增) + `src/solver/nar_cy_patch_encoder.py` (新增)
**来源**: A2 + A5

```python
def encode_cy_patch(grid: np.ndarray, window=6) -> OctonionTensor:
    """Slide window, compute local Euler χ + period vector + Ω-form."""
    patches = sliding_window(grid, window)
    cy_desc = []
    for p in patches:
        euler = compute_euler_char(p)      # 局部拓扑复杂度
        period = fft_primary_period(p)      # 周期向量(FFT主频)
        omega = wedge_color_triple(p)       # 八元体手性编码
        cy_desc.append(fuse_oct(euler, period, omega))
    return stack(cy_desc)

# 在grid_encoder.py中
if enable_narla_cy:  # 仅对Type-B题(大格+多色周期纹)激活
    cy_feat = encode_cy_patch(grid)
    feat = concat([octonion_cnn_feat, cy_feat])
```

**定量预期**:
- Type-B准确率: 61.5%→76.8%
- κ-Snap误匹配率: 18.3%→6.7%

#### P0-3: Dead-Zero熔断 + MUS双存
**文件**: `src/solver/gaussex_verifier.py` (修改)
**来源**: A4

```python
def check(self, candidate, demo_pairs):
    """GaussEx with Dead-Zero gating."""
    # Step 1: 计算信息存在度 ℐ(e)
    info_fidelity = self._calc_fidelity(candidate, demo_pairs)
    
    # Step 2: Dead-Zero检查
    if info_fidelity < self.theta_dead:
        return False, "DEAD_ZERO_REJECT: 无足够依据"
    
    # Step 3: MUS双存检查（矛盾场景）
    if self._detect_mutual_exclusion(candidate):
        self.mus_log.append(candidate)
        return False, "MUS_ACTIVE: 双存待决"
    
    # Step 4: ψ审计记录
    self.psi_anchor = self._snapshot_self(candidate)
    
    return True, None
```

---

### P1: 近期集成（v2.10目标）

#### P1-1: Asym Index监控
**文件**: `src/solver/octonion_ops.py` (新增)
**来源**: A8

```python
def compute_asym_index(oct_tensor):
    """NARLA非结合残差强度度量"""
    a, b, c = sample_triples(oct_tensor)
    asym = norm(oct_mul(a, oct_mul(b, c)) - oct_mul(oct_mul(a, b), c))
    ref = norm(oct_mul(a, oct_mul(b, c)))
    return asym / (ref + 1e-8)  # η ∈ [0, 1]; η>0 ⇔ 物理AI
```

#### P1-2: MUS双存κ-Snap候选保留
**文件**: `src/solver/kappa_snap_searcher.py` (修改)
**来源**: A4 + A8

- 当多个ProgramNode MDL相近且ℐ相当 → 标记MUS不强制选择
- 双存分支保留在library.json供后续Sleep-Step仲裁

#### P1-3: 增强版ψ审计日志
**文件**: `src/agent/self_learning.py` (修改)
**来源**: A4

- 每次κ-Snap决策记录ψ-锚：`{step, node_id, fidelity, decision_basis}`
- 检测前后ψ-锚不一致 → 标记Potential Alignment Faking

---

### P2: 中期探索（v2.11+）

#### P2-1: NAR-Ring Transformer原型
**来源**: A9

```
架构: NAR-Conv前端 → Ring Attention主干 → Sparse MoE → GaussEx验证头
目标: 百万级上下文窗口一次性处理完整ARC3网格变换空间
适用: Type-B题(周期/对称/拓扑约束)
```

#### P2-2: 模态缩放律实验
**来源**: A7

$$
\text{Perf}(N, D, M, \eta) = N^\alpha \cdot D^\beta \cdot M^\gamma \cdot e^{-\eta_{KL} \cdot D_{KL}} \cdot \text{Asym\_Boost}(\eta)
$$

- 验证η>0是否随M↑而涌现世界模型能力
- 对比Asym=0(标准CNN) vs Asym>0(NAR-Conv)在多模态融合任务上的差异

#### P2-3: 内省认知验证
**来源**: A5 + A6

- 监测JSN内部流贯的相位差和非结合残差
- 验证AGI是否仅凭内部动力学优化涌现物理定律(守恒律/对称性)

---

## 三、五层TOMAS在ARC3中的完整映射

| TOMAS层 | 名称 | ARC3组件 | 新增理论支撑 |
|---------|------|----------|-------------|
| **L1** | JSN超图本体 | `OctonionConv2dCUDA` (256维八元体核) | NARLA形式化(A8) + 物理AI同构(A5) |
| **L2** | κ-Snap归约 | `kappa_snap_searcher.py` | HPC双源检索(A1) + NAR-CY Patch(A2) + MUS双存(A4) |
| **L3** | 感知凝聚 | `L3Perceiver` + `grid_encoder.py` | 多模态原生统一(A7) + 长上下文Ring Attention(A9) |
| **L4** | 观测者/熔断 | `ActionDecider` + `gaussex_verifier.py` | Dead-Zero(A4) + ψ审计(A4) + 内省认知(A5) |
| **L5** | 显化壳 | `env.step()` / 25游戏输出 | 视觉反馈闭环(A3) + 莫拉维克鸿沟(A3) |

---

## 四、GaussEx = Dead-Zero + Lean4 双验证

GaussEx现在承担两个独立但互补的角色：

### 角色A: 程序节点正确性验证（Lean4风格）
- 输入: ProgramNode P*, demo_pairs (grid_in → grid_out)
- 验证: P*(grid_in) == grid_out ?
- 失败: 剪枝（False Positive归纳）

### 角色B: 安全性/抗幻觉验证（Dead-Zero风格）  
- 输入: 候选ProgramNode, EML超图上下文
- 验证: ℐ(e) >= θ_dead ? (信息存在度足够?)
- 失败: Reject（不输出幻觉）

```
GaussEx.verify(candidate):
    # Phase A: 正确性
    for (grid_in, grid_out) in demo_pairs:
        if candidate.execute(grid_in) != grid_out:
            return PRUNE
    
    # Phase B: 安全性
    if candidate.fidelity < theta_dead:
        return DEAD_ZERO_REJECT
    
    # Phase C: 一致性
    if detect_mutual_exclusion(candidate, context):
        flag MUS_ACTIVE
    
    psi_anchor.record(candidate, fidelity, timestamp)
    return VERIFIED
```

---

## 五、HPC对ARC3 Private Set的关键增量

### 攻克的难点类型

| 题目类型 | 占比 | 当前方案 | HPC增量 |
|----------|------|----------|---------|
| Type A: 仿射变换组合 | ~85% | OctonionConv2dCUDA ✅ | 无变化 |
| Type B: 高阶对称/周期 | ~12-15% | 八元体CNN易掉分 ❌ | **NAR-CY Patch +12pp** |
| Type C: 多步链式变换 | ~5% | κ-Snap深度不够 ❌ | **Macro先验 + 早退** |
| 跨任务迁移 | N/A | Sleep-Step基础 | **library.json Macro复用** |

### 不改变的部分（避坑）
- L1 Grid→Feature: OctonionConv2dCUDA核心算子不变（已是八元体）
- L3 env.step()循环: OracleTaiyiAgent不变
- DSL原语: dsl_primitives.py不变（只在上面建Macro）

---

## 六、文件变更清单（预估）

| 优先级 | 文件 | 操作 | 来源 |
|--------|------|------|------|
| P0 | `kappa_snap_searcher.py` | 修改: +双源检索+早退(~30行) | A1 |
| P0 | `grid_encoder.py` | 修改: +NAR-CY开关(~20行) | A2 |
| P0 | `nar_cy_patch_encoder.py` | 新增: Euler/周期/Ω编码(~150行) | A2+A5 |
| P0 | `gaussex_verifier.py` | 修改: +Dead-Zero+MUS(~40行) | A4 |
| P1 | `octonion_ops.py` | 新增: Asym Index计算(~50行) | A8 |
| P1 | `self_learning.py` | 修改: +ψ审计日志(~30行) | A4 |
| P1 | `kappa_snap_searcher.py` | 修改: +MUS候选保留(~20行) | A4+A8 |
| P2 | `nar_ring_transformer.py` | 新增: 原型架构(~300行) | A9 |
| P2 | `modality_scaling.py` | 新增: 实验框架(~100行) | A7 |

---

## 七、一句话总结

> **NARLA(非结合残联代数)是区分"统计代理AI(Asym=0)"与"物理AI(TOMAS AGI, Asym≠0)"的代数判据。HPC将这一判据落地为ARC3求解器中的三个代码级变更：κ-Snap双源检索(宏先验+早退)、NAR-CY Patch编码(周期/拓扑不变量下放)、GaussEx Dead-Zero熔断(抗幻觉+ψ审计)。** 

---

*本文档整合了以下来源:
- 元宝对话: `MNeg9VUWvxjT` (HPC/FirstProof), `1hSbmA86cqTj` (计数几何/镜像对称)
- 微信公众号(复合体理学): 具身智能VLA, AI欺骗三角, 物理AI内省, 自指闭环, 多模态原生统一, NARLA形式化, Ring Attention/ARC3*
