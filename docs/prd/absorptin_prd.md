# TOMAS ARC-AGI-3 Solver v2.4 — 跨仓库吸收 PRD

**日期**: 2026-06-23
**来源仓库**: `tomas-agi` (master), `mnq-golden-spirit-ball-simulator`
**吸收目标**: tomas-arc3-solver (v2.3 → v2.4)

---

## 1. 产品目标

1. **接入官方 RHAE 评估框架**，使 TOMAS 可用官方 ARC-AGI-3 metric 评估
2. **升级融合决策**：用 ψ-Gate 语义门控替代简化 fusion，提升搜索质量
3. **升级搜索策略**：用 AEGIS 进化引擎升级 κ-Snap 为进化搜索
4. **引入因果先验**：从历史人物学习 DSL 因果依赖，引导剪枝顺序
5. **统一数学工具**：消除 Luzhao DNA 重复代码，提供统一 Fibonacci/Lucas/Bagua 接口
6. **提供 Pure Python GAT 公理**：作为 GATLab 不可用时 fallback
7. **（待完成）Web 可视化 Dashboard**：实时查看求解过程

---

## 2. 用户故事

### 第一轮（P0 — 立即吸收）

- 作为研究者，我希望用官方 RHAE 评分评估 TOMAS，以便与 frontier AI baseline 对比
- 作为研究者，我希望用 ψ-Gate 语义门控替代当前简化 fusion，以提升搜索质量

### 第二轮（P1 — 近期吸收）

- 作为研究者，我希望 κ-Snap 支持进化搜索（AEGIS），以在困难任务上获得更优程序
- 作为开发者，我希望有 Web Dashboard 实时查看求解过程（搜索树、纤维验证、剪枝率）

### 第三轮（P2 — 中期吸收）

- 作为研究者，我希望 DSL 先验能引导搜索顺序，以减少无效搜索
- 作为开发者，我希望 GAT 公理有 Pure Python 实现，以便在没有 Julia 运行时也能做形式化验证
- 作为开发者，我希望 Luzhao DNA 数学工具是统一的，以避免两处维护重复代码

---

## 3. 需求池

| 优先级 | 组件 | 来源文件 | 目标文件 | 状态 |
|--------|------|----------|----------|------|
| P0 | RHAE 评估框架 | `tomas-agi: arc_agi3_eval.py` | `src/eval/arc_agi3_evaluator.py` | ✅ 完成 |
| P0 | 数据集构建器 | `tomas-agi: arc_agi3_dataset_builder.py` | `src/eval/dataset_builder.py` | ✅ 完成 |
| P0 | ψ-Gate 语义门控 | `tomas-agi: psi_gate.py` | `src/solver/psi_fusion_gate.py` | ✅ 完成 |
| P1 | AEGIS 演进引擎 | `tomas-agi: harness_aegis.py` | `src/solver/aegis_evolver.py` | ✅ 完成 |
| P1 | Web Dashboard | `mnq: backend/ + frontend/` | `src/web/` | ⬜ 待完成 |
| P2 | GAT 公理 (Pure Python) | `tomas-agi: gat_axioms.py` | `src/core/gat_axioms.py` | ✅ 完成 |
| P2 | 因果 DSL 先验 | `tomas-agi: causal_world_model_tomas.py` | `src/solver/causal_dsl_prior.py` | ✅ 完成 |
| P2 | 统一数学工具 | 两仓库 Luzhao DNA | `src/core/math_sequences.py` | ✅ 完成 |
| P2 | 集成更新 | — | `src/solver/tomas_solver.py`, `config/default.yaml` | ✅ 完成 |

---

## 4. 吸收改动摘要

### 4.1 新建文件（7 个）

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/eval/__init__.py` | ~20 | 评估包初始化 |
| `src/eval/arc_agi3_evaluator.py` | ~350 | RHAE 评分器 + TOMAS 评估器 |
| `src/eval/dataset_builder.py` | ~180 | ARC 任务 → 数据集 JSON |
| `src/solver/psi_fusion_gate.py` | ~420 | ψ-Gate 五大核心能力 |
| `src/solver/aegis_evolver.py` | ~380 | AEGIS 四阶段进化流水线 |
| `src/solver/causal_dsl_prior.py` | ~350 | 因果图学习 + do-calculus |
| `src/core/gat_axioms.py` | ~280 | GATTheory + ArcDSL_GAT |
| `src/core/math_sequences.py` | ~220 | Fibonacci/Lucas/Bagua 统一工具 |

### 4.2 修改文件（2 个）

| 文件 | 改动 |
|------|------|
| `src/solver/tomas_solver.py` | +PsiGate 可选导入 + `solve_fusion()` 集成 ψ-Gate |
| `config/default.yaml` | +psi_gate / aegis / causal_prior / eval / math_sequences 五段配置 |

### 4.3 配置项（新增）

```yaml
psi_gate:
  enabled: false   # true → 替换 fusion
aegis:
  enabled: false   # true → κ-Snap 升级为进化搜索
causal_prior:
  enabled: false   # true → heuristic_order 引入因果引导
eval:
  enabled: false   # true → 启用 RHAE 评估
```

---

## 5. 待确认问题

1. **Web Dashboard** 工作量较大（Flask + React + Recharts），是否在本 PR 中包含，还是单独 PR？
   → 建议：本 PR 仅含 Python 后端，前端单独 PR
2. **`luzhao_dna.py` 是否改为引用 `math_sequences.py`**？
   → 建议：渐进迁移（`math_sequences.py` 作为工具库，`luzhao_dna.py` 保留高层接口）
3. **AEGIS 是否默认启用**？
   → 默认关闭（`aegis.enabled: false`），避免性能回归

---

## 6. 下一步

1. ✅ 新建 7 个吸收文件
2. ✅ 更新 `tomas_solver.py` + `config/default.yaml`
3. ⬜ 创建 Web Dashboard 最小化后端（Flask API）
4. ⬜ 更新 `luzhao_dna.py` 引用 `math_sequences.py`
5. ⬜ 运行 290 测试确保无回归
6. ⬜ Git commit → push → 创建 GitHub PR
