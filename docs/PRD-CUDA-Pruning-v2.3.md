# 增量产品需求文档（PRD）- TOMAS v2.3 CUDA/Pruning 优化

> **产品**: TOMAS ARC-AGI-3 Solver  
> **版本**: v2.3 (CUDA + 高级剪枝优化版)  
> **基线**: v2.2 (Numba JIT 层, 227/227 测试通过)  
> **日期**: 2026-06-22  
> **产品经理**: 许清楚 (Xu Qingchu)

---

## 一、产品目标

在 v2.2 Numba JIT 优化基础上，通过 **GPU CUDA 批量加速** 和 **高级搜索剪枝策略** 将 TOMAS ARC-AGI-3 Solver 的搜索-验证流水线吞吐量提升 5-20x，使其能在 Kaggle T4 GPU (16GB) 上高效运行，支撑 ARC-AGI-3 竞赛的实时推理需求。

**核心指标**：
- Phase B 批量验证：GPU 并行 → **10-50x** 加速（对比当前 4 线程 ThreadPoolExecutor）
- Phase A 候选过滤率：从 90% 提升至 **95%+**（减少 Phase B 负担）
- 深度 3 候选枚举：剪枝后减少 **50-80%** 无效候选
- 端到端 CLI 延迟：task_001 从 ~15s 降至 **< 8s**

---

## 二、用户故事

| ID | 角色 | 需求 | 价值 | 验收标准 |
|----|------|------|------|----------|
| US01 | 竞赛参赛者 | 我想要在 GPU 上批量验证候选程序 | 以便大幅缩短搜索时间 | 1. CUDA 可用时自动启用 GPU 批量验证 2. 无 GPU 时优雅降级到 CPU 3. 批量验证 1000 候选 × 3 demo pair 延迟 < 0.5s |
| US02 | 竞赛参赛者 | 我想要更智能的候选剪枝 | 以便减少无效候选进入 Phase B | 1. Betti₀ 不变量剪枝在 Phase A 后额外过滤 30%+ 2. MDL 增量剪枝在枚举阶段即剔除超阈值候选 3. 对称性去重消除等价候选 |
| US03 | 竞赛参赛者 | 我想要启发式候选排序 | 以便高概率程序优先验证 | 1. 基于历史命中率的先验排序 2. 低 MDL + 高拓扑匹配度优先 3. ENPV 提前终止更精准 |
| US04 | 研究开发者 | 我想要 GPU 加速的网格操作 | 以便在更大网格上高效运行 | 1. CuPy 实现 mirror/rotate/move/copy 等核心操作 2. 批量 grid_equal 在 GPU 上并行 3. 批量 grid_distance 用于贝叶斯似然 |
| US05 | 部署运维 | 我想要配置化的 GPU/CPU 模式切换 | 以便在不同环境灵活部署 | 1. config 中新增 cuda 配置段 2. 支持 auto/force-cpu/force-gpu 模式 3. 启动时输出设备信息日志 |

---

## 三、需求池（按优先级排序）

### P0（必须做，不做无法上线）

- [ ] **CUDA 批量网格验证内核** (`src/core/cuda_kernels.py`)
  - `batch_grid_equal_gpu`: 批量验证 N 个候选 × M 个 demo pair 的网格相等性
  - `batch_grid_distance_gpu`: 批量计算网格距离（贝叶斯似然用）
  - 基于 CuPy 实现，自动 fallback 到 Numba CPU 内核
  - `HAS_CUDA` 标志 + 优雅降级

- [ ] **高级剪枝优化器** (`src/solver/pruning_optimizer.py`)
  - `betti0_prune`: 基于 Betti₀ 拓扑不变量的候选剪枝
  - `incremental_mdl_prune`: 枚举阶段 MDL 增量剪枝（部分 MDL 超阈值即跳过）
  - `symmetry_dedup`: 对称等价候选去重
  - `heuristic_order`: 启发式候选排序（低 MDL + 高拓扑匹配优先）

- [ ] **KappaSnap 搜索器集成** (修改 `src/solver/kappa_snap_searcher.py`)
  - `phase_b_enumerate`: GPU 批量验证路径（CUDA 可用时）
  - `enumerate_candidates`: 集成增量 MDL 剪枝
  - `two_phase_search`: 集成 Betti₀ 剪枝 + 对称去重 + 启发式排序

- [ ] **GaussEx 验证器集成** (修改 `src/solver/gaussex_verifier.py`)
  - `verify_program_batch`: 批量验证接口（GPU 加速）

- [ ] **配置更新** (修改 `config/default.yaml`)
  - 新增 `cuda` 配置段：device, batch_size, enable_batch_verify
  - 新增 `pruning` 配置段：enable_betti0, enable_symmetry_dedup, enable_heuristic_order

- [ ] **依赖更新** (修改 `requirements.txt`)
  - 添加 `cupy-cuda12x>=13.0` (可选依赖)

### P1（应该做，做了体验完整）

- [ ] **GPU 优化器扩展** (修改 `src/utils/gpu_optimizer.py`)
  - 新增 `batch_verify_candidates` 方法：管理 GPU 内存、批量验证
  - 自适应批大小：根据 VRAM 动态调整验证批量

- [ ] **贝叶斯置信度集成** (修改 `src/solver/bayesian_confidence.py`)
  - `compute_likelihood`: 使用 GPU 批量 grid_distance

- [ ] **TOMAS Solver 集成** (修改 `src/solver/tomas_solver.py`)
  - 初始化时检测 CUDA 可用性
  - 传递 CUDA 配置给搜索器和验证器

### P2（可以做，后续迭代）

- [ ] **Phase A GPU 并行化** — 拓扑哈希计算也用 GPU 加速
- [ ] **CUDA 流水线** — 多 CUDA stream 重叠计算和数据传输
- [ ] **混合精度验证** — FP16 验证 + FP32 确认两阶段策略
- [ ] **候选缓存** — 跨任务的候选程序缓存（避免重复计算）

---

## 四、技术约束

1. **Kaggle T4 GPU**: 16GB VRAM, CUDA 12.x, 需支持 CuPy
2. **优雅降级**: 无 GPU 时必须回退到 v2.2 的 Numba CPU 路径，性能不低于当前
3. **测试兼容**: 227 个现有测试必须全部通过（无 GPU 环境下）
4. **Python 3.10+**: 兼容 Kaggle Notebook 环境
5. **CuPy 可选**: `cupy` 不在硬依赖中，import 失败时优雅降级

---

## 五、待确认问题

1. **CuPy 版本**: Kaggle T4 环境用 `cupy-cuda12x` 还是 `cupy-cuda11x`？→ 默认 `cupy-cuda12x`，fallback 到 `cupy-cuda11x`
2. **批量验证粒度**: 一次验证多少候选？→ 默认 256，自适应 VRAM
3. **Betti₀ 计算开销**: Betti₀ 剪枝本身是否值得？→ 预计算 + 缓存，O(1) 查表
4. **对称去重策略**: 使用哪种对称等价？→ 4 种基本对称（恒等、水平镜像、垂直镜像、180°旋转）

---

## 六、验收标准（整体）

| 验收项 | 标准 |
|--------|------|
| 现有测试 | 227/227 PASSED（无 GPU 环境） |
| 新增测试 | CUDA 内核测试 + 剪枝优化器测试 ≥ 20 个 |
| CLI 验证 | 3 任务 × 3 模式 = 9/9 通过 |
| 性能基准 | GPU 模式下批量验证加速 ≥ 10x（对比 CPU 4 线程） |
| 剪枝效果 | Phase A 后候选减少 ≥ 30%（Betti₀ + 对称去重） |
| 优雅降级 | 无 CUDA/cupy 时功能正常，无报错 |
| 代码规范 | 类型注解、docstring、ruff 通过 |
