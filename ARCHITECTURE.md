# TOMAS ARC-AGI-3 Solver — System Architecture

> Version 2.3.0 | Updated 2026-06-22

## 1. Design Philosophy

TOMAS (太乙互搏 — Taiyi Mutual-Play) is grounded in three theoretical pillars:

1. **Algebraic Topology** — Grid states are encoded as octonion hyperedges; topological invariants (Betti₀, symmetry residuals, color permutation groups) serve as necessary conditions for candidate filtering.

2. **Behavioral System Theory (Willems)** — Demo pairs define behavioral fiber constraints; valid programs must lie in the fiber intersection across all demos.

3. **Bayesian Inductive Inference** — Program hypotheses are ranked by posterior probability P(H|E), combining MDL-based priors with Gaussian likelihoods under adaptive noise estimation.

The system follows a **compositional DSL** paradigm: transformations are expressed as compositions of ~30 primitive operations, with program complexity measured by Minimum Description Length (MDL).

## 2. Five-Layer Architecture

### Layer 1: Input Adaptation

**Module**: `src/utils/kaggle_format.py`

Converts Kaggle ARC-AGI-3 JSON task format into internal `VideoARCTask` objects containing:
- `demo_pairs`: List of {input: [grids], output: [grids]} pairs
- `test_frames`: List of test input grids for prediction

**Key class**: `KaggleFormatAdapter`
- `parse_input(task_dict) → VideoARCTask`
- Handles both single-grid and multi-frame (video) formats
- Content-based caching to avoid redundant parsing

### Layer 2: Encoding

#### 2.1 Octonion HyperEdge Encoding

**Module**: `src/core/octonion_hyperedge.py`

Each grid pixel is encoded as an 8-component octonion vector with explicit cognitive semantics:

| Component | Symbol | Semantic | Range |
|-----------|--------|----------|-------|
| e₀ | real_mask | Object existence | [0, 1] |
| e₁ | e1_x | Object constancy (x-coord) | [-1, 1] |
| e₂ | e2_y | Translation invariance (y-coord) | [-1, 1] |
| e₃ | e3_color | Rotation / color phase angle | [0, 2π] |
| e₄ | e4_component_id | Scale / connected component ID | [0, 1] |
| e₅ | e5_symmetry | Boundary touch / symmetry residual | [0, 1] |
| e₆ | e6_boundary | Color mapping / boundary mask | [0, 1] |
| e₇ | e7_area_ratio | Time phase / area ratio | [0, 1] |

**Reversibility**: `decode_to_grid()` reconstructs the original grid from the octonion array by extracting color from the e₃ phase angle. This is verified by `EncodingReversibilityError`.

**Encoding pipeline**:
```
Grid (H×W, int8)
  → np.meshgrid (coordinates)
  → 8 component computations (vectorized)
  → normalize (clip to valid ranges)
  → Encoded array (H*W × 8, float32)
```

**Neural analogy**: The orthogonality of octonion imaginary components mirrors feature selectivity of single neurons in visual cortex (Nature single-neuron study).

#### 2.2 HyperGraph

**Module**: `src/core/hypergraph.py`

A `HyperGraph` manages collections of `OctonionHyperEdge` objects and provides:
- **Topological hash**: MD5 fingerprint encoding Betti₀, symmetry, colors, pixels, shapes, temporal length
- **Isomorphism checking**: Hash fingerprint as necessary condition + full invariant comparison
- **Willems fiber computation**: `compute_fiber(program) → set[int]` — set of valid output state hashes
- **Temporal splitting**: `get_temporal_sequence() → list[HyperGraph]` for per-frame analysis

#### 2.3 Luzhao DNA Topological Invariants

**Module**: `src/core/luzhao_dna.py`

Encodes hypergraph structure into three Luzhao DNA sequences:

| Sequence | Encoding | Source |
|----------|----------|--------|
| Fibonacci | F_n terms from Betti₀ | Connected components |
| Lucas | L_n terms from symmetry residual | Symmetry axis |
| Bagua | 八卦 constants from color permutation | Color distribution |

The DNA hash `LUZHAO:F|L|B` is appended to the base topological hash for enhanced Phase A discrimination.

#### 2.4 Video Temporal Encoder

**Module**: `src/core/video_tensor.py`

Encodes video frame sequences into temporal hypergraphs and extracts inter-frame transformations:
- `encode_frames()` — Each frame → OctonionHyperEdge → HyperGraph
- `extract_deltaT(frame_a, frame_b) → ProgramNode` — Infers transformation (resize, copy, color-swap, mirror, rotate, move, overlay)
- `detect_singularity() → list[int]` — Betti₀ singularity detection (sudden connected component drops > 50%)
- `predict_next_frame(deltaT, last_frame) → ndarray` — Apply inferred transformation

#### 2.5 Delta History Buffer

**Module**: `src/core/delta_history_buffer.py`

Maintains a sliding window (default 5) of delta-T transformations with pattern detection:
- **Arithmetic**: Uniform/constant motion (all deltas identical)
- **Periodic**: Repeating sub-sequences
- **Mutation**: Sudden pattern break (Betti₀ drop or structural change)
- **Linear prediction**: Extrapolate next delta from temporal receptive field

### Layer 3: Search & Verification

#### 3.1 κ-Snap Two-Phase Search

**Module**: `src/solver/kappa_snap_searcher.py`

The core search algorithm operates in a multi-stage pipeline:

```
Candidates (depth 1-3)
  │
  ▼
Pre-Phase A: Fast Invariant Filters (v2.3)
  │ 1. Grid shape filter (O(1) per candidate)
  │ 2. Non-zero pixel count filter (O(H×W))
  │ 3. Color histogram filter (O(H×W))
  │ 4. Betti0 filter (O(H×W) + scipy)
  │ → Eliminates 60-80% of candidates
  ▼
Phase A: Topological Hash Filter
  │ Compute topo_hash(P(I_i)) vs topo_hash(O_i)
  │ + Luzhao DNA hash for enhanced discrimination
  │ → Eliminates 90%+ of remaining candidates
  ▼
Post-Phase A: Refinement
  │ 5. Symmetry equivalence deduplication
  │ 6. Heuristic candidate ordering (MDL ↑, primitives ↑)
  ▼
Phase B: MDL-Priority Enumeration + Verification
  │ 7. MDL threshold filter (prune > threshold)
  │ 8. GaussEx fiber verification (or GPU batch verify)
  │ 9. ENPV early termination (abandon if ENPV < 0)
  ▼
Ranked Valid Programs
```

**Key design decisions**:
- Depth 3 enumeration is limited to primitives with MDL ≤ 10 to avoid combinatorial explosion
- Incremental MDL pruning: if first two primitives at depth 3 already exceed threshold, skip third
- ThreadPoolExecutor parallel verification (CPU) or GPU batch verification (CUDA)

#### 3.2 Pruning Optimizer

**Module**: `src/solver/pruning_optimizer.py` (v2.3)

Eight pruning strategies organized by cost:

| Stage | Strategy | Cost | Target |
|-------|----------|------|--------|
| Pre-A | Grid shape filter | O(1) | Wrong output dimensions |
| Pre-A | Non-zero count filter | O(H×W) | Pixel count mismatch |
| Pre-A | Color histogram filter | O(H×W) | Color distribution mismatch |
| Pre-A | Betti0 invariant | O(H×W)+scipy | Topological mismatch |
| Post-A | Symmetry dedup | O(1) per candidate | Equivalent programs |
| Enum | Incremental MDL | O(1) | Excessively complex branches |
| Post-A | Heuristic ordering | O(N log N) | ENPV effectiveness |
| Verify | MDL threshold | O(1) | Over-complex programs |

All filters use content-based caching (grid bytes hash) for deterministic, repeatable behavior.

#### 3.3 GaussEx Fiber Verification

**Module**: `src/solver/gaussex_verifier.py`

Verifies candidate programs using Willems behavioral system theory:
- Each demo pair defines a **behavioral fiber** — the set of valid output states
- A program is valid iff it passes **all** demo pair fibers (fiber intersection is non-empty)
- CRC32 hash (faster than Python `hash()` for bytes) for state hashing
- Fiber cache: `(program_id, pair_idx) → fiber_set` to avoid recomputation
- Batch verification interface for GPU acceleration

**Cross-video transfer**: `cross_video_transfer()` checks if programs generalize across multiple videos via fiber intersection.

#### 3.4 ENPV Decision

**Module**: `src/solver/enpv_decision.py`

Expected Net Positive Value for search termination:

```
ENPV = Σ_i P(H_i|E) × (1 - MDL_i / max_MDL) - cost(remaining_evaluations)
```

- ENPV > 0 → continue search (expected value positive)
- ENPV < 0 → abandon search (not worth continuing)
- Trend tracking: improving / declining / stable based on 3-sample window

### Layer 4: Confidence & Fusion

#### 4.1 Bayesian Confidence

**Module**: `src/solver/bayesian_confidence.py`

Computes posterior probability P(H|E) = P(E|H) × P(H) / P(E):

- **Prior** P(H) ∝ exp(-λ × MDL) — simpler programs have higher prior
- **Likelihood** P(E|H) — Gaussian fit with adaptive sigma (Jitter estimation):
  - sigma = 0.7 × std(residuals) + 0.3 × base_sigma
  - Match ratio = 1 - ||predicted - observed|| / (size × 9)
- **Evidence** P(E) = Σ P(E|H_i) × P(H_i) — marginal likelihood
- **Residual trend detection**: Monotonically increasing residuals → 50% penalty

**Jitter estimation**: Adaptive noise sigma from multi-frame residual autocorrelation. Programs with erratic predictions get higher sigma (lower likelihood).

#### 4.2 Fusion Scorer

**Module**: `src/solver/fusion_scorer.py`

Multi-modal score fusion:
```
score = α × symbolic_score + β × visual_score + γ × cross_modal_score
```
- α = 0.4, β = 0.35, γ = 0.25 (default)
- Degraded weights (VL unavailable): α = 0.6, γ = 0.4
- Adaptive weight adjustment based on demo pair count

#### 4.3 Multi-Scale Analyzer

**Module**: `src/core/multi_scale_analyzer.py`

Multi-scale frame analysis using keyframe extraction and VL (DeepSeek VL) analysis:
- Keyframe extraction identifies temporally significant frames
- VL adapter provides visual-language embeddings for cross-modal scoring
- Graceful degradation when VL API is unavailable

### Layer 5: Output & Optimization

#### 5.1 Library Learning

**Module**: `src/solver/library_learning.py`

DreamCoder-style subroutine extraction:
- Pattern frequency tracking (≥ 3 occurrences → abstraction)
- JSON persistence (`library.json`)
- MDL reduction: matched abstractions reduce program complexity
- Max 200 abstractions stored

#### 5.2 Transfer Engine

**Module**: `src/core/transfer_engine.py`

Cross-video transfer via fiber intersection:
- Only transfers **universal** DSL subroutines (via Library Learning)
- Task-specific patterns are not transferred (competition compliance)
- `compute_cross_video_fiber(source, target) → set[int]`

#### 5.3 TOMAS Solver (Main Orchestrator)

**Module**: `src/solver/tomas_solver.py`

The `TOMASSolver` class orchestrates the full pipeline:
- Mode dispatch: video / bayesian / fusion / auto
- Auto-select mode based on time budget and task complexity
- Pre-parse input once, share across mode-specific solvers
- Audit logging for all solve actions

## 3. Acceleration Layer

### 3.1 Numba JIT Kernels (v2.2)

**Module**: `src/core/numba_kernels.py`

20 `@njit` compiled kernels with explicit type signatures:

| Kernel | Function | Speedup |
|--------|----------|---------|
| `grid_equal_kernel` | Grid equality comparison | 5.1× vs np.array_equal |
| `overlay_kernel` | Pixel overlay | 3-5× |
| `subtract_kernel` | Pixel subtraction | 3-5× |
| `union_kernel` | Pixel union | 3-5× |
| `intersection_kernel` | Pixel intersection | 3-5× |
| `move_kernel` | Grid translation | 3-5× |
| `copy_with_offset_kernel` | Copy with offset | 3-5× |
| `draw_line_kernel` | Bresenham line | 20-50× |
| `gravity_kernel` | Gravity simulation | 3-5× |
| `resize_kernel` | Grid resize | 2-3× |
| `symmetry_kernel` | Symmetry completion | 3-5× |
| `color_swap_kernel` | Color swap | 3-5× |
| `skeleton_kernel` | Skeleton extraction | 3-5× |
| `grid_distance_kernel` | Pixel distance | 3-5× |
| `count_nonzero_kernel` | Non-zero count | 2-3× |
| `extract_pattern_kernel` | Pattern extraction | 2-3× |
| `detect_compound_pattern_kernel` | Compound pattern | 2-3× |
| `bresenham_line` | Line algorithm | 20-50× |
| `fill_missing_symmetry` | Symmetry fill | 3-5× |
| `compute_area_ratio_kernel` | Area ratio | 2-3× |

All kernels use `cache=True`, `nogil=True`, `fastmath=True` for maximum performance.

### 3.2 Numba CUDA GPU Kernels (v2.3)

**Module**: `src/core/numba_cuda_kernels.py`

7 `@cuda.jit` GPU kernels with automatic CPU fallback:

| Kernel | GPU Strategy | CPU Fallback |
|--------|-------------|--------------|
| `batch_grid_equal_cuda` | N×M block grid, early exit | `grid_equal_kernel` |
| `batch_mirror_cuda` | Thread-per-pixel parallel | `np.fliplr/flipud` |
| `batch_rotate_cuda` | Thread-per-pixel parallel | `np.rot90` |
| `batch_betti0_cuda` | Thread-per-grid, atomic.add | `scipy.ndimage.label` |
| `batch_color_hist_cuda` | 10 threads/grid, one per color | `np.bincount` |
| `batch_nonzero_count_cuda` | atomic.add reduction | `np.count_nonzero` |
| `batch_grid_distance_cuda` | N×M block grid | `np.abs` + `np.sum` |

**Dual CUDA backend**: `KappaSnapSearcher` prefers numba `@cuda.jit` over CuPy, falling back to CPU when no GPU is available.

### 3.3 CuPy GPU Batch Verification (v2.3)

**Module**: `src/core/cuda_kernels.py`

CuPy-based GPU batch verification as alternative to numba CUDA:
- `CudaBatchVerifier.batch_grid_equal(preds, expected) → (N, M) bool matrix`
- `CudaBatchVerifier.batch_grid_distance(preds, expected) → (N, M) float matrix`
- `CudaBatchVerifier.batch_mirror(grids, axis) → (N, H, W) array`
- `CudaBatchVerifier.batch_rotate(grids, angle) → (N, H, W) array`

## 4. Data Flow

```
Task JSON
  │
  ├─→ KaggleFormatAdapter.parse_input()
  │     → VideoARCTask { demo_pairs, test_frames }
  │
  ├─→ TOMASSolver.solve(task, mode)
  │     │
  │     ├─→ auto_select_mode(time_budget, complexity)
  │     │     → "video" | "bayesian" | "fusion"
  │     │
  │     ├─→ [video mode]
  │     │     └─→ VideoSolver.solve()
  │     │           ├─→ VideoTemporalEncoder.encode_frames()
  │     │           ├─→ VideoTemporalEncoder.extract_all_deltaT()
  │     │           ├─→ DeltaHistoryBuffer.detect_pattern_full()
  │     │           └─→ predict_next_frame()
  │     │
  │     ├─→ [bayesian mode]
  │     │     ├─→ KappaSnapSearcher.two_phase_search()
  │     │     │     ├─→ enumerate_candidates(depth 1-3)
  │     │     │     ├─→ PruningOptimizer.fast_pre_filter()  [v2.3]
  │     │     │     ├─→ TopoHashFilter.quick_filter()        [Phase A]
  │     │     │     ├─→ PruningOptimizer.symmetry_dedup()    [v2.3]
  │     │     │     ├─→ PruningOptimizer.heuristic_order()   [v2.3]
  │     │     │     └─→ phase_b_enumerate()                  [Phase B]
  │     │     │           ├─→ [GPU] _phase_b_gpu_verify()
  │     │     │           └─→ [CPU] _phase_b_cpu_verify()
  │     │     ├─→ BayesianConfidence.rank_candidates()
  │     │     └─→ _predict_with_program()
  │     │
  │     └─→ [fusion mode]
  │           ├─→ (same as bayesian for search)
  │           ├─→ MultiScaleAnalyzer.analyze()
  │           ├─→ FusionScorer.fuse()
  │           └─→ _predict_with_program()
  │
  └─→ Output: { predictions, best_program_mdl, mode, ... }
```

## 5. Design Decisions

### 5.1 Why Octonion Encoding?

Octonions (8-dimensional normed division algebra) provide:
- **Sufficient dimensions**: 8 components cover object existence, position, color, topology, symmetry, boundary, and temporal context
- **Orthogonality**: Components are semantically independent, mirroring neural feature selectivity
- **Reversibility**: Encoding is fully reversible — decode_to_grid() reconstructs the original grid
- **Normalization**: All components clip to bounded ranges, preventing overflow in long sequences

### 5.2 Why Two-Phase Search?

The candidate space grows exponentially with composition depth (~30^depth). Two-Phase search addresses this:
- **Phase A** uses cheap topological invariants as necessary conditions, eliminating 90%+ of candidates
- **Phase B** performs expensive exact verification only on the surviving ~10%
- **Pre-Phase A** (v2.3) adds even cheaper filters (shape, pixel count, color histogram) before Phase A

### 5.3 Why Willems Behavioral Systems?

Willems' behavioral approach provides a natural framework for demo constraint satisfaction:
- Each demo defines a fiber (set of valid behaviors)
- Valid programs must lie in the **intersection** of all demo fibers
- This is more general than input-output mapping — it captures the full behavioral repertoire

### 5.4 Why ENPV for Termination?

Search can be time-consuming. ENPV provides a principled stopping criterion:
- If the expected benefit of continuing (weighted by posteriors) is less than the computational cost, stop
- This prevents wasted computation on diminishing returns
- Trend tracking (improving/declining/stable) provides additional signal

### 5.5 Why Dual CUDA Backend?

Different GPU environments have different toolchains:
- **Numba @cuda.jit**: Pure Python, no external CUDA toolkit needed, best for Kaggle
- **CuPy**: Higher-level, better for complex batch operations, requires CUDA toolkit
- The system auto-detects availability and selects the best backend, falling back to CPU

## 6. Module Dependency Graph

```
tomas_solver.py
  ├── kappa_snap_searcher.py
  │     ├── topo_hash.py
  │     │     ├── hypergraph.py
  │     │     │     └── octonion_hyperedge.py
  │     │     └── luzhao_dna.py
  │     ├── pruning_optimizer.py (v2.3)
  │     ├── numba_kernels.py (v2.2)
  │     ├── numba_cuda_kernels.py (v2.3)
  │     ├── cuda_kernels.py (v2.3)
  │     └── enpv_decision.py
  ├── gaussex_verifier.py
  │     └── numba_kernels.py
  ├── bayesian_confidence.py
  │     └── numba_kernels.py
  ├── fusion_scorer.py
  │     └── deepseek_vl.py
  ├── video_solver.py
  │     ├── video_tensor.py
  │     │     └── dsl_primitives.py
  │     │           └── numba_kernels.py
  │     └── delta_composer.py
  │           └── slip_cost.py
  ├── library_learning.py
  ├── transfer_solver.py
  │     └── transfer_engine.py
  ├── multi_scale_analyzer.py
  │     └── keyframe_extractor.py
  ├── delta_history_buffer.py
  ├── kaggle_format.py
  ├── gpu_optimizer.py
  └── logger.py
```
