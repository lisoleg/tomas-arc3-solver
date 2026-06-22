# Changelog

All notable changes to TOMAS ARC-AGI-3 Solver are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.3.0] — 2026-06-22

### Added — CUDA GPU Parallelization
- New `src/core/numba_cuda_kernels.py`: 7 `@cuda.jit` GPU kernels with automatic CPU fallback
  - `batch_grid_equal_cuda` — N×M grid equality matrix (one block per pair)
  - `batch_mirror_cuda` — parallel mirror of N grids (thread-per-pixel)
  - `batch_rotate_cuda` — parallel rotation of N grids
  - `batch_betti0_cuda` — parallel connected component counting (atomic.add)
  - `batch_color_hist_cuda` — 10-bin color histograms (10 threads per grid)
  - `batch_nonzero_count_cuda` — parallel non-zero pixel counting
  - `batch_grid_distance_cuda` — N×M pixel distance matrix
- New `src/core/cuda_kernels.py`: CuPy-based GPU batch verification
  - `CudaBatchVerifier` with `batch_grid_equal`, `batch_grid_distance`, `batch_mirror`, `batch_rotate`
- Dual CUDA backend in `KappaSnapSearcher`: numba `@cuda.jit` preferred → CuPy fallback → CPU

### Added — Advanced Pruning Pipeline
- New `src/solver/pruning_optimizer.py`: 8 pruning strategies
  - **Pre-Phase A fast filters** (run before topo hash):
    1. `grid_shape_prune` — O(1) output dimension check
    2. `nonzero_count_prune` — O(H×W) pixel count check
    3. `color_histogram_prune` — O(H×W) color distribution check
    4. `betti0_prune` — O(H×W) connected component check
  - **Post-Phase A**:
    5. `symmetry_dedup` — symmetry-equivalent candidate removal
    6. `incremental_mdl_prune` — depth-3 partial MDL pruning
    7. `heuristic_order` — MDL/primitive-count/name ordering
  - **Verification**:
    8. MDL threshold filter
- `fast_pre_filter()` pipeline: shape → nonzero → color_hist → Betti0 (60-80% elimination)
- All filters use content-based caching (grid bytes hash)

### Added — Tests
- New `tests/test_cuda_kernels.py`: 31 tests for CUDA/CuPy kernels
- New `tests/test_pruning_optimizer.py`: 32 tests for pruning strategies

### Changed
- `kappa_snap_searcher.py`: Integrated Pre-Phase A filtering, dual CUDA backend, pruning stats
- `pruning_optimizer.py`: Added `enable_shape_filter`, `enable_color_hist`, `enable_nonzero_count` flags
- `tomas_solver.py`: Merged CUDA and pruning config into search config
- `config/default.yaml`: Added `cuda` and `pruning` sections

### Test Results
- **290/290 tests PASSED** (up from 227 in v2.2)

---

## [2.2.0] — 2026-06-22

### Added — Numba JIT Compilation
- New `src/core/numba_kernels.py`: 20 `@njit` compiled CPU kernels
  - `grid_equal_kernel` — grid equality comparison (5.1× vs np.array_equal)
  - `overlay_kernel`, `subtract_kernel`, `union_kernel`, `intersection_kernel` — set operations (3-5×)
  - `move_kernel`, `copy_with_offset_kernel` — grid translation (3-5×)
  - `draw_line_kernel` — Bresenham line drawing (20-50×)
  - `gravity_kernel` — gravity simulation (3-5×)
  - `resize_kernel` — grid resize (2-3×)
  - `symmetry_kernel` — symmetry completion (3-5×)
  - `color_swap_kernel` — color swap (3-5×)
  - `skeleton_kernel` — morphological skeleton (3-5×)
  - `grid_distance_kernel` — pixel distance (3-5×)
  - `count_nonzero_kernel` — non-zero count (2-3×)
  - `extract_pattern_kernel` — pattern extraction (2-3×)
  - `detect_compound_pattern_kernel` — compound pattern detection
  - `bresenham_line` — Bresenham algorithm (20-50×)
  - `fill_missing_symmetry` — symmetry fill (3-5×)
  - `compute_area_ratio_kernel` — area ratio (2-3×)
  - `match_ratio_kernel` — match ratio
  - `residual_trend_kernel` — residual trend detection
  - `betti0_kernel` — Betti0 computation

### Changed — JIT Integration
- `dsl_primitives.py`: Integrated 13 numba kernels into hot-path primitives
- `kappa_snap_searcher.py`: JIT-accelerated grid comparison in `_verify_against_demos`
- `gaussex_verifier.py`: JIT-accelerated `verify_program`
- `video_tensor.py`: JIT-accelerated `extract_deltaT` move detection
- `bayesian_confidence.py`: JIT-accelerated likelihood computation

### Added — Dependencies
- `numba>=0.59.0` added to `requirements.txt`

### Test Results
- **227/227 tests PASSED**

---

## [2.1.0] — 2026-06-22

### Changed — Vectorization Optimizations (14 items)
- `dsl_primitives.py`:
  - Vectorized `_resize` using `np.ix_` indexing
  - Vectorized `_move` using array slicing
  - Vectorized `_copy` using `np.where` for offset computation
  - Vectorized `_gravity` using compact column/row compaction
  - Vectorized `_symmetry_detect` using `np.fliplr`/`np.flipud`
  - Vectorized `_extract_pattern` using `np.tile` for full-grid comparison
  - Vectorized `_skeleton` using `ndimage.skeletonize`
  - Vectorized `_draw_line`, `_color_swap`, `_overlay`, `_subtract`, `_union`, `_intersection`
- `octonion_hyperedge.py`:
  - Vectorized `decode_to_grid` using `np.where` for batch pixel extraction
  - Vectorized `_compute_area_ratios` using `np.bincount` lookup table
- `video_tensor.py`:
  - Cached Betti₀ computation in `detect_singularity`/`find_singularity_frames`
- `delta_composer.py`:
  - Replaced `copy.deepcopy` with `clone()` method on ProgramNode (~3× faster)
- `kappa_snap_searcher.py`:
  - `ThreadPoolExecutor` parallel verification in Phase B
  - Removed redundant MDL recomputation in `enumerate_candidates`
- `gaussex_verifier.py`:
  - CRC32 hash (`zlib.crc32`) replacing Python `hash()` for array hashing
  - Fiber cache: `(program_id, pair_idx) → fiber_set`
- `tomas_solver.py`:
  - Cached `parse_input` result to avoid redundant task parsing

### Test Results
- **227/227 tests PASSED**

---

## [2.0.0] — 2026-06-22

### Added — Core Theoretical Components
- New `src/core/luzhao_dna.py`: Luzhao DNA topological invariants
  - Fibonacci sequence encoding of Betti₀
  - Lucas sequence encoding of symmetry residual
  - Bagua (八卦) constants for color permutation groups
  - DNA hash for enhanced Phase A filtering precision
- New Betti₀ singularity detection in `video_tensor.py`
  - `detect_singularity()` — sudden connected component drops
  - `find_singularity_frames()` — singularity events with before/after values
- New `src/solver/enpv_decision.py`: ENPV decision module
  - Expected Net Positive Value for search termination
  - Trend tracking (improving/declining/stable)
- New VLM differential analysis in `multi_scale_analyzer.py`
- New slip cost calculator (`src/utils/slip_cost.py`) for phase alignment fee in MDL
- New GATlab axiom interfaces on `DSLElement`:
  - `verify_compositionality()` — f(g(x)) = (f∘g)(x)
  - `verify_reversibility()` — f⁻¹(f(x)) = x

### Added — Enhanced Components
- `bayesian_confidence.py`: Jitter variance estimation
  - Adaptive sigma from residual standard deviation
  - Residual trend detection (monotonically increasing → 50% penalty)
- `delta_history_buffer.py`: Enhanced pattern detection
  - Window size 5 (5-frame temporal receptive field)
  - Betti₀-based mutation detection
  - Linear prediction with confidence scoring
- `topo_hash.py`: Luzhao DNA hash integration
- `hypergraph.py`: Temporal sequence splitting

### Changed
- `dsl_primitives.py`: Added `octonion_transform` descriptions for all primitives
- `delta_composer.py`: Slip cost integration into MDL
- `config/default.yaml`: Added `luzhao_dna`, `enpv`, `jitter`, `slip_cost` sections

### Test Results
- **227/227 tests PASSED**

---

## [1.0.0] — 2026-06-22

### Added — Complete 5-Layer Pipeline
- **Layer 1**: `KaggleFormatAdapter` — Kaggle JSON → `VideoARCTask`
- **Layer 2**: `OctonionHyperEdge` — 8-component reversible grid encoding
  - `HyperGraph` — topological invariants, hash fingerprinting, Willems fibers
  - `VideoTemporalEncoder` — frame encoding, ΔT extraction, singularity detection
  - `DeltaHistoryBuffer` — arithmetic/periodic/mutation pattern detection
  - `DSLElement` + `ProgramNode` — 30 ARC DSL primitives with composition tree
  - `TopoHashFilter` — Phase A topological hash quick filter
- **Layer 3**: `KappaSnapSearcher` — κ-Snap Two-Phase search
  - `GaussExVerifier` — Willems behavioral fiber verification
  - `DeltaTCombinator` — chain/additive/conditional composition
  - `TransferEngine` — cross-video transfer via fiber intersection
  - `LibraryLearning` — DreamCoder-style subroutine extraction
- **Layer 4**: `BayesianConfidence` — posterior ranking with MDL priors
  - `FusionScorer` — multi-modal score fusion (symbolic + VL + cross-modal)
  - `MultiScaleAnalyzer` — multi-scale frame analysis with keyframe extraction
- **Layer 5**: `TOMASSolver` — mode dispatch and auto-switching
  - `VideoSolver` — video mode (pure symbolic)
  - `TransferSolver` — transfer learning mode

### Added — Infrastructure
- `config/default.yaml` — complete YAML configuration
- `src/api/deepseek_adapter.py` + `deepseek_vl.py` — DeepSeek VL API adapters
- `src/verify/a100_verifier.py` + `a100_video_verifier.py` — A100 batch verification
- `src/utils/` — config loader, audit logger, GPU optimizer, Kaggle format, TensorBoard monitor
- `docker/docker-compose.yml` — Docker deployment
- `main.py` — CLI entry point with 4 modes (video/bayesian/fusion/auto)
- `setup.py` — package installation

### Added — Tests (227 tests)
- `test_octonion.py` — encoding/decoding reversibility, invariants
- `test_dsl.py` — all 30 primitives, ProgramNode composition
- `test_kappa_snap.py` — Two-Phase search, candidate enumeration, MDL ranking
- `test_gaussex.py` — fiber verification, fiber intersection, cross-video transfer
- `test_video.py` — temporal encoding, ΔT extraction, singularity detection
- `test_bayesian.py` — prior/likelihood/posterior, evidence, candidate ranking

### Test Results
- **227/227 tests PASSED**
