# TOMAS ARC-AGI-3 Solver

> **TOMAS** (太乙互搏 — Taiyi Mutual-Play) framework for the [ARC-AGI-3](https://www.kaggle.com/competitions/arc-agi-3) video reasoning competition.

[![Tests](https://img.shields.io/badge/tests-315%2F315-brightgreen)]()
[![Version](https://img.shields.io/badge/version-2.4.0-blue)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview

TOMAS ARC-AGI-3 Solver is an end-to-end video reasoning system that combines algebraic topology, behavioral system theory, and Bayesian inference to solve abstract reasoning tasks. It encodes ARC grid pixels as **octonion hyperedges** (8-component cognitive-semantic vectors), searches for transformation programs via **κ-Snap Two-Phase abductive search**, and verifies candidates through **GaussEx fiber intersection** (Willems behavioral systems).

### Core Innovations

| Component | Description |
|-----------|-------------|
| **Octonion HyperEdge Encoding** | Reversible grid-to-octonion transformation with 8 cognitive-semantic components (object existence, coordinates, color phase, component ID, symmetry, boundary, area ratio) |
| **κ-Snap Two-Phase Search** | Phase A: topological hash + Luzhao DNA quick-filter (90%+ elimination). Phase B: MDL-priority enumeration with ENPV early termination |
| **GaussEx Fiber Verification** | Willems behavioral system fiber intersection for demo constraint satisfaction |
| **Luzhao DNA Invariants** | Fibonacci/Lucas/Bagua sequence encoding of topological invariants for enhanced Phase A discrimination |
| **Bayesian Confidence** | Posterior ranking with MDL-based priors, Gaussian likelihoods, Jitter adaptive sigma, and residual trend detection |
| **Multi-Modal Fusion** | Symbolic + Visual (DeepSeek VL) + Cross-modal score fusion with Kaggle degradation |
| **ΔT Composition** | Chain (⊙), additive (+), and conditional (if-then) temporal transformation composition with pattern detection |
| **Library Learning** | DreamCoder-style subroutine extraction with JSON persistence and cross-video transfer |
| **Betti₀ Singularity Detection** | Topological event detection via connected-component drop analysis |
| **ENPV Decision** | Expected Net Positive Value for search termination — abandon when ENPV < 0 |

### Performance Optimizations (v2.1–v2.3)

| Version | Optimization | Speedup |
|---------|-------------|---------|
| v2.1 | NumPy vectorization (14 primitives) | 2–5× |
| v2.1 | ThreadPoolExecutor parallel verification | 3–4× |
| v2.1 | CRC32 hash + fiber cache | 1.5× |
| v2.2 | Numba @njit JIT compilation (20 kernels) | 3–20× per kernel |
| v2.3 | Numba @cuda.jit GPU kernels (7 kernels) | 10–100× (on GPU) |
| v2.3 | Advanced pruning pipeline (8 strategies) | 60–80% candidate reduction |

### v2.4 New Features (Cross-Repo Absorption)

| Component | Source | Description |
|-----------|--------|-------------|
| **RHAE Evaluator** | tomas-agi | Official ARC-AGI-3 evaluation framework with RHAE scoring |
| **ψ-Gate Fusion** | tomas-agi | Semantic gating with 5 core capabilities (ψ-anchor, MUS, φ-Gate, multi-world, tolerance decay) |
| **AEGIS Evolver** | tomas-agi | Program evolution engine (Digester → Planner → Evolver → Critic) |
| **Causal DSL Prior** | tomas-agi | SCM-based causal prior learning for DSL primitives |
| **GAT Axioms (Pure Python)** | tomas-agi | Pure Python GAT axioms as GATLab fallback |
| **Math Sequences** | Both repos | Unified Fibonacci/Lucas/Bagua mathematical tools |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Layer 1: Input Adaptation                    │
│              KaggleFormatAdapter → VideoARCTask                  │
├─────────────────────────────────────────────────────────────────┤
│                     Layer 2: Encoding                            │
│    OctonionHyperEdge → HyperGraph → VideoTemporalEncoder         │
│    (8-component octonion vectors)  (topo invariants)  (ΔT)      │
├─────────────────────────────────────────────────────────────────┤
│                  Layer 3: Search & Verification                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Pre-Phase A  │→ │  Phase A     │→ │     Phase B          │   │
│  │ Fast Filters │  │  Topo Hash   │  │  GaussEx Verify      │   │
│  │ (shape,nz,   │  │  + LuzhaoDNA │  │  + GPU Batch         │   │
│  │  color,Betti0)│ │  + Symmetry  │  │  + ENPV Terminate    │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                 Layer 4: Confidence & Fusion                     │
│     BayesianConfidence → FusionScorer → MultiScaleAnalyzer      │
│     (P(H|E) posterior)  (α·sym+β·VL+γ·cross)  (keyframe ext.)  │
├─────────────────────────────────────────────────────────────────┤
│                 Layer 5: Output & Optimization                   │
│     LibraryLearning → TransferEngine → Submission Output        │
├─────────────────────────────────────────────────────────────────┤
│                    Acceleration Layer                            │
│  NumbaKernels (@njit) │ NumbaCudaKernels (@cuda.jit) │ CuPy     │
│  (CPU JIT, 20 kernels)│ (GPU, 7 kernels)            │ (GPU alt) │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/lisoleg/tomas-arc3-solver.git
cd tomas-arc3-solver

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: GPU acceleration (CUDA 12.x)
pip install cupy-cuda12x
```

### Solving Tasks

```bash
# Solve a single task
python main.py --mode auto --input data/task_001.json --output output/result.json

# Solve all tasks in a directory
python main.py --mode auto --input data/ --output output/submission.json

# Use specific inference mode
python main.py --mode video --input data/ --output output/submission.json
python main.py --mode bayesian --input data/ --output output/submission.json
python main.py --mode fusion --input data/ --output output/submission.json
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up
```

### Kaggle

Copy `kaggle/notebook_template.ipynb` to Kaggle, upload the solver as a dataset, and run.

## Inference Modes

| Mode | Description | Speed | Accuracy | When to Use |
|------|-------------|-------|----------|-------------|
| `video` | Pure symbolic reasoning | Fast (< 40s) | Baseline | Low time budget, simple tasks |
| `bayesian` | Bayesian posterior ranking | Medium (40–80s) | Higher | Standard competition mode |
| `fusion` | Multi-modal fusion (symbolic + VL) | Slow (> 80s) | Highest | Complex tasks, maximum accuracy |
| `auto` | Time-budget adaptive selection | Adaptive | Optimal | Default — auto-selects best mode |

Auto mode selects based on time budget and task complexity:
- `time < 40s` or `complexity < 2` → `video`
- `40s ≤ time < 80s` or `complexity < 4` → `bayesian`
- `time ≥ 80s` and `complexity ≥ 4` → `fusion`

## DSL Primitives (30 Operations)

### Simple (MDL = 5)
`resize`, `fill-region`, `color-swap`, `mirror`, `rotate`, `scale`, `map-color`, `move`, `copy`, `crop`, `draw-line`, `boundary-detect`

### Medium (MDL = 10)
`complete-shape`, `objects-complete`, `gravity`, `flood-fill`, `extract-pattern`, `symmetry-detect`, `tile`, `overlay`, `subtract`, `union`, `intersection`, `skeleton`, `dilate`, `erode`, `label-connected`, `histogram`, `find-objects`, `count`

Each primitive supports:
- `apply(grid)` — fail-safe execution (returns original on error)
- `apply_to_hyperedge(edge)` — octonion-space transformation
- `verify_compositionality()` — GATlab axiom check
- `verify_reversibility()` — reversibility verification

## Configuration

All parameters are in `config/default.yaml`. Key sections:

```yaml
search:
  max_depth: 3              # max composition depth
  mdl_threshold: 50         # max MDL for valid programs
  time_limit_seconds: 80    # per-task time budget

bayesian:
  prior_lambda: 0.1         # P(H) ∝ exp(-λ·MDL)
  noise_sigma: 0.5          # observation noise std dev

fusion:
  alpha: 0.4                # symbolic weight
  beta: 0.35                # visual weight
  gamma: 0.25               # cross-modal weight

cuda:
  enabled: true             # auto-enable when GPU detected
  batch_size: 256           # candidates per GPU batch

pruning:
  enable_betti0: true       # Betti0 topological pruning
  enable_symmetry_dedup: true
  enable_heuristic_order: true
  enable_incremental_mdl: true
```

Environment variable overrides: `TOMAS__SECTION__KEY` format (e.g., `TOMAS__SEARCH__MAX_DEPTH=3`).

## Project Structure

```
tomas-arc3-solver/
├── src/
│   ├── core/                    # Core algorithms
│   │   ├── octonion_hyperedge.py    # Octonion encoding (reversible)
│   │   ├── hypergraph.py            # HyperGraph + topo invariants + Willems fibers
│   │   ├── topo_hash.py             # Phase A topological hash filter
│   │   ├── luzhao_dna.py            # Luzhao DNA (Fibonacci/Lucas/Bagua)
│   │   ├── dsl_primitives.py        # 30 ARC DSL primitives + ProgramNode
│   │   ├── video_tensor.py          # Video temporal encoder + ΔT extraction
│   │   ├── delta_history_buffer.py  # Pattern detection (arithmetic/periodic/mutation)
│   │   ├── keyframe_extractor.py    # Keyframe extraction for VL analysis
│   │   ├── multi_scale_analyzer.py  # Multi-scale frame analysis
│   │   ├── transfer_engine.py       # Cross-video transfer via fiber intersection
│   │   ├── numba_kernels.py         # 20 @njit CPU JIT kernels (v2.2)
│   │   ├── numba_cuda_kernels.py    # 7 @cuda.jit GPU kernels (v2.3)
│   │   ├── cuda_kernels.py          # CuPy GPU batch verification (v2.3)
│   │   ├── gat_axioms.py            # Pure Python GAT axioms (v2.4, GATLab fallback)
│   │   └── math_sequences.py        # Unified Fibonacci/Lucas/Bagua tools (v2.4)
│   ├── solver/                  # Search & verification
│   │   ├── kappa_snap_searcher.py   # κ-Snap Two-Phase search
│   │   ├── gaussex_verifier.py      # GaussEx fiber verification
│   │   ├── pruning_optimizer.py     # 8-strategy pruning pipeline (v2.3)
│   │   ├── bayesian_confidence.py   # Bayesian posterior ranking
│   │   ├── delta_composer.py        # ΔT composition (chain/additive/conditional)
│   │   ├── enpv_decision.py         # ENPV early termination
│   │   ├── fusion_scorer.py         # Multi-modal fusion scoring
│   │   ├── library_learning.py      # DreamCoder-style library learning
│   │   ├── tomas_solver.py          # Main solver (mode dispatch, ψ-Gate integrated)
│   │   ├── video_solver.py          # Video mode solver
│   │   ├── transfer_solver.py       # Transfer learning solver
│   │   ├── psi_fusion_gate.py       # ψ-Gate semantic gating (v2.4)
│   │   ├── aegis_evolver.py         # AEGIS program evolution engine (v2.4)
│   │   ├── causal_dsl_prior.py      # Causal DSL prior learning (v2.4)
│   │   ├── conditional_tree.py      # Conditional tree inducer
│   │   ├── sequence_dsl.py          # Sequence DSL operations
│   │   └── optimized_solver.py      # Optimized solver variant
│   ├── eval/                    # Evaluation framework (v2.4)
│   │   ├── __init__.py             # Package initializer
│   │   ├── arc_agi3_evaluator.py  # RHAE scoring framework
│   │   └── dataset_builder.py     # ARC tasks → dataset JSON
│   ├── api/                     # External API adapters
│   │   ├── deepseek_adapter.py      # DeepSeek API base adapter
│   │   └── deepseek_vl.py           # DeepSeek VL vision adapter
│   ├── verify/                  # Batch verification
│   │   ├── a100_verifier.py         # A100 batch grid verification
│   │   └── a100_video_verifier.py   # A100 batch video verification
│   └── utils/                   # Utilities
│       ├── config.py                # YAML config loader
│       ├── logger.py                # Audit logger
│       ├── gpu_optimizer.py         # GPU memory management
│       ├── kaggle_format.py         # Kaggle I/O format adapter
│       ├── slip_cost.py             # Slip cost calculator
│       └── tensorboard_monitor.py   # TensorBoard monitoring
├── config/                      # Configuration
│   └── default.yaml                # Default configuration (v2.4)
├── data/                        # Sample tasks
├── docker/                      # Docker deployment
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md             # System architecture
│   ├── CHANGELOG.md                # Version history
│   ├── paper/                      # Research paper
│   └── prd/                        # Product requirements docs
│       └── absorptin_prd.md        # v2.4 absorption PRD
├── tests/                       # Test suite (315 tests)
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
└── setup.py                     # Package setup
```

## Test Suite

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_octonion.py -v
python -m pytest tests/test_kappa_snap.py -v
python -m pytest tests/test_cuda_kernels.py -v
python -m pytest tests/test_pruning_optimizer.py -v
```

Test coverage: 290 tests across 8 test modules:
- `test_octonion.py` — Octonion encoding/decoding reversibility
- `test_dsl.py` — DSL primitives and ProgramNode composition
- `test_kappa_snap.py` — κ-Snap Two-Phase search
- `test_gaussex.py` — GaussEx fiber verification
- `test_video.py` — Video temporal encoding and ΔT extraction
- `test_bayesian.py` — Bayesian confidence and Jitter estimation
- `test_cuda_kernels.py` — CUDA GPU kernels (31 tests)
- `test_pruning_optimizer.py` — Pruning pipeline (32 tests)

## Version History

| Version | Date | Key Features |
|---------|------|-------------|
| v1.0 | 2026-06-22 | Complete 5-layer pipeline, 227 tests |
| v2.0 | 2026-06-22 | Luzhao DNA, Betti₀ singularity, VLM diff, ENPV, slip cost, GATlab |
| v2.1 | 2026-06-22 | 14 vectorization optimizations, CRC32 cache, parallel verification |
| v2.2 | 2026-06-22 | Numba @njit JIT compilation (20 kernels), 5.1× grid_equal speedup |
| v2.3 | 2026-06-22 | Numba @cuda.jit GPU kernels, CuPy batch verification, 8-strategy pruning, 290 tests |
| v2.4 | 2026-06-23 | Cross-repo absorption: RHAE eval, ψ-Gate fusion, AEGIS evolver, causal prior, GAT axioms, math sequences, 315 tests |

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — Full system architecture and design decisions
- [Changelog](docs/CHANGELOG.md) — Version history and feature additions
- [Research Paper](docs/paper/TOMAS_ARC_AGI_3_Paper.md) — Principles, technology, and implementation
- [Web Dashboard Guide](docs/web_dashboard_guide.md) — Dashboard installation, API reference, and visualization usage
- [ψ-Gate Configuration Guide](docs/psi_gate_guide.md) — ψ-Gate semantic gating setup, tuning, and 5 core capabilities
- [AEGIS Evolution Engine Guide](docs/aegis_guide.md) — AEGIS 4-stage pipeline (Digester→Planner→Evolver→Critic) configuration and usage

## Repository

- **GitHub**: https://github.com/lisoleg/tomas-arc3-solver
- **Commits**: `51fca1c` (v1.0) → `750c5f1` (v2.0) → `b8a4e1b` (v2.1+v2.2+v2.3)

## License

MIT
