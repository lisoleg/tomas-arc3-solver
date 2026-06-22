# TOMAS ARC-AGI-3 Solver

> **TOMAS** (太乙互搏 — Taiyi Mutual-Play) framework for the [ARC-AGI-3](https://www.kaggle.com/competitions/arc-agi-3) video reasoning competition.

## Overview

TOMAS ARC-AGI-3 Solver is an end-to-end reasoning system that combines:

- **Octonion HyperEdge Encoding** — Reversible grid-to-octonion encoding with 8 cognitive-semantic components
- **κ-Snap Abductive Search** — Two-Phase search: topological hash quick-filter → MDL-priority enumeration
- **GaussEx Fiber Verification** — Willems behavioral system fiber intersection for demo constraint satisfaction
- **ΔT Composition** — Chain (⊙), additive (+), and conditional (if-then) temporal transformation composition
- **Bayesian Confidence** — Posterior ranking with MDL-based priors and Gaussian likelihoods
- **Multi-Modal Fusion** — Symbolic + Visual + Cross-modal score fusion with Kaggle degradation
- **Library Learning** — DreamCoder-style subroutine extraction with JSON persistence

## Architecture

```
Layer 1: Input Adaptation     (KaggleFormatAdapter → VideoARCTask)
Layer 2: Encoding              (OctonionHyperEdge → HyperGraph → ΔT extraction)
Layer 3: Search & Verification (κ-Snap Two-Phase → GaussEx fiber verification)
Layer 4: Confidence & Fusion   (Bayesian posterior → Multi-modal fusion scoring)
Layer 5: Output & Optimization (Batch verification → Library Learning → Submission)
```

## Quick Start

### Local

```bash
# Setup environment
bash scripts/setup_env.sh

# Download data
bash scripts/download_data.sh data/

# Solve tasks
python main.py --mode auto --input data/ --output submission.json
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up
```

### Kaggle

Copy `kaggle/notebook_template.ipynb` to Kaggle, upload the solver as a dataset, and run.

## Three Inference Modes

| Mode | Description | Speed | Accuracy |
|------|-------------|-------|----------|
| `video` | Pure symbolic reasoning | Fast | Baseline |
| `bayesian` | Bayesian posterior ranking | Medium | Higher |
| `fusion` | Multi-modal fusion (symbolic + VL) | Slow | Highest |
| `auto` | Time-budget adaptive selection | Adaptive | Optimal |

Auto mode selects based on time budget:
- `< 40s` → video
- `40-80s` → bayesian
- `> 80s` → fusion

## Configuration

All parameters are in `config/default.yaml`. Environment variable overrides use `TOMAS__SECTION__KEY` format.

## Project Structure

```
tomas-arc3-solver/
├── src/
│   ├── core/          # Octonion encoding, hypergraph, DSL, video tensor
│   ├── solver/        # Search, verification, confidence, fusion, main solver
│   ├── api/           # DeepSeek VL API adapters
│   ├── verify/        # A100 batch verification
│   └── utils/         # Config, logger, GPU optimizer, Kaggle format
├── config/            # YAML configuration
├── docker/            # Docker deployment
├── kaggle/            # Kaggle notebook template
├── scripts/           # Setup, download, solve, benchmark scripts
└── main.py            # CLI entry point
```

## License

MIT
