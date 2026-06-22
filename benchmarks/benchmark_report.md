# TOMAS ARC-AGI-3 Solver Performance Benchmark Report

**Generated**: 2026-06-23 01:28:00
**Tasks**: 15

## 1. psi-Gate Semantic Gating Comparison

| Metric | psi-Gate Disabled | psi-Gate Enabled | Delta |
|--------|-------------------|------------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 2.7925 | 2.6428 | -0.14970000000000017 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 2. AEGIS Evolution Engine Comparison

| Metric | Normal Search | AEGIS Evolution | Delta |
|--------|--------------|-----------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 2.6738 | 2.6865 | +0.012700000000000156 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 3. Causal DSL Prior Comparison

| Metric | Causal Prior Disabled | Causal Prior Enabled | Delta |
|--------|----------------------|----------------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 2.4005 | 2.3556 | -0.04490000000000016 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 4. Per-Task Details

### psi-Gate Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | psi_gate_disabled | completed | 5.7901 | 0 | 0.0 | False |
| task_001.json | psi_gate_enabled | completed | 4.3665 | 0 | 0.0 | False |
| task_002.json | psi_gate_disabled | completed | 4.6899 | 633 | 0.0313 | True |
| task_002.json | psi_gate_enabled | completed | 4.6352 | 633 | 0.0313 | True |
| task_003.json | psi_gate_disabled | completed | 2.2422 | 0 | 0.0 | False |
| task_003.json | psi_gate_enabled | completed | 2.1624 | 0 | 0.0 | False |
| task_004.json | psi_gate_disabled | completed | 2.3126 | 0 | 0.0 | False |
| task_004.json | psi_gate_enabled | completed | 2.138 | 0 | 0.0 | False |
| task_005.json | psi_gate_disabled | completed | 3.5335 | 519 | 0.0316 | True |
| task_005.json | psi_gate_enabled | completed | 3.473 | 519 | 0.0316 | True |
| task_006.json | psi_gate_disabled | completed | 3.375 | 0 | 0.0 | False |
| task_006.json | psi_gate_enabled | completed | 3.4593 | 0 | 0.0 | False |
| task_007.json | psi_gate_disabled | completed | 3.1301 | 1244 | 0.0295 | True |
| task_007.json | psi_gate_enabled | completed | 3.1735 | 1244 | 0.0295 | True |
| task_008.json | psi_gate_disabled | completed | 3.2773 | 29 | 0.0731 | True |
| task_008.json | psi_gate_enabled | completed | 3.305 | 29 | 0.0731 | True |
| task_009.json | psi_gate_disabled | completed | 1.9192 | 0 | 0.0 | False |
| task_009.json | psi_gate_enabled | completed | 1.4173 | 0 | 0.0 | False |
| task_010.json | psi_gate_disabled | completed | 2.0731 | 694 | 0.0313 | True |
| task_010.json | psi_gate_enabled | completed | 2.0686 | 694 | 0.0313 | True |
| task_011.json | psi_gate_disabled | completed | 1.1345 | 783 | 0.0313 | True |
| task_011.json | psi_gate_enabled | completed | 0.9783 | 783 | 0.0313 | True |
| task_012.json | psi_gate_disabled | completed | 0.8412 | 0 | 0.0 | False |
| task_012.json | psi_gate_enabled | completed | 0.799 | 0 | 0.0 | False |
| task_013.json | psi_gate_disabled | completed | 2.2382 | 0 | 0.0 | False |
| task_013.json | psi_gate_enabled | completed | 1.7291 | 0 | 0.0 | False |
| task_014.json | psi_gate_disabled | completed | 2.7943 | 7041 | 0.0217 | True |
| task_014.json | psi_gate_enabled | completed | 3.816 | 7041 | 0.0217 | True |
| task_015.json | psi_gate_disabled | completed | 2.5366 | 1488 | 0.0295 | True |
| task_015.json | psi_gate_enabled | completed | 2.121 | 1488 | 0.0295 | True |

### AEGIS Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | aegis_disabled | completed | 3.5617 | 0 | 0.0 | False |
| task_001.json | aegis_enabled | completed | 3.7265 | 0 | 0.0 | False |
| task_002.json | aegis_disabled | completed | 4.1514 | 633 | 0.0313 | True |
| task_002.json | aegis_enabled | completed | 4.8769 | 633 | 0.0313 | True |
| task_003.json | aegis_disabled | completed | 2.5634 | 0 | 0.0 | False |
| task_003.json | aegis_enabled | completed | 2.3272 | 0 | 0.0 | False |
| task_004.json | aegis_disabled | completed | 2.0471 | 0 | 0.0 | False |
| task_004.json | aegis_enabled | completed | 1.6657 | 0 | 0.0 | False |
| task_005.json | aegis_disabled | completed | 4.3712 | 519 | 0.0316 | True |
| task_005.json | aegis_enabled | completed | 5.2199 | 519 | 0.0316 | True |
| task_006.json | aegis_disabled | completed | 4.9711 | 0 | 0.0 | False |
| task_006.json | aegis_enabled | completed | 3.6465 | 0 | 0.0 | False |
| task_007.json | aegis_disabled | completed | 2.9719 | 1244 | 0.0295 | True |
| task_007.json | aegis_enabled | completed | 3.233 | 1244 | 0.0295 | True |
| task_008.json | aegis_disabled | completed | 2.7534 | 29 | 0.0731 | True |
| task_008.json | aegis_enabled | completed | 2.5956 | 29 | 0.0731 | True |
| task_009.json | aegis_disabled | completed | 1.3179 | 0 | 0.0 | False |
| task_009.json | aegis_enabled | completed | 1.2395 | 0 | 0.0 | False |
| task_010.json | aegis_disabled | completed | 1.5584 | 694 | 0.0313 | True |
| task_010.json | aegis_enabled | completed | 1.7096 | 694 | 0.0313 | True |
| task_011.json | aegis_disabled | completed | 1.309 | 783 | 0.0313 | True |
| task_011.json | aegis_enabled | completed | 1.2068 | 783 | 0.0313 | True |
| task_012.json | aegis_disabled | completed | 0.8548 | 0 | 0.0 | False |
| task_012.json | aegis_enabled | completed | 0.7275 | 0 | 0.0 | False |
| task_013.json | aegis_disabled | completed | 1.8249 | 0 | 0.0 | False |
| task_013.json | aegis_enabled | completed | 1.7937 | 0 | 0.0 | False |
| task_014.json | aegis_disabled | completed | 3.4969 | 7041 | 0.0217 | True |
| task_014.json | aegis_enabled | completed | 4.5179 | 7041 | 0.0217 | True |
| task_015.json | aegis_disabled | completed | 2.3544 | 1488 | 0.0295 | True |
| task_015.json | aegis_enabled | completed | 1.8117 | 1488 | 0.0295 | True |

## 5. Pruning Statistics

**Task**: task_001.json | **Config**: psi_gate_disabled

| Strategy | Count Pruned |
|----------|-------------|
| betti0_pruned | 0 |
| symmetry_deduped | 1710 |
| mdl_pruned | 0 |
| shape_pruned | 7010 |
| color_hist_pruned | 2141 |
| nonzero_pruned | 10765 |

## 6. Conclusions


---
*Report generated by TOMAS benchmark suite*