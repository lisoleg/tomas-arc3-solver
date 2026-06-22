# TOMAS ARC-AGI-3 Solver Performance Benchmark Report

**Generated**: 2026-06-23 02:39:07
**Tasks**: 15

## 1. psi-Gate Semantic Gating Comparison

| Metric | psi-Gate Disabled | psi-Gate Enabled | Delta |
|--------|-------------------|------------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 2.3379 | 2.3597 | +0.021800000000000264 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 2. AEGIS Evolution Engine Comparison

| Metric | Normal Search | AEGIS Evolution | Delta |
|--------|--------------|-----------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 2.8518 | 2.8615 | +0.009700000000000042 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 3. Causal DSL Prior Comparison

| Metric | Causal Prior Disabled | Causal Prior Enabled | Delta |
|--------|----------------------|----------------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 2.3756 | 2.4121 | +0.0365000000000002 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 4. Per-Task Details

### psi-Gate Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | psi_gate_disabled | completed | 2.8017 | 0 | 0.0 | False |
| task_001.json | psi_gate_enabled | completed | 3.1419 | 0 | 0.0 | False |
| task_002.json | psi_gate_disabled | completed | 4.357 | 633 | 0.0313 | True |
| task_002.json | psi_gate_enabled | completed | 3.7368 | 633 | 0.0313 | True |
| task_003.json | psi_gate_disabled | completed | 2.6112 | 0 | 0.0 | False |
| task_003.json | psi_gate_enabled | completed | 3.171 | 0 | 0.0 | False |
| task_004.json | psi_gate_disabled | completed | 2.3256 | 0 | 0.0 | False |
| task_004.json | psi_gate_enabled | completed | 1.9747 | 0 | 0.0 | False |
| task_005.json | psi_gate_disabled | completed | 3.0099 | 519 | 0.0316 | True |
| task_005.json | psi_gate_enabled | completed | 3.2246 | 519 | 0.0316 | True |
| task_006.json | psi_gate_disabled | completed | 2.6058 | 0 | 0.0 | False |
| task_006.json | psi_gate_enabled | completed | 2.3872 | 0 | 0.0 | False |
| task_007.json | psi_gate_disabled | completed | 2.2703 | 1244 | 0.0295 | True |
| task_007.json | psi_gate_enabled | completed | 2.1055 | 1244 | 0.0295 | True |
| task_008.json | psi_gate_disabled | completed | 2.0853 | 29 | 0.0731 | True |
| task_008.json | psi_gate_enabled | completed | 2.1607 | 29 | 0.0731 | True |
| task_009.json | psi_gate_disabled | completed | 1.3153 | 0 | 0.0 | False |
| task_009.json | psi_gate_enabled | completed | 1.1308 | 0 | 0.0 | False |
| task_010.json | psi_gate_disabled | completed | 1.776 | 694 | 0.0313 | True |
| task_010.json | psi_gate_enabled | completed | 1.5786 | 694 | 0.0313 | True |
| task_011.json | psi_gate_disabled | completed | 0.9242 | 783 | 0.0313 | True |
| task_011.json | psi_gate_enabled | completed | 1.1032 | 783 | 0.0313 | True |
| task_012.json | psi_gate_disabled | completed | 0.6726 | 0 | 0.0 | False |
| task_012.json | psi_gate_enabled | completed | 0.6588 | 0 | 0.0 | False |
| task_013.json | psi_gate_disabled | completed | 1.8121 | 0 | 0.0 | False |
| task_013.json | psi_gate_enabled | completed | 2.9826 | 0 | 0.0 | False |
| task_014.json | psi_gate_disabled | completed | 3.5331 | 7041 | 0.0217 | True |
| task_014.json | psi_gate_enabled | completed | 3.6851 | 7041 | 0.0217 | True |
| task_015.json | psi_gate_disabled | completed | 2.9688 | 1488 | 0.0295 | True |
| task_015.json | psi_gate_enabled | completed | 2.3544 | 1488 | 0.0295 | True |

### AEGIS Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | aegis_disabled | completed | 2.6423 | 0 | 0.0 | False |
| task_001.json | aegis_enabled | completed | 3.5132 | 0 | 0.0 | False |
| task_002.json | aegis_disabled | completed | 5.0238 | 633 | 0.0313 | True |
| task_002.json | aegis_enabled | completed | 3.8408 | 633 | 0.0313 | True |
| task_003.json | aegis_disabled | completed | 2.1018 | 0 | 0.0 | False |
| task_003.json | aegis_enabled | completed | 1.9001 | 0 | 0.0 | False |
| task_004.json | aegis_disabled | completed | 2.0641 | 0 | 0.0 | False |
| task_004.json | aegis_enabled | completed | 1.7466 | 0 | 0.0 | False |
| task_005.json | aegis_disabled | completed | 3.2972 | 519 | 0.0316 | True |
| task_005.json | aegis_enabled | completed | 3.1344 | 519 | 0.0316 | True |
| task_006.json | aegis_disabled | completed | 4.6291 | 0 | 0.0 | False |
| task_006.json | aegis_enabled | completed | 3.6366 | 0 | 0.0 | False |
| task_007.json | aegis_disabled | completed | 3.0118 | 1244 | 0.0295 | True |
| task_007.json | aegis_enabled | completed | 4.5736 | 1244 | 0.0295 | True |
| task_008.json | aegis_disabled | completed | 3.5479 | 29 | 0.0731 | True |
| task_008.json | aegis_enabled | completed | 4.0987 | 29 | 0.0731 | True |
| task_009.json | aegis_disabled | completed | 1.954 | 0 | 0.0 | False |
| task_009.json | aegis_enabled | completed | 2.2622 | 0 | 0.0 | False |
| task_010.json | aegis_disabled | completed | 2.8945 | 694 | 0.0313 | True |
| task_010.json | aegis_enabled | completed | 2.5441 | 694 | 0.0313 | True |
| task_011.json | aegis_disabled | completed | 1.7131 | 783 | 0.0313 | True |
| task_011.json | aegis_enabled | completed | 1.5888 | 783 | 0.0313 | True |
| task_012.json | aegis_disabled | completed | 1.0616 | 0 | 0.0 | False |
| task_012.json | aegis_enabled | completed | 0.9445 | 0 | 0.0 | False |
| task_013.json | aegis_disabled | completed | 2.1032 | 0 | 0.0 | False |
| task_013.json | aegis_enabled | completed | 2.3074 | 0 | 0.0 | False |
| task_014.json | aegis_disabled | completed | 3.5296 | 7041 | 0.0217 | True |
| task_014.json | aegis_enabled | completed | 3.886 | 7041 | 0.0217 | True |
| task_015.json | aegis_disabled | completed | 3.2032 | 1488 | 0.0295 | True |
| task_015.json | aegis_enabled | completed | 2.9457 | 1488 | 0.0295 | True |

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