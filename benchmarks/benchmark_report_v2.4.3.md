# TOMAS ARC-AGI-3 Solver Performance Benchmark Report

**Generated**: 2026-06-23 03:27:00
**Tasks**: 15

## 1. psi-Gate Semantic Gating Comparison

| Metric | psi-Gate Disabled | psi-Gate Enabled | Delta |
|--------|-------------------|------------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 6 | 6 | 0 |
| Accuracy (%) | 40.0 | 40.0 | +0.0 |
| Avg Time (s) | 3.5583 | 3.1675 | -0.39080000000000004 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.5333 | 0.0 | -0.5333 |
| Avg MDL | 6.1 | 6.1 | +0.0 |

## 2. AEGIS Evolution Engine Comparison

| Metric | Normal Search | AEGIS Evolution | Delta |
|--------|--------------|-----------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 6 | 6 | 0 |
| Accuracy (%) | 40.0 | 40.0 | +0.0 |
| Avg Time (s) | 3.4908 | 3.4325 | -0.05830000000000002 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0 | 0.0 | +0.0 |
| Avg MDL | 6.1 | 6.1 | +0.0 |

## 3. Causal DSL Prior Comparison

| Metric | Causal Prior Disabled | Causal Prior Enabled | Delta |
|--------|----------------------|----------------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 6 | 6 | 0 |
| Accuracy (%) | 40.0 | 40.0 | +0.0 |
| Avg Time (s) | 2.9188 | 3.0394 | +0.12060000000000004 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0 | 0.0 | +0.0 |
| Avg MDL | 6.1 | 6.1 | +0.0 |

## 4. Per-Task Details

### psi-Gate Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | psi_gate_disabled | completed | 5.7112 | 0 | 0.0 | False |
| task_001.json | psi_gate_enabled | completed | 6.1009 | 0 | 0.0 | False |
| task_002.json | psi_gate_disabled | completed | 5.7162 | 633 | 1.0 | False |
| task_002.json | psi_gate_enabled | completed | 6.006 | 633 | 0.0 | False |
| task_003.json | psi_gate_disabled | completed | 2.3977 | 0 | 0.0 | False |
| task_003.json | psi_gate_enabled | completed | 2.1187 | 0 | 0.0 | False |
| task_004.json | psi_gate_disabled | completed | 2.1214 | 0 | 0.0 | False |
| task_004.json | psi_gate_enabled | completed | 2.015 | 0 | 0.0 | False |
| task_005.json | psi_gate_disabled | completed | 3.3321 | 519 | 1.0 | False |
| task_005.json | psi_gate_enabled | completed | 3.282 | 519 | 0.0 | False |
| task_006.json | psi_gate_disabled | completed | 3.3683 | 0 | 0.0 | False |
| task_006.json | psi_gate_enabled | completed | 3.0395 | 0 | 0.0 | False |
| task_007.json | psi_gate_disabled | completed | 2.8429 | 1244 | 1.0 | True |
| task_007.json | psi_gate_enabled | completed | 2.9499 | 1244 | 0.0 | True |
| task_008.json | psi_gate_disabled | completed | 2.839 | 29 | 1.0 | True |
| task_008.json | psi_gate_enabled | completed | 3.4143 | 29 | 0.0 | True |
| task_009.json | psi_gate_disabled | completed | 2.065 | 0 | 0.0 | False |
| task_009.json | psi_gate_enabled | completed | 1.6129 | 0 | 0.0 | False |
| task_010.json | psi_gate_disabled | completed | 2.1003 | 694 | 1.0 | True |
| task_010.json | psi_gate_enabled | completed | 2.1079 | 694 | 0.0 | True |
| task_011.json | psi_gate_disabled | completed | 1.5229 | 783 | 1.0 | True |
| task_011.json | psi_gate_enabled | completed | 1.6135 | 783 | 0.0 | True |
| task_012.json | psi_gate_disabled | completed | 1.2236 | 0 | 0.0 | False |
| task_012.json | psi_gate_enabled | completed | 1.0345 | 0 | 0.0 | False |
| task_013.json | psi_gate_disabled | completed | 3.1028 | 0 | 0.0 | False |
| task_013.json | psi_gate_enabled | completed | 2.7282 | 0 | 0.0 | False |
| task_014.json | psi_gate_disabled | completed | 10.9534 | 7041 | 1.0 | True |
| task_014.json | psi_gate_enabled | completed | 5.4518 | 7041 | 0.0 | True |
| task_015.json | psi_gate_disabled | completed | 4.0781 | 1488 | 1.0 | True |
| task_015.json | psi_gate_enabled | completed | 4.038 | 1488 | 0.0 | True |

### AEGIS Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | aegis_disabled | completed | 6.2868 | 0 | 0.0 | False |
| task_001.json | aegis_enabled | completed | 6.9833 | 0 | 0.0 | False |
| task_002.json | aegis_disabled | completed | 7.7175 | 633 | 0.0 | False |
| task_002.json | aegis_enabled | completed | 5.3812 | 633 | 0.0 | False |
| task_003.json | aegis_disabled | completed | 3.1447 | 0 | 0.0 | False |
| task_003.json | aegis_enabled | completed | 2.6753 | 0 | 0.0 | False |
| task_004.json | aegis_disabled | completed | 2.7499 | 0 | 0.0 | False |
| task_004.json | aegis_enabled | completed | 3.2321 | 0 | 0.0 | False |
| task_005.json | aegis_disabled | completed | 3.8869 | 519 | 0.0 | False |
| task_005.json | aegis_enabled | completed | 3.6864 | 519 | 0.0 | False |
| task_006.json | aegis_disabled | completed | 3.4008 | 0 | 0.0 | False |
| task_006.json | aegis_enabled | completed | 4.72 | 0 | 0.0 | False |
| task_007.json | aegis_disabled | completed | 3.0477 | 1244 | 0.0 | True |
| task_007.json | aegis_enabled | completed | 3.5959 | 1244 | 0.0 | True |
| task_008.json | aegis_disabled | completed | 5.0582 | 29 | 0.0 | True |
| task_008.json | aegis_enabled | completed | 4.4243 | 29 | 0.0 | True |
| task_009.json | aegis_disabled | completed | 1.5934 | 0 | 0.0 | False |
| task_009.json | aegis_enabled | completed | 1.3563 | 0 | 0.0 | False |
| task_010.json | aegis_disabled | completed | 1.8966 | 694 | 0.0 | True |
| task_010.json | aegis_enabled | completed | 2.3504 | 694 | 0.0 | True |
| task_011.json | aegis_disabled | completed | 1.5487 | 783 | 0.0 | True |
| task_011.json | aegis_enabled | completed | 1.7072 | 783 | 0.0 | True |
| task_012.json | aegis_disabled | completed | 1.074 | 0 | 0.0 | False |
| task_012.json | aegis_enabled | completed | 1.2323 | 0 | 0.0 | False |
| task_013.json | aegis_disabled | completed | 2.9298 | 0 | 0.0 | False |
| task_013.json | aegis_enabled | completed | 3.0942 | 0 | 0.0 | False |
| task_014.json | aegis_disabled | completed | 4.5349 | 7041 | 0.0 | True |
| task_014.json | aegis_enabled | completed | 4.0149 | 7041 | 0.0 | True |
| task_015.json | aegis_disabled | completed | 3.4923 | 1488 | 0.0 | True |
| task_015.json | aegis_enabled | completed | 3.0334 | 1488 | 0.0 | True |

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