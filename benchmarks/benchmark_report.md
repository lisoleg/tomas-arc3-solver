# TOMAS ARC-AGI-3 Solver Performance Benchmark Report

**Generated**: 2026-06-23 01:52:08
**Tasks**: 15

## 1. psi-Gate Semantic Gating Comparison

| Metric | psi-Gate Disabled | psi-Gate Enabled | Delta |
|--------|-------------------|------------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 7.0933 | 7.4752 | +0.3818999999999999 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 2. AEGIS Evolution Engine Comparison

| Metric | Normal Search | AEGIS Evolution | Delta |
|--------|--------------|-----------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 6.3545 | 6.3168 | -0.03770000000000007 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 3. Causal DSL Prior Comparison

| Metric | Causal Prior Disabled | Causal Prior Enabled | Delta |
|--------|----------------------|----------------------|-------|
| Tasks Completed | 15 | 15 | 0 |
| Correct | 8 | 8 | 0 |
| Accuracy (%) | 53.3 | 53.3 | +0.0 |
| Avg Time (s) | 5.4727 | 4.8557 | -0.617 |
| Avg Candidates | 828.7 | 828.7 | +0.0 |
| Avg Confidence | 0.0186 | 0.0186 | +0.0 |
| Avg MDL | 3.8 | 3.8 | +0.0 |

## 4. Per-Task Details

### psi-Gate Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | psi_gate_disabled | completed | 5.6823 | 0 | 0.0 | False |
| task_001.json | psi_gate_enabled | completed | 12.897 | 0 | 0.0 | False |
| task_002.json | psi_gate_disabled | completed | 12.9829 | 633 | 0.0313 | True |
| task_002.json | psi_gate_enabled | completed | 12.0303 | 633 | 0.0313 | True |
| task_003.json | psi_gate_disabled | completed | 5.8863 | 0 | 0.0 | False |
| task_003.json | psi_gate_enabled | completed | 5.7164 | 0 | 0.0 | False |
| task_004.json | psi_gate_disabled | completed | 6.1167 | 0 | 0.0 | False |
| task_004.json | psi_gate_enabled | completed | 6.2041 | 0 | 0.0 | False |
| task_005.json | psi_gate_disabled | completed | 9.6748 | 519 | 0.0316 | True |
| task_005.json | psi_gate_enabled | completed | 9.0582 | 519 | 0.0316 | True |
| task_006.json | psi_gate_disabled | completed | 8.3139 | 0 | 0.0 | False |
| task_006.json | psi_gate_enabled | completed | 8.5327 | 0 | 0.0 | False |
| task_007.json | psi_gate_disabled | completed | 7.8982 | 1244 | 0.0295 | True |
| task_007.json | psi_gate_enabled | completed | 7.5046 | 1244 | 0.0295 | True |
| task_008.json | psi_gate_disabled | completed | 7.7852 | 29 | 0.0731 | True |
| task_008.json | psi_gate_enabled | completed | 7.7914 | 29 | 0.0731 | True |
| task_009.json | psi_gate_disabled | completed | 4.8476 | 0 | 0.0 | False |
| task_009.json | psi_gate_enabled | completed | 4.7989 | 0 | 0.0 | False |
| task_010.json | psi_gate_disabled | completed | 6.782 | 694 | 0.0313 | True |
| task_010.json | psi_gate_enabled | completed | 6.1454 | 694 | 0.0313 | True |
| task_011.json | psi_gate_disabled | completed | 3.5659 | 783 | 0.0313 | True |
| task_011.json | psi_gate_enabled | completed | 3.771 | 783 | 0.0313 | True |
| task_012.json | psi_gate_disabled | completed | 3.3295 | 0 | 0.0 | False |
| task_012.json | psi_gate_enabled | completed | 2.9121 | 0 | 0.0 | False |
| task_013.json | psi_gate_disabled | completed | 6.8243 | 0 | 0.0 | False |
| task_013.json | psi_gate_enabled | completed | 6.8585 | 0 | 0.0 | False |
| task_014.json | psi_gate_disabled | completed | 8.9022 | 7041 | 0.0217 | True |
| task_014.json | psi_gate_enabled | completed | 9.8288 | 7041 | 0.0217 | True |
| task_015.json | psi_gate_disabled | completed | 7.8079 | 1488 | 0.0295 | True |
| task_015.json | psi_gate_enabled | completed | 8.0785 | 1488 | 0.0295 | True |

### AEGIS Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | aegis_disabled | completed | 12.3142 | 0 | 0.0 | False |
| task_001.json | aegis_enabled | completed | 11.8856 | 0 | 0.0 | False |
| task_002.json | aegis_disabled | completed | 11.7559 | 633 | 0.0313 | True |
| task_002.json | aegis_enabled | completed | 13.511 | 633 | 0.0313 | True |
| task_003.json | aegis_disabled | completed | 6.345 | 0 | 0.0 | False |
| task_003.json | aegis_enabled | completed | 6.2635 | 0 | 0.0 | False |
| task_004.json | aegis_disabled | completed | 5.9971 | 0 | 0.0 | False |
| task_004.json | aegis_enabled | completed | 5.8176 | 0 | 0.0 | False |
| task_005.json | aegis_disabled | completed | 9.8221 | 519 | 0.0316 | True |
| task_005.json | aegis_enabled | completed | 9.4975 | 519 | 0.0316 | True |
| task_006.json | aegis_disabled | completed | 8.7183 | 0 | 0.0 | False |
| task_006.json | aegis_enabled | completed | 8.7768 | 0 | 0.0 | False |
| task_007.json | aegis_disabled | completed | 7.6954 | 1244 | 0.0295 | True |
| task_007.json | aegis_enabled | completed | 7.9507 | 1244 | 0.0295 | True |
| task_008.json | aegis_disabled | completed | 8.055 | 29 | 0.0731 | True |
| task_008.json | aegis_enabled | completed | 8.9134 | 29 | 0.0731 | True |
| task_009.json | aegis_disabled | completed | 5.6404 | 0 | 0.0 | False |
| task_009.json | aegis_enabled | completed | 4.8471 | 0 | 0.0 | False |
| task_010.json | aegis_disabled | completed | 6.9142 | 694 | 0.0313 | True |
| task_010.json | aegis_enabled | completed | 7.2647 | 694 | 0.0313 | True |
| task_011.json | aegis_disabled | completed | 3.6961 | 783 | 0.0313 | True |
| task_011.json | aegis_enabled | completed | 1.9966 | 783 | 0.0313 | True |
| task_012.json | aegis_disabled | completed | 1.0089 | 0 | 0.0 | False |
| task_012.json | aegis_enabled | completed | 0.6983 | 0 | 0.0 | False |
| task_013.json | aegis_disabled | completed | 1.7721 | 0 | 0.0 | False |
| task_013.json | aegis_enabled | completed | 1.714 | 0 | 0.0 | False |
| task_014.json | aegis_disabled | completed | 3.4371 | 7041 | 0.0217 | True |
| task_014.json | aegis_enabled | completed | 3.6338 | 7041 | 0.0217 | True |
| task_015.json | aegis_disabled | completed | 2.1459 | 1488 | 0.0295 | True |
| task_015.json | aegis_enabled | completed | 1.9819 | 1488 | 0.0295 | True |

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