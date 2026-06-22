# TOMAS ARC-AGI-3 Solver Performance Benchmark Report

**Generated**: 2026-06-23 03:45:01
**Tasks**: 65

## 1. psi-Gate Semantic Gating Comparison

| Metric | psi-Gate Disabled | psi-Gate Enabled | Delta |
|--------|-------------------|------------------|-------|
| Tasks Completed | 65 | 65 | 0 |
| Correct | 56 | 56 | 0 |
| Accuracy (%) | 86.2 | 86.2 | +0.0 |
| Avg Time (s) | 4.4512 | 4.3073 | -0.14390000000000036 |
| Avg Candidates | 648.3 | 648.3 | +0.0 |
| Avg Confidence | 0.8923 | 0.0 | -0.8923 |
| Avg MDL | 5.3 | 5.3 | +0.0 |

## 2. AEGIS Evolution Engine Comparison

| Metric | Normal Search | AEGIS Evolution | Delta |
|--------|--------------|-----------------|-------|
| Tasks Completed | 65 | 65 | 0 |
| Correct | 56 | 56 | 0 |
| Accuracy (%) | 86.2 | 86.2 | +0.0 |
| Avg Time (s) | 4.0976 | 3.9639 | -0.1336999999999997 |
| Avg Candidates | 648.3 | 648.3 | +0.0 |
| Avg Confidence | 0.0 | 0.0 | +0.0 |
| Avg MDL | 5.3 | 5.3 | +0.0 |

## 3. Causal DSL Prior Comparison

| Metric | Causal Prior Disabled | Causal Prior Enabled | Delta |
|--------|----------------------|----------------------|-------|
| Tasks Completed | 65 | 65 | 0 |
| Correct | 56 | 56 | 0 |
| Accuracy (%) | 86.2 | 86.2 | +0.0 |
| Avg Time (s) | 4.1988 | 4.2693 | +0.07050000000000001 |
| Avg Candidates | 648.3 | 648.3 | +0.0 |
| Avg Confidence | 0.0 | 0.0 | +0.0 |
| Avg MDL | 5.3 | 5.3 | +0.0 |

## 4. Per-Task Details

### psi-Gate Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | psi_gate_disabled | completed | 4.9109 | 0 | 0.0 | False |
| task_001.json | psi_gate_enabled | completed | 4.9036 | 0 | 0.0 | False |
| task_002.json | psi_gate_disabled | completed | 5.3932 | 633 | 1.0 | False |
| task_002.json | psi_gate_enabled | completed | 5.0796 | 633 | 0.0 | False |
| task_003.json | psi_gate_disabled | completed | 2.124 | 0 | 0.0 | False |
| task_003.json | psi_gate_enabled | completed | 2.1045 | 0 | 0.0 | False |
| task_004.json | psi_gate_disabled | completed | 1.8768 | 0 | 0.0 | False |
| task_004.json | psi_gate_enabled | completed | 2.3861 | 0 | 0.0 | False |
| task_005.json | psi_gate_disabled | completed | 4.0008 | 519 | 1.0 | False |
| task_005.json | psi_gate_enabled | completed | 3.4955 | 519 | 0.0 | False |
| task_006.json | psi_gate_disabled | completed | 3.3942 | 0 | 0.0 | False |
| task_006.json | psi_gate_enabled | completed | 4.1197 | 0 | 0.0 | False |
| task_007.json | psi_gate_disabled | completed | 3.3155 | 1244 | 1.0 | True |
| task_007.json | psi_gate_enabled | completed | 3.2439 | 1244 | 0.0 | True |
| task_008.json | psi_gate_disabled | completed | 3.8557 | 29 | 1.0 | True |
| task_008.json | psi_gate_enabled | completed | 3.6388 | 29 | 0.0 | True |
| task_009.json | psi_gate_disabled | completed | 1.9309 | 0 | 0.0 | False |
| task_009.json | psi_gate_enabled | completed | 1.9305 | 0 | 0.0 | False |
| task_010.json | psi_gate_disabled | completed | 2.6726 | 694 | 1.0 | True |
| task_010.json | psi_gate_enabled | completed | 2.6289 | 694 | 0.0 | True |
| task_011.json | psi_gate_disabled | completed | 1.7246 | 783 | 1.0 | True |
| task_011.json | psi_gate_enabled | completed | 1.7562 | 783 | 0.0 | True |
| task_012.json | psi_gate_disabled | completed | 1.3179 | 0 | 0.0 | False |
| task_012.json | psi_gate_enabled | completed | 1.1233 | 0 | 0.0 | False |
| task_013.json | psi_gate_disabled | completed | 2.5082 | 0 | 0.0 | False |
| task_013.json | psi_gate_enabled | completed | 2.633 | 0 | 0.0 | False |
| task_014.json | psi_gate_disabled | completed | 4.099 | 7041 | 1.0 | True |
| task_014.json | psi_gate_enabled | completed | 4.1875 | 7041 | 0.0 | True |
| task_015.json | psi_gate_disabled | completed | 5.0303 | 1488 | 1.0 | True |
| task_015.json | psi_gate_enabled | completed | 3.9142 | 1488 | 0.0 | True |
| task_016.json | psi_gate_disabled | completed | 3.7713 | 605 | 1.0 | True |
| task_016.json | psi_gate_enabled | completed | 4.3625 | 605 | 0.0 | True |
| task_017.json | psi_gate_disabled | completed | 4.2365 | 586 | 1.0 | True |
| task_017.json | psi_gate_enabled | completed | 3.7057 | 586 | 0.0 | True |
| task_018.json | psi_gate_disabled | completed | 3.9229 | 575 | 1.0 | True |
| task_018.json | psi_gate_enabled | completed | 3.7808 | 575 | 0.0 | True |
| task_019.json | psi_gate_disabled | completed | 3.2164 | 586 | 1.0 | True |
| task_019.json | psi_gate_enabled | completed | 3.2164 | 586 | 0.0 | True |
| task_020.json | psi_gate_disabled | completed | 3.8228 | 593 | 1.0 | True |
| task_020.json | psi_gate_enabled | completed | 5.0949 | 593 | 0.0 | True |
| task_021.json | psi_gate_disabled | completed | 11.0312 | 604 | 1.0 | True |
| task_021.json | psi_gate_enabled | completed | 8.5705 | 604 | 0.0 | True |
| task_022.json | psi_gate_disabled | completed | 5.0252 | 605 | 1.0 | True |
| task_022.json | psi_gate_enabled | completed | 4.1698 | 605 | 0.0 | True |
| task_023.json | psi_gate_disabled | completed | 5.009 | 617 | 1.0 | True |
| task_023.json | psi_gate_enabled | completed | 4.7938 | 617 | 0.0 | True |
| task_024.json | psi_gate_disabled | completed | 5.4242 | 575 | 1.0 | True |
| task_024.json | psi_gate_enabled | completed | 4.4963 | 575 | 0.0 | True |
| task_025.json | psi_gate_disabled | completed | 3.947 | 586 | 1.0 | True |
| task_025.json | psi_gate_enabled | completed | 4.6748 | 586 | 0.0 | True |
| task_026.json | psi_gate_disabled | completed | 4.0163 | 593 | 1.0 | True |
| task_026.json | psi_gate_enabled | completed | 4.035 | 593 | 0.0 | True |
| task_027.json | psi_gate_disabled | completed | 4.7989 | 586 | 1.0 | True |
| task_027.json | psi_gate_enabled | completed | 5.4589 | 586 | 0.0 | True |
| task_028.json | psi_gate_disabled | completed | 6.3246 | 605 | 1.0 | True |
| task_028.json | psi_gate_enabled | completed | 5.7266 | 605 | 0.0 | True |
| task_029.json | psi_gate_disabled | completed | 4.7524 | 617 | 1.0 | True |
| task_029.json | psi_gate_enabled | completed | 4.8666 | 617 | 0.0 | True |
| task_030.json | psi_gate_disabled | completed | 5.396 | 575 | 1.0 | True |
| task_030.json | psi_gate_enabled | completed | 5.6304 | 575 | 0.0 | True |
| task_031.json | psi_gate_disabled | completed | 6.3436 | 586 | 1.0 | True |
| task_031.json | psi_gate_enabled | completed | 5.3 | 586 | 0.0 | True |
| task_032.json | psi_gate_disabled | completed | 3.4303 | 575 | 1.0 | True |
| task_032.json | psi_gate_enabled | completed | 3.8797 | 575 | 0.0 | True |
| task_033.json | psi_gate_disabled | completed | 3.9853 | 604 | 1.0 | True |
| task_033.json | psi_gate_enabled | completed | 4.6424 | 604 | 0.0 | True |
| task_034.json | psi_gate_disabled | completed | 5.0302 | 593 | 1.0 | True |
| task_034.json | psi_gate_enabled | completed | 5.126 | 593 | 0.0 | True |
| task_035.json | psi_gate_disabled | completed | 4.7765 | 626 | 1.0 | True |
| task_035.json | psi_gate_enabled | completed | 5.1321 | 626 | 0.0 | True |
| task_036.json | psi_gate_disabled | completed | 4.0413 | 575 | 1.0 | True |
| task_036.json | psi_gate_enabled | completed | 3.3703 | 575 | 0.0 | True |
| task_037.json | psi_gate_disabled | completed | 3.5575 | 586 | 1.0 | True |
| task_037.json | psi_gate_enabled | completed | 3.3326 | 586 | 0.0 | True |
| task_038.json | psi_gate_disabled | completed | 4.1977 | 575 | 1.0 | True |
| task_038.json | psi_gate_enabled | completed | 4.2385 | 575 | 0.0 | True |
| task_039.json | psi_gate_disabled | completed | 2.9799 | 586 | 1.0 | True |
| task_039.json | psi_gate_enabled | completed | 3.3813 | 586 | 0.0 | True |
| task_040.json | psi_gate_disabled | completed | 4.7442 | 621 | 1.0 | True |
| task_040.json | psi_gate_enabled | completed | 6.0389 | 621 | 0.0 | True |
| task_041.json | psi_gate_disabled | completed | 3.183 | 617 | 1.0 | True |
| task_041.json | psi_gate_enabled | completed | 3.6658 | 617 | 0.0 | True |
| task_042.json | psi_gate_disabled | completed | 3.7303 | 575 | 1.0 | True |
| task_042.json | psi_gate_enabled | completed | 3.9928 | 575 | 0.0 | True |
| task_043.json | psi_gate_disabled | completed | 10.2681 | 586 | 1.0 | True |
| task_043.json | psi_gate_enabled | completed | 5.8331 | 586 | 0.0 | True |
| task_044.json | psi_gate_disabled | completed | 5.7145 | 605 | 1.0 | True |
| task_044.json | psi_gate_enabled | completed | 5.1968 | 605 | 0.0 | True |
| task_045.json | psi_gate_disabled | completed | 5.1523 | 633 | 1.0 | True |
| task_045.json | psi_gate_enabled | completed | 5.5155 | 633 | 0.0 | True |
| task_046.json | psi_gate_disabled | completed | 8.7818 | 593 | 1.0 | True |
| task_046.json | psi_gate_enabled | completed | 5.5474 | 593 | 0.0 | True |
| task_047.json | psi_gate_disabled | completed | 5.8643 | 617 | 1.0 | True |
| task_047.json | psi_gate_enabled | completed | 6.0724 | 617 | 0.0 | True |
| task_048.json | psi_gate_disabled | completed | 3.8115 | 575 | 1.0 | True |
| task_048.json | psi_gate_enabled | completed | 3.4462 | 575 | 0.0 | True |
| task_049.json | psi_gate_disabled | completed | 3.5752 | 586 | 1.0 | True |
| task_049.json | psi_gate_enabled | completed | 4.5663 | 586 | 0.0 | True |
| task_050.json | psi_gate_disabled | completed | 4.0021 | 593 | 1.0 | True |
| task_050.json | psi_gate_enabled | completed | 5.5059 | 593 | 0.0 | True |
| task_051.json | psi_gate_disabled | completed | 6.7821 | 586 | 1.0 | True |
| task_051.json | psi_gate_enabled | completed | 4.185 | 586 | 0.0 | True |
| task_052.json | psi_gate_disabled | completed | 5.2053 | 575 | 1.0 | True |
| task_052.json | psi_gate_enabled | completed | 5.5527 | 575 | 0.0 | True |
| task_053.json | psi_gate_disabled | completed | 5.4545 | 617 | 1.0 | True |
| task_053.json | psi_gate_enabled | completed | 5.1621 | 617 | 0.0 | True |
| task_054.json | psi_gate_disabled | completed | 5.7173 | 575 | 1.0 | True |
| task_054.json | psi_gate_enabled | completed | 5.0406 | 575 | 0.0 | True |
| task_055.json | psi_gate_disabled | completed | 3.9154 | 586 | 1.0 | True |
| task_055.json | psi_gate_enabled | completed | 3.5976 | 586 | 0.0 | True |
| task_056.json | psi_gate_disabled | completed | 3.8287 | 593 | 1.0 | True |
| task_056.json | psi_gate_enabled | completed | 4.0424 | 593 | 0.0 | True |
| task_057.json | psi_gate_disabled | completed | 3.562 | 604 | 1.0 | True |
| task_057.json | psi_gate_enabled | completed | 3.5495 | 604 | 0.0 | True |
| task_058.json | psi_gate_disabled | completed | 3.7893 | 593 | 1.0 | True |
| task_058.json | psi_gate_enabled | completed | 3.6049 | 593 | 0.0 | True |
| task_059.json | psi_gate_disabled | completed | 5.1733 | 586 | 1.0 | True |
| task_059.json | psi_gate_enabled | completed | 5.0884 | 586 | 0.0 | True |
| task_060.json | psi_gate_disabled | completed | 3.0129 | 593 | 1.0 | True |
| task_060.json | psi_gate_enabled | completed | 3.2769 | 593 | 0.0 | True |
| task_061.json | psi_gate_disabled | completed | 5.7071 | 617 | 1.0 | True |
| task_061.json | psi_gate_enabled | completed | 5.2125 | 617 | 0.0 | True |
| task_062.json | psi_gate_disabled | completed | 3.8837 | 593 | 1.0 | True |
| task_062.json | psi_gate_enabled | completed | 4.6969 | 593 | 0.0 | True |
| task_063.json | psi_gate_disabled | completed | 4.8729 | 586 | 1.0 | True |
| task_063.json | psi_gate_enabled | completed | 4.3727 | 586 | 0.0 | True |
| task_064.json | psi_gate_disabled | completed | 3.8298 | 605 | 1.0 | True |
| task_064.json | psi_gate_enabled | completed | 5.29 | 605 | 0.0 | True |
| task_065.json | psi_gate_disabled | completed | 4.5758 | 586 | 1.0 | True |
| task_065.json | psi_gate_enabled | completed | 3.7877 | 586 | 0.0 | True |

### AEGIS Benchmark

| Task | Config | Status | Time (s) | Candidates | Confidence | Correct |
|------|--------|--------|----------|------------|------------|---------|
| task_001.json | aegis_disabled | completed | 3.71 | 0 | 0.0 | False |
| task_001.json | aegis_enabled | completed | 3.9346 | 0 | 0.0 | False |
| task_002.json | aegis_disabled | completed | 3.6658 | 633 | 0.0 | False |
| task_002.json | aegis_enabled | completed | 3.9094 | 633 | 0.0 | False |
| task_003.json | aegis_disabled | completed | 1.9043 | 0 | 0.0 | False |
| task_003.json | aegis_enabled | completed | 1.8871 | 0 | 0.0 | False |
| task_004.json | aegis_disabled | completed | 1.7872 | 0 | 0.0 | False |
| task_004.json | aegis_enabled | completed | 1.9655 | 0 | 0.0 | False |
| task_005.json | aegis_disabled | completed | 3.0101 | 519 | 0.0 | False |
| task_005.json | aegis_enabled | completed | 2.9966 | 519 | 0.0 | False |
| task_006.json | aegis_disabled | completed | 3.1414 | 0 | 0.0 | False |
| task_006.json | aegis_enabled | completed | 3.1589 | 0 | 0.0 | False |
| task_007.json | aegis_disabled | completed | 3.4232 | 1244 | 0.0 | True |
| task_007.json | aegis_enabled | completed | 2.8188 | 1244 | 0.0 | True |
| task_008.json | aegis_disabled | completed | 2.7352 | 29 | 0.0 | True |
| task_008.json | aegis_enabled | completed | 2.8317 | 29 | 0.0 | True |
| task_009.json | aegis_disabled | completed | 1.7907 | 0 | 0.0 | False |
| task_009.json | aegis_enabled | completed | 1.99 | 0 | 0.0 | False |
| task_010.json | aegis_disabled | completed | 2.4709 | 694 | 0.0 | True |
| task_010.json | aegis_enabled | completed | 1.8645 | 694 | 0.0 | True |
| task_011.json | aegis_disabled | completed | 1.2891 | 783 | 0.0 | True |
| task_011.json | aegis_enabled | completed | 1.5726 | 783 | 0.0 | True |
| task_012.json | aegis_disabled | completed | 1.4262 | 0 | 0.0 | False |
| task_012.json | aegis_enabled | completed | 2.0418 | 0 | 0.0 | False |
| task_013.json | aegis_disabled | completed | 4.7279 | 0 | 0.0 | False |
| task_013.json | aegis_enabled | completed | 3.2981 | 0 | 0.0 | False |
| task_014.json | aegis_disabled | completed | 4.9883 | 7041 | 0.0 | True |
| task_014.json | aegis_enabled | completed | 3.8335 | 7041 | 0.0 | True |
| task_015.json | aegis_disabled | completed | 3.9457 | 1488 | 0.0 | True |
| task_015.json | aegis_enabled | completed | 4.3701 | 1488 | 0.0 | True |
| task_016.json | aegis_disabled | completed | 5.1641 | 605 | 0.0 | True |
| task_016.json | aegis_enabled | completed | 5.6368 | 605 | 0.0 | True |
| task_017.json | aegis_disabled | completed | 4.1917 | 586 | 0.0 | True |
| task_017.json | aegis_enabled | completed | 4.3736 | 586 | 0.0 | True |
| task_018.json | aegis_disabled | completed | 5.468 | 575 | 0.0 | True |
| task_018.json | aegis_enabled | completed | 5.2102 | 575 | 0.0 | True |
| task_019.json | aegis_disabled | completed | 4.4702 | 586 | 0.0 | True |
| task_019.json | aegis_enabled | completed | 4.2117 | 586 | 0.0 | True |
| task_020.json | aegis_disabled | completed | 5.4232 | 593 | 0.0 | True |
| task_020.json | aegis_enabled | completed | 4.0202 | 593 | 0.0 | True |
| task_021.json | aegis_disabled | completed | 4.9723 | 604 | 0.0 | True |
| task_021.json | aegis_enabled | completed | 4.3458 | 604 | 0.0 | True |
| task_022.json | aegis_disabled | completed | 3.1135 | 605 | 0.0 | True |
| task_022.json | aegis_enabled | completed | 3.555 | 605 | 0.0 | True |
| task_023.json | aegis_disabled | completed | 3.9942 | 617 | 0.0 | True |
| task_023.json | aegis_enabled | completed | 4.4323 | 617 | 0.0 | True |
| task_024.json | aegis_disabled | completed | 4.7054 | 575 | 0.0 | True |
| task_024.json | aegis_enabled | completed | 4.9856 | 575 | 0.0 | True |
| task_025.json | aegis_disabled | completed | 5.0408 | 586 | 0.0 | True |
| task_025.json | aegis_enabled | completed | 4.4878 | 586 | 0.0 | True |
| task_026.json | aegis_disabled | completed | 4.4879 | 593 | 0.0 | True |
| task_026.json | aegis_enabled | completed | 4.6144 | 593 | 0.0 | True |
| task_027.json | aegis_disabled | completed | 3.9751 | 586 | 0.0 | True |
| task_027.json | aegis_enabled | completed | 4.139 | 586 | 0.0 | True |
| task_028.json | aegis_disabled | completed | 4.4136 | 605 | 0.0 | True |
| task_028.json | aegis_enabled | completed | 4.2818 | 605 | 0.0 | True |
| task_029.json | aegis_disabled | completed | 3.5859 | 617 | 0.0 | True |
| task_029.json | aegis_enabled | completed | 5.9508 | 617 | 0.0 | True |
| task_030.json | aegis_disabled | completed | 4.9747 | 575 | 0.0 | True |
| task_030.json | aegis_enabled | completed | 4.2607 | 575 | 0.0 | True |
| task_031.json | aegis_disabled | completed | 4.6284 | 586 | 0.0 | True |
| task_031.json | aegis_enabled | completed | 4.8185 | 586 | 0.0 | True |
| task_032.json | aegis_disabled | completed | 3.1818 | 575 | 0.0 | True |
| task_032.json | aegis_enabled | completed | 4.0526 | 575 | 0.0 | True |
| task_033.json | aegis_disabled | completed | 4.0468 | 604 | 0.0 | True |
| task_033.json | aegis_enabled | completed | 3.1353 | 604 | 0.0 | True |
| task_034.json | aegis_disabled | completed | 4.078 | 593 | 0.0 | True |
| task_034.json | aegis_enabled | completed | 3.7 | 593 | 0.0 | True |
| task_035.json | aegis_disabled | completed | 2.8494 | 626 | 0.0 | True |
| task_035.json | aegis_enabled | completed | 3.041 | 626 | 0.0 | True |
| task_036.json | aegis_disabled | completed | 3.5457 | 575 | 0.0 | True |
| task_036.json | aegis_enabled | completed | 3.0236 | 575 | 0.0 | True |
| task_037.json | aegis_disabled | completed | 7.2344 | 586 | 0.0 | True |
| task_037.json | aegis_enabled | completed | 6.2395 | 586 | 0.0 | True |
| task_038.json | aegis_disabled | completed | 7.914 | 575 | 0.0 | True |
| task_038.json | aegis_enabled | completed | 5.1292 | 575 | 0.0 | True |
| task_039.json | aegis_disabled | completed | 3.8835 | 586 | 0.0 | True |
| task_039.json | aegis_enabled | completed | 3.7841 | 586 | 0.0 | True |
| task_040.json | aegis_disabled | completed | 5.561 | 621 | 0.0 | True |
| task_040.json | aegis_enabled | completed | 4.8161 | 621 | 0.0 | True |
| task_041.json | aegis_disabled | completed | 3.8681 | 617 | 0.0 | True |
| task_041.json | aegis_enabled | completed | 3.7635 | 617 | 0.0 | True |
| task_042.json | aegis_disabled | completed | 4.4875 | 575 | 0.0 | True |
| task_042.json | aegis_enabled | completed | 3.7308 | 575 | 0.0 | True |
| task_043.json | aegis_disabled | completed | 3.9497 | 586 | 0.0 | True |
| task_043.json | aegis_enabled | completed | 4.0436 | 586 | 0.0 | True |
| task_044.json | aegis_disabled | completed | 3.9975 | 605 | 0.0 | True |
| task_044.json | aegis_enabled | completed | 3.3451 | 605 | 0.0 | True |
| task_045.json | aegis_disabled | completed | 3.8622 | 633 | 0.0 | True |
| task_045.json | aegis_enabled | completed | 3.5087 | 633 | 0.0 | True |
| task_046.json | aegis_disabled | completed | 4.6409 | 593 | 0.0 | True |
| task_046.json | aegis_enabled | completed | 4.4138 | 593 | 0.0 | True |
| task_047.json | aegis_disabled | completed | 3.9949 | 617 | 0.0 | True |
| task_047.json | aegis_enabled | completed | 4.8139 | 617 | 0.0 | True |
| task_048.json | aegis_disabled | completed | 3.7388 | 575 | 0.0 | True |
| task_048.json | aegis_enabled | completed | 3.6586 | 575 | 0.0 | True |
| task_049.json | aegis_disabled | completed | 3.5843 | 586 | 0.0 | True |
| task_049.json | aegis_enabled | completed | 3.6771 | 586 | 0.0 | True |
| task_050.json | aegis_disabled | completed | 3.9335 | 593 | 0.0 | True |
| task_050.json | aegis_enabled | completed | 3.9604 | 593 | 0.0 | True |
| task_051.json | aegis_disabled | completed | 4.4922 | 586 | 0.0 | True |
| task_051.json | aegis_enabled | completed | 4.772 | 586 | 0.0 | True |
| task_052.json | aegis_disabled | completed | 5.9673 | 575 | 0.0 | True |
| task_052.json | aegis_enabled | completed | 5.2309 | 575 | 0.0 | True |
| task_053.json | aegis_disabled | completed | 4.7913 | 617 | 0.0 | True |
| task_053.json | aegis_enabled | completed | 5.5966 | 617 | 0.0 | True |
| task_054.json | aegis_disabled | completed | 5.3683 | 575 | 0.0 | True |
| task_054.json | aegis_enabled | completed | 5.056 | 575 | 0.0 | True |
| task_055.json | aegis_disabled | completed | 3.795 | 586 | 0.0 | True |
| task_055.json | aegis_enabled | completed | 3.8072 | 586 | 0.0 | True |
| task_056.json | aegis_disabled | completed | 3.7459 | 593 | 0.0 | True |
| task_056.json | aegis_enabled | completed | 3.5126 | 593 | 0.0 | True |
| task_057.json | aegis_disabled | completed | 3.7415 | 604 | 0.0 | True |
| task_057.json | aegis_enabled | completed | 3.6878 | 604 | 0.0 | True |
| task_058.json | aegis_disabled | completed | 3.909 | 593 | 0.0 | True |
| task_058.json | aegis_enabled | completed | 3.4995 | 593 | 0.0 | True |
| task_059.json | aegis_disabled | completed | 4.8149 | 586 | 0.0 | True |
| task_059.json | aegis_enabled | completed | 5.7633 | 586 | 0.0 | True |
| task_060.json | aegis_disabled | completed | 4.0092 | 593 | 0.0 | True |
| task_060.json | aegis_enabled | completed | 3.7149 | 593 | 0.0 | True |
| task_061.json | aegis_disabled | completed | 5.7633 | 617 | 0.0 | True |
| task_061.json | aegis_enabled | completed | 5.3398 | 617 | 0.0 | True |
| task_062.json | aegis_disabled | completed | 3.8117 | 593 | 0.0 | True |
| task_062.json | aegis_enabled | completed | 3.6144 | 593 | 0.0 | True |
| task_063.json | aegis_disabled | completed | 4.6249 | 586 | 0.0 | True |
| task_063.json | aegis_enabled | completed | 4.2866 | 586 | 0.0 | True |
| task_064.json | aegis_disabled | completed | 4.029 | 605 | 0.0 | True |
| task_064.json | aegis_enabled | completed | 3.6099 | 605 | 0.0 | True |
| task_065.json | aegis_disabled | completed | 4.5039 | 586 | 0.0 | True |
| task_065.json | aegis_enabled | completed | 4.5247 | 586 | 0.0 | True |

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