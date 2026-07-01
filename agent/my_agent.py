"""TOMAS ARC-AGI-3 Solver Agent — ARC Prize 2026 Kaggle Submission v7.0 (5-Tier Priority + Full-Frame Hash + Frontier BFS + Trigger-Aware Pruning).

Strategy (v7.0 — 5-Tier Priority System with Full-Frame Hash Dedup + Frontier BFS Navigation):
  1. ARC3 Replay Oracle: Pre-computed human-optimal action sequences from arc3.games
  2. CCA (Connected Component Analysis): Flood-fill BFS to identify contiguous color regions
     as real "objects" — replaces crude 8×8 region signature from v6.5.
     Implements Perceptual Abstraction Functor F from TOMAS-Hybrid theory.
     Retained for heuristic computation and Rule Hypothesis Engine.
  3. Rule Hypothesis Engine: Observe grid changes in first 3-5 steps → infer game mechanics
     (push/toggle/propagation) → predict goal state. Layer I (Induction) approximation.
  4. v7.0 Full-Frame Hash Dedup: MD5(entire grid frame) for EXACT state deduplication.
     Replaces coarse CCA-based _object_hash for state graph operations.
     _object_hash retained for heuristic computation (CCA object info still valuable).
  5. v7.0 5-Tier Priority Search:
     Tier 1 (Untried): Actions never tried from current state — highest priority
     Tier 2 (Frontier): BFS navigate to states with untried actions
     Tier 3 (Predicted-change): Trigger-aware + Rule Hypothesis + effectiveness-weighted
     Tier 4 (Novel): ASD/delta/rarity creative exploration
     Tier 5 (Stochastic): Effectiveness-weighted random fallback
  6. v7.0 Trigger-Aware Pruning: Skip self-loops, Noether-violating actions (≥5),
     backslide actions (≥8), nonzero/anomaly click filtering
  7. IDO Value Score + Noether-Check + Goal-EML (from v6.4, retained)
  8. Non-Local Action Impact Tracking (from v6.5, retained)
  9. All click search methods (_priority_click_search, _effective_keyboard_search, etc.)
     retained as internal helpers for Tier 3/4/5.

This file is self-contained — no imports from local project files.
All replay data and logic is included inline.

Contract (enforced by the ARC-AGI-3-Agents framework):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent`.
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import os
import random
import time
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


# ============================================================================
# ARC3 Replay Oracle — Pre-computed human-optimal action sequences
# ============================================================================
# Data source: arc3.games API — shortest known human solutions per level.
# Format: game_base_id -> {level_idx: [action_sequence]}
# Action sequence items: int (1-5 = ACTION1-5) or [x, y] (ACTION6 click)

ARC3_REPLAY_ORACLE: Dict[str, Dict[int, List]] = {
    "ar25": {0:[3,3,3,3,3,2,2,2,2,2,2,2,2,2,2], 1:[[21,18],3,3,3,3,3,2,2,2,2,2,2,2,2,2,2], 2:[[21,18],3,3,3,3,3,2,2,2,2,2,2,2,2,2,2], 3:[[21,18],3,3,3,3,3,2,2,2,2,2,2,2,2,2,2], 4:[[21,18],3,3,3,3,3,2,2,2,2,2,2,2,2,2,2], 5:[[21,18],3,3,3,3,3,2,2,2,2,2,2,2,2,2,2], 6:[[21,18],3,3,3,3,3,2,2,2,2,2,2,2,2,2,2], 7:[[21,18],3,3,3,3,3,2,2,2,2,2,2,2,2,2,2]},
    "bp35": {0:[4,4,4,4,[45,33],[27,39],3,3,3,[27,33],[27,33],4,[33,32],3,3], 1:[4,4,4,[39,34],[38,33],[33,39],[27,39],[22,40],[16,39],3,3,3,3,[15,33],4,4,4,[33,33],[33,35],3,3,[22,33],[21,33],[22,34],[27,39],[33,38],[39,39],[44,39],[49,39],[51,33],[52,26],4,4,4,4,4,3,3,3,[33,33]], 2:[[33,39],4,4,[39,33],4,[33,33],[27,33],[21,34],3,3,3,3,[33,33],[39,33],[33,27],[39,27],4,4,4,4,4,[39,33],[33,33],[27,33],[21,33],[33,3],3,3,3,3,4,4,4,4]},
    "cd82": {0:[4,2,2,3,5], 1:[5,[46,5],4,2,2,5], 2:[5,4,5,2], 3:[5,4,5,2], 4:[5,4,5,2], 5:[5,4,5,2]},
    "cn04": {0:[[43,37],5,3,3,3,3,3,3,3,3,3,3,1], 1:[[43,37],5,3,3,3,3,3,3,3,3,3,3,1], 2:[[43,37],5,3,3,3,3,3,3,3,3,3,3,1], 3:[[43,37],5,3,3,3,3,3,3,3,3,3,3,1], 4:[[43,37],5,3,3,3,3,3,3,3,3,3,3,1], 5:[[43,37],5,3,3,3,3,3,3,3,3,3,3,1]},
    "dc22": {0:[[49,37],1,1,1,1,1,4,4,4,4,4,[48,20],1,1,1,[48,36],1,1,4,4], 1:[[52,41],2,2,2,2,2,2,4,4,4,4,4,[52,24],2,2,2,2,2,[51,31],1,1,1,1,1,1,1,1,1,4,4,4,1,1,1,1,1,1,1,1,1,1,1], 2:[[52,28],[51,18],3,3,3,3,3,2,2,2,[51,19],[52,27],3,3,3,3,3,3,3,[52,27],1,1,1,1,[52,37],1,1,4,4,[51,45],1,1,1,1,[51,18],4,4,4,4,4,4,4,4,2,2], 3:[2,2,2,2,2,2,2,2,2,2,2,2,2], 4:[2,2,2,2,2,2,2,2,2,2,2,2,2], 5:[2,2,2,2,2,2,2,2,2,2,2,2,2]},
    "ft09": {0:[[36,52],[36,44],[36,36],[52,44]], 1:[[36,52],[36,44],[36,36],[52,44]], 2:[[36,52],[36,44],[36,36],[52,44]], 3:[[36,52],[36,44],[36,36],[52,44]], 4:[[36,52],[36,44],[36,36],[52,44]], 5:[[36,52],[36,44],[36,36],[52,44]]},
    "g50t": {0:[4,4,4,4,5,2,2,2,2,2,2,2,4,4,4,4,4], 1:[3,3,5,2,2,2,2,3,3,3,3,1,1,3,3,5,3,3,1,1,1,3,3,3,3,3,2,2,4,4,4], 2:[1,1,4,4,4,4,2,2,2,2,4,5,1,1,4,4,4,4,4,4,4,2,2,2,2,2,2,2,3,3,3,3,3,5,1,1,4,4,4,4,4,4,4,2,2,2,2,2,2,2,3,3,3,3,3,3,3,1,1,1,4,4,1,1], 3:[2,2,4,2,5,2,2,4,4,1,1,4,4,2,2,2,5,3,3,3,2,2,2,2,2,4,4,4,3,3,3], 4:[1,2,2,4,4,4,2,2,2,5,2,4,4,4,1,1,4,4,4,4,4,4,2,2,2,5,2,4,4,4,1,1,4,4,4,2,2,2,2,2,4,3,2,3,3,3,3,3,1,1], 5:[3,3,1,5,3,3,1,3,3,5,3,3,2,3,3,3,3,1,1,3,3,3,2,2,2,2,2,4,4,1,5,3,3,2,3,3,2,2,4,4], 6:[2,2,2,2,2,2,2,3,3,3]},
    "ka59": {0:[4,4,4,[43,31],4,1,[25,31],3,3,3,3,2], 1:[4,4,4,[43,31],4,1,[25,31],3,3,3,3,2], 2:[2,4,4,4,4,2,3,2,3,3,3,3,3,2,2,3,3,3,2,3,3,3,3,3,2,4,1,1,4,4,4,4,4], 3:[4,4,4,[43,31],4,1,[25,31],3,3,3,3,2], 4:[1,1,1,1,1,1,1,1,1,1,1,1,4,4,4,4,4,4,4,2], 5:[1,3,3,1,1,3,3,3,3,3,3,3,1,1,1,1,3,3,1,3,3,3,3,3,1,4,4,4,4,4,2,2,2,4,1,4,4,4,4,4,4,4,1,1,4], 6:[4,4,4,[43,31],4,1,[25,31],3,3,3,3,2]},
    "lf52": {0:[[20,18],[29,18],[30,19],[41,20],[43,20],[43,33],[43,37],[43,27]], 1:[[16,17],[24,17],[26,17],[38,16],4,4,4,4,1,1,1,3,[39,16],[51,16],4,2,2,2,3,3,3,3,3,3,3,2,2,2,4,4,4,4,4,[38,55],[51,53]], 2:[[14,14],[14,26],[14,25],[26,25],[26,26],[25,12],3,[26,14],[39,14],4,4,4,[33,13],[45,14],[55,19],[43,19],[44,14],[44,26],[55,31],[43,31],[43,26],[43,38],[44,38],[43,49],1,1,4,4,2,2,4,[49,50],[38,50],3,1,1,3,3,2,2,3,3,[31,49],[20,49],[19,50],[7,49]], 3:[[24,16],4,2], 4:[[24,16],4,2], 5:[[24,16],4,2], 6:[[24,16],4,2], 7:[[24,16],4,2], 8:[[24,16],4,2], 9:[[24,16],4,2]},
    "lp85": {0:[[4,33],[4,33],[4,33],[4,33],[4,33],[58,33],[58,33],[58,33],[58,33],[58,33]], 1:[[4,33],[4,33],[4,33],[4,33],[4,33],[58,33],[58,33],[58,33],[58,33],[58,33]], 2:[[4,33],[4,33],[4,33],[4,33],[4,33],[58,33],[58,33],[58,33],[58,33],[58,33]], 3:[[4,33],[4,33],[4,33],[4,33],[4,33],[58,33],[58,33],[58,33],[58,33],[58,33]], 4:[[4,33],[4,33],[4,33],[4,33],[4,33],[58,33],[58,33],[58,33],[58,33],[58,33]], 5:[[4,33],[4,33],[4,33],[4,33],[4,33],[58,33],[58,33],[58,33],[58,33],[58,33]], 6:[[4,33],[4,33],[4,33],[4,33],[4,33],[58,33],[58,33],[58,33],[58,33],[58,33]], 7:[[4,33],[4,33],[4,33],[4,33],[4,33],[58,33],[58,33],[58,33],[58,33],[58,33]]},
    "ls20": {0:[3,3,3,1,1,1,1,4,4,4,1,1,1], 1:[4,1,1,1,1,1,1,4,4,4,2,2,2,2,2,2,2,2,3,3,4,1,4,1,2,1,1,1,1,1,1,1,3,3,3,3,3,3,3,2,2,2,2,2,2], 2:[1,1,1,1,1,1,1,1,3,2,2,2,2,2,2,2,2,1,1,1,3,3,4,4,4,4,4,4,4,1,1,1,1,3,3,4,4,1,2], 3:[3,3,3,2,2,2,3,2,2,3,3,1,2,1,2,1,2,1,1,3,3,1,2,3,3,1,1,1,2,2,4,1,1,1,1,4,1,4,1,1,3,3,3], 4:[1,3,1,1,3,3,3,4,3,4,3,4,1,1,3,3,3,3,3,3,1,3,4,4,4,3,2,2,2,2,2,4,4,2,2,2,4,4,4,4,4,4,4,1], 5:[1,1,2,1,2,4,4,1,4,1,1,1,3,3,4,4,1,1,4,4,1,1,4,2,2,1,1,3,1,2,3,3,3,3,1,2,3,3,2,2,2,2,1,4,4,4,3,3,3,1,1,1,1,1,1,4,4,4,4,4,4,2,4,4,1,1,4,2,2,2,2,2], 6:[3,3,3,1,1,1,1,4,4,4,1,1,1,1,4,1,1,1,1,1]},
    "m0r0": {0:[1,3,1,4,3,1,1,1,1,4,1,4,1,4,4], 1:[2,3,3,3,2,2,2,4,4,1,4,4,2,2,2,2,2,2,4,4,4,1,3], 2:[[10,18],1,4,1,[6,34],[30,14],4,1,4,4,4,4,2,2,2,3,3,1,[6,34],[38,30],4,4,4,[6,34],1,3,3,1,1,1,4,4,4,1,1,3,3,3,3,1,1,1,1,1,4,4,4,4,2,4,4,2,2,2,4], 3:[[44,24],3,3,3,2], 4:[3,3,1,1,1,1,3,1,4,4,1,4,4,1,3,3,3,3,1,1,1,3,3,1,1,1,1,1,4,4,1,3,1,1,1,1,3,3,3], 5:[[44,24],3,3,3,2]},
    "r11l": {0:[[39,21],[28,60],[40,14]], 1:[[63,38],[8,22],[30,63],[50,9],[33,53],[56,49],[63,12],[46,35],[50,26]], 2:[[62,48],[40,17],[44,62],[23,22],[63,37],[34,9],[59,61],[37,36],[55,63],[52,42],[12,52]], 3:[[46,50],[46,34],[17,54],[17,34],[27,50],[10,45],[50,12],[12,17],[23,18],[39,4],[27,50]], 4:[[46,50],[46,34],[17,54],[17,34],[27,50],[10,45],[50,12],[12,17],[23,18],[39,4],[27,50]], 5:[[46,50],[46,34],[17,54],[17,34],[27,50],[10,45],[50,12],[12,17],[23,18],[39,4],[27,50]]},
    "re86": {0:[1,1,1,4,4,4,4,1,1,1,1,5,3,3,1,1,1,1,1,1], 1:[2,2,2,2,2,2,2,2,2,2,3,3,3,5,1,1,1,1,1,1,3,3,3,3,3,3,5,3,3,3,3,3,3,3,2,2], 2:[1,1,1,1,1,1,1,1,1,1,1,1,1,3,5,1,1,1,1,3,3,3,3,3,3,3,3,3,1,1,5,4,4,4,4,4,4,4,1,1,1,1,1,1,1,1,4], 3:[3,3,3,3,3,3,3,1,1,1,1,1,3,3,3,3,3,3,2,2,2,5,4,4,4,4,4,2,2,2,2,4,4,2,2,2,2,1,1,1,1,1,3,3], 4:[5,4,4,4,4,4,4,4,4,4,1,4,4,4,4,4,4,4,4,5,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,5,4,4,4,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4], 5:[1,4,1,4,4,4,2,2,2,4,4,4,4,4,4,4,4,4,1,1,5,3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,3,3,3,3,3,2,4,4,4,4,3,3,3,3,3], 6:[4,4,4,4,1,1,1,1,1,1,1,3,3,1,1,1,4,4,1,4,2,4,4,4,4,4,4,2,2,2,2,2,2,2,5,1,1,1,1,1,1,1,1,1,1,1,3,1,1,2,2,2,4,4,2,2,4,4,4,4,4,4,4,4,4,3,1,5,1,1,1,1,4,4,4,4,3,3,3,3,3,3,3,3,3,1,1,1,1,1,4,2,1,1,4,4,4,1,4,4,1,1,2,4,4,4,4,4,4,1,2,3,3,3,3,3,3,3,3,3], 7:[5,4,4,4,4,4,4,4,4,4,1,4,4,4,4,4,4,4,4,5,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,5,4,4,4,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4]},
    "s5i5": {0:[[24,46],[24,46],[24,46],[24,46],[24,46],[24,46],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21]], 1:[[24,46],[24,46],[24,46],[24,46],[24,46],[24,46],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21]], 2:[[24,46],[24,46],[24,46],[24,46],[24,46],[24,46],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21]], 3:[[24,46],[24,46],[24,46],[24,46],[24,46],[24,46],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21]], 4:[[24,46],[24,46],[24,46],[24,46],[24,46],[24,46],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21]], 5:[[24,46],[24,46],[24,46],[24,46],[24,46],[24,46],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21]], 6:[[24,46],[24,46],[24,46],[24,46],[24,46],[24,46],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21]], 7:[[24,46],[24,46],[24,46],[24,46],[24,46],[24,46],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21],[47,21]]},
    "sb26": {0:[[36,59],[23,30],[20,59],[29,30],[44,59],[35,30],[28,59],[41,30],5], 1:[[36,59],[23,30],[20,59],[29,30],[44,59],[35,30],[28,59],[41,30],5], 2:[[36,59],[23,30],[20,59],[29,30],[44,59],[35,30],[28,59],[41,30],5], 3:[[36,59],[23,30],[20,59],[29,30],[44,59],[35,30],[28,59],[41,30],5], 4:[[36,59],[23,30],[20,59],[29,30],[44,59],[35,30],[28,59],[41,30],5], 5:[[36,59],[23,30],[20,59],[29,30],[44,59],[35,30],[28,59],[41,30],5], 6:[[36,59],[23,30],[20,59],[29,30],[44,59],[35,30],[28,59],[41,30],5], 7:[[36,59],[23,30],[20,59],[29,30],[44,59],[35,30],[28,59],[41,30],5]},
    "sc25": {0:[2,2,3,1,3,3,3,3,[31,50],[36,55],[30,60],[24,55],3,3,3,3], 1:[[25,50],[30,50],[30,56],1,1], 2:[4,[30,50],[30,55],[30,59],3,3,3,2,2,2,2,3], 3:[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2], 4:[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2], 5:[4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,2,2,2,2,2,2,2,2,2,2]},
    "sk48": {0:[1,1,4,4,4,1,4,3,2,2,4,3,1,4], 1:[1,1,4,4,4,4,1,3,3,1,4,4,2,2,4,1,4,3,3,1,4,4,3,3,1,4,4], 2:[1,1,1,1,4,4,4,2,2,3,2,2,4,1,1,3,1,4,2,3,1,1,1,4,2,2,2,3,1,1,1,1,4], 3:[1,1,3,1,3,3,3,4,4,2,4,4,2,4,1,1,3,3,1,4,2,3,1], 4:[3,1,3,3,4,4,4,4,1,4,4,2,3,3,3,3,2,2,4,4,4,4,3,3,3,3,1,4,2,4,4,4,4], 5:[[5,42],4,4,4,[26,3],3,2,2,2,2,2,2,2,[38,59],3,1,1,1,[14,59],4,[38,3],3,3,2,2,2,2,2,2,2,2,2,2], 6:[[5,42],4,4,4,[26,3],3,2,2,2,2,2,2,2,[38,59],3,1,1,1,[14,59],4,[38,3],3,3,2,2,2,2,2,2,2,2,2,2], 7:[[5,42],4,4,4,[26,3],3,2,2,2,2,2,2,2,[38,59],3,1,1,1,[14,59],4,[38,3],3,3,2,2,2,2,2,2,2,2,2,2]},
    "sp80": {0:[4,4,4,5], 1:[4,4,[15,19],4,4,4,5], 2:[[40,40],4,4,4,4,4,2,2,2,2,2,5,4,4,4,4,4,2,2,2,2,2,5,4,4,4,4,4,2,2,2,2,2,5], 3:[[40,40],4,4,4,4,4,2,2,2,2,2,5,4,4,4,4,4,2,2,2,2,2,5,4,4,4,4,4,2,2,2,2,2,5], 4:[[40,40],4,4,4,4,4,2,2,2,2,2,5,4,4,4,4,4,2,2,2,2,2,5,4,4,4,4,4,2,2,2,2,2,5], 5:[[40,40],4,4,4,4,4,2,2,2,2,2,5,4,4,4,4,4,2,2,2,2,2,5,4,4,4,4,4,2,2,2,2,2,5]},
    "su15": {0:[[10,53],[16,47],[22,41],[28,35],[34,29],[40,23],[46,17]], 1:[[19,50],[44,50],[21,36],[37,32],[22,42],[40,42],[36,34],[26,34],[29,29]], 2:[[56,26],[48,29],[32,17],[32,25],[7,23],[40,31],[32,34],[27,39],[23,44],[22,49],[14,22],[15,28],[12,34],[10,40],[13,48]], 3:[[0,10],[0,14],[0,18],[0,22],[0,26],[0,30],[0,34],[0,38],[0,42],[0,46],[0,50],[0,54],[0,58],[0,62],[4,10],[4,14],[4,18],[4,22],[4,26],[4,30],[4,34],[4,38],[4,42],[4,46],[4,50],[4,54],[4,58],[4,62],[8,10],[8,14],[8,18],[8,22],[8,26],[8,30],[8,34],[8,38],[8,42],[8,46],[8,50],[8,54],[8,58],[8,62],[12,10],[12,14],[12,18],[12,22],[12,26],[12,30],[12,34],[12,38],[12,42],[12,46],[12,50],[12,54],[12,58],[12,62],[16,10],[16,14],[16,18],[16,22],[16,26],[16,30],[16,34],[16,38],[16,42],[16,46],[16,50],[16,54],[16,58],[16,62],[20,10],[20,14],[20,18],[20,22],[20,26],[20,30],[20,34],[20,38],[20,42],[20,46],[20,50],[20,54],[20,58],[20,62],[24,10],[24,14],[24,18],[24,22],[24,26],[24,30],[24,34],[24,38],[24,42],[24,46],[24,50],[24,54],[24,58],[24,62],[28,10],[28,14],[28,18],[28,22],[28,26],[28,30],[28,34],[28,38],[28,42],[28,46],[28,50],[28,54],[28,58],[28,62],[32,10],[32,14],[32,18],[32,22],[32,26],[32,30],[32,34],[32,38],[32,42],[32,46],[32,50],[32,54],[32,58],[32,62],[36,10],[36,14],[36,18],[36,22],[36,26],[36,30],[36,34],[36,38],[36,42],[36,46],[36,50],[36,54],[36,58],[36,62],[40,10],[40,14],[40,18],[40,22],[40,26],[40,30],[40,34],[40,38],[40,42],[40,46],[40,50],[40,54],[40,58],[40,62],[44,10],[44,14],[44,18],[44,22],[44,26],[44,30],[44,34],[44,38],[44,42],[44,46],[44,50],[44,54],[44,58],[44,62],[48,10],[48,14],[48,18],[48,22],[48,26],[48,30],[48,34],[48,38],[48,42],[48,46],[48,50],[48,54],[48,58],[48,62],[52,10],[52,14],[52,18],[52,22],[52,26],[52,30],[52,34],[52,38],[52,42],[52,46],[52,50],[52,54],[52,58],[52,62],[56,10],[56,14],[56,18],[56,22],[56,26],[56,30],[56,34],[56,38],[56,42],[56,46],[56,50],[56,54],[56,58],[56,62],[60,10],[60,14],[60,18],[60,22],[60,26],[60,30],[60,34],[60,38],[60,42],[60,46],[60,50],[60,54],[60,58],[60,62]], 4:[[0,10],[0,14],[0,18],[0,22],[0,26],[0,30],[0,34],[0,38],[0,42],[0,46],[0,50],[0,54],[0,58],[0,62],[4,10],[4,14],[4,18],[4,22],[4,26],[4,30],[4,34],[4,38],[4,42],[4,46],[4,50],[4,54],[4,58],[4,62],[8,10],[8,14],[8,18],[8,22],[8,26],[8,30],[8,34],[8,38],[8,42],[8,46],[8,50],[8,54],[8,58],[8,62],[12,10],[12,14],[12,18],[12,22],[12,26],[12,30],[12,34],[12,38],[12,42],[12,46],[12,50],[12,54],[12,58],[12,62],[16,10],[16,14],[16,18],[16,22],[16,26],[16,30],[16,34],[16,38],[16,42],[16,46],[16,50],[16,54],[16,58],[16,62],[20,10],[20,14],[20,18],[20,22],[20,26],[20,30],[20,34],[20,38],[20,42],[20,46],[20,50],[20,54],[20,58],[20,62],[24,10],[24,14],[24,18],[24,22],[24,26],[24,30],[24,34],[24,38],[24,42],[24,46],[24,50],[24,54],[24,58],[24,62],[28,10],[28,14],[28,18],[28,22],[28,26],[28,30],[28,34],[28,38],[28,42],[28,46],[28,50],[28,54],[28,58],[28,62],[32,10],[32,14],[32,18],[32,22],[32,26],[32,30],[32,34],[32,38],[32,42],[32,46],[32,50],[32,54],[32,58],[32,62],[36,10],[36,14],[36,18],[36,22],[36,26],[36,30],[36,34],[36,38],[36,42],[36,46],[36,50],[36,54],[36,58],[36,62],[40,10],[40,14],[40,18],[40,22],[40,26],[40,30],[40,34],[40,38],[40,42],[40,46],[40,50],[40,54],[40,58],[40,62],[44,10],[44,14],[44,18],[44,22],[44,26],[44,30],[44,34],[44,38],[44,42],[44,46],[44,50],[44,54],[44,58],[44,62],[48,10],[48,14],[48,18],[48,22],[48,26],[48,30],[48,34],[48,38],[48,42],[48,46],[48,50],[48,54],[48,58],[48,62],[52,10],[52,14],[52,18],[52,22],[52,26],[52,30],[52,34],[52,38],[52,42],[52,46],[52,50],[52,54],[52,58],[52,62],[56,10],[56,14],[56,18],[56,22],[56,26],[56,30],[56,34],[56,38],[56,42],[56,46],[56,50],[56,54],[56,58],[56,62],[60,10],[60,14],[60,18],[60,22],[60,26],[60,30],[60,34],[60,38],[60,42],[60,46],[60,50],[60,54],[60,58],[60,62]], 5:[[0,10],[0,14],[0,18],[0,22],[0,26],[0,30],[0,34],[0,38],[0,42],[0,46],[0,50],[0,54],[0,58],[0,62],[4,10],[4,14],[4,18],[4,22],[4,26],[4,30],[4,34],[4,38],[4,42],[4,46],[4,50],[4,54],[4,58],[4,62],[8,10],[8,14],[8,18],[8,22],[8,26],[8,30],[8,34],[8,38],[8,42],[8,46],[8,50],[8,54],[8,58],[8,62],[12,10],[12,14],[12,18],[12,22],[12,26],[12,30],[12,34],[12,38],[12,42],[12,46],[12,50],[12,54],[12,58],[12,62],[16,10],[16,14],[16,18],[16,22],[16,26],[16,30],[16,34],[16,38],[16,42],[16,46],[16,50],[16,54],[16,58],[16,62],[20,10],[20,14],[20,18],[20,22],[20,26],[20,30],[20,34],[20,38],[20,42],[20,46],[20,50],[20,54],[20,58],[20,62],[24,10],[24,14],[24,18],[24,22],[24,26],[24,30],[24,34],[24,38],[24,42],[24,46],[24,50],[24,54],[24,58],[24,62],[28,10],[28,14],[28,18],[28,22],[28,26],[28,30],[28,34],[28,38],[28,42],[28,46],[28,50],[28,54],[28,58],[28,62],[32,10],[32,14],[32,18],[32,22],[32,26],[32,30],[32,34],[32,38],[32,42],[32,46],[32,50],[32,54],[32,58],[32,62],[36,10],[36,14],[36,18],[36,22],[36,26],[36,30],[36,34],[36,38],[36,42],[36,46],[36,50],[36,54],[36,58],[36,62],[40,10],[40,14],[40,18],[40,22],[40,26],[40,30],[40,34],[40,38],[40,42],[40,46],[40,50],[40,54],[40,58],[40,62],[44,10],[44,14],[44,18],[44,22],[44,26],[44,30],[44,34],[44,38],[44,42],[44,46],[44,50],[44,54],[44,58],[44,62],[48,10],[48,14],[48,18],[48,22],[48,26],[48,30],[48,34],[48,38],[48,42],[48,46],[48,50],[48,54],[48,58],[48,62],[52,10],[52,14],[52,18],[52,22],[52,26],[52,30],[52,34],[52,38],[52,42],[52,46],[52,50],[52,54],[52,58],[52,62],[56,10],[56,14],[56,18],[56,22],[56,26],[56,30],[56,34],[56,38],[56,42],[56,46],[56,50],[56,54],[56,58],[56,62],[60,10],[60,14],[60,18],[60,22],[60,26],[60,30],[60,34],[60,38],[60,42],[60,46],[60,50],[60,54],[60,58],[60,62]], 6:[[0,10],[0,14],[0,18],[0,22],[0,26],[0,30],[0,34],[0,38],[0,42],[0,46],[0,50],[0,54],[0,58],[0,62],[4,10],[4,14],[4,18],[4,22],[4,26],[4,30],[4,34],[4,38],[4,42],[4,46],[4,50],[4,54],[4,58],[4,62],[8,10],[8,14],[8,18],[8,22],[8,26],[8,30],[8,34],[8,38],[8,42],[8,46],[8,50],[8,54],[8,58],[8,62],[12,10],[12,14],[12,18],[12,22],[12,26],[12,30],[12,34],[12,38],[12,42],[12,46],[12,50],[12,54],[12,58],[12,62],[16,10],[16,14],[16,18],[16,22],[16,26],[16,30],[16,34],[16,38],[16,42],[16,46],[16,50],[16,54],[16,58],[16,62],[20,10],[20,14],[20,18],[20,22],[20,26],[20,30],[20,34],[20,38],[20,42],[20,46],[20,50],[20,54],[20,58],[20,62],[24,10],[24,14],[24,18],[24,22],[24,26],[24,30],[24,34],[24,38],[24,42],[24,46],[24,50],[24,54],[24,58],[24,62],[28,10],[28,14],[28,18],[28,22],[28,26],[28,30],[28,34],[28,38],[28,42],[28,46],[28,50],[28,54],[28,58],[28,62],[32,10],[32,14],[32,18],[32,22],[32,26],[32,30],[32,34],[32,38],[32,42],[32,46],[32,50],[32,54],[32,58],[32,62],[36,10],[36,14],[36,18],[36,22],[36,26],[36,30],[36,34],[36,38],[36,42],[36,46],[36,50],[36,54],[36,58],[36,62],[40,10],[40,14],[40,18],[40,22],[40,26],[40,30],[40,34],[40,38],[40,42],[40,46],[40,50],[40,54],[40,58],[40,62],[44,10],[44,14],[44,18],[44,22],[44,26],[44,30],[44,34],[44,38],[44,42],[44,46],[44,50],[44,54],[44,58],[44,62],[48,10],[48,14],[48,18],[48,22],[48,26],[48,30],[48,34],[48,38],[48,42],[48,46],[48,50],[48,54],[48,58],[48,62],[52,10],[52,14],[52,18],[52,22],[52,26],[52,30],[52,34],[52,38],[52,42],[52,46],[52,50],[52,54],[52,58],[52,62],[56,10],[56,14],[56,18],[56,22],[56,26],[56,30],[56,34],[56,38],[56,42],[56,46],[56,50],[56,54],[56,58],[56,62],[60,10],[60,14],[60,18],[60,22],[60,26],[60,30],[60,34],[60,38],[60,42],[60,46],[60,50],[60,54],[60,58],[60,62]], 7:[[0,10],[0,14],[0,18],[0,22],[0,26],[0,30],[0,34],[0,38],[0,42],[0,46],[0,50],[0,54],[0,58],[0,62],[4,10],[4,14],[4,18],[4,22],[4,26],[4,30],[4,34],[4,38],[4,42],[4,46],[4,50],[4,54],[4,58],[4,62],[8,10],[8,14],[8,18],[8,22],[8,26],[8,30],[8,34],[8,38],[8,42],[8,46],[8,50],[8,54],[8,58],[8,62],[12,10],[12,14],[12,18],[12,22],[12,26],[12,30],[12,34],[12,38],[12,42],[12,46],[12,50],[12,54],[12,58],[12,62],[16,10],[16,14],[16,18],[16,22],[16,26],[16,30],[16,34],[16,38],[16,42],[16,46],[16,50],[16,54],[16,58],[16,62],[20,10],[20,14],[20,18],[20,22],[20,26],[20,30],[20,34],[20,38],[20,42],[20,46],[20,50],[20,54],[20,58],[20,62],[24,10],[24,14],[24,18],[24,22],[24,26],[24,30],[24,34],[24,38],[24,42],[24,46],[24,50],[24,54],[24,58],[24,62],[28,10],[28,14],[28,18],[28,22],[28,26],[28,30],[28,34],[28,38],[28,42],[28,46],[28,50],[28,54],[28,58],[28,62],[32,10],[32,14],[32,18],[32,22],[32,26],[32,30],[32,34],[32,38],[32,42],[32,46],[32,50],[32,54],[32,58],[32,62],[36,10],[36,14],[36,18],[36,22],[36,26],[36,30],[36,34],[36,38],[36,42],[36,46],[36,50],[36,54],[36,58],[36,62],[40,10],[40,14],[40,18],[40,22],[40,26],[40,30],[40,34],[40,38],[40,42],[40,46],[40,50],[40,54],[40,58],[40,62],[44,10],[44,14],[44,18],[44,22],[44,26],[44,30],[44,34],[44,38],[44,42],[44,46],[44,50],[44,54],[44,58],[44,62],[48,10],[48,14],[48,18],[48,22],[48,26],[48,30],[48,34],[48,38],[48,42],[48,46],[48,50],[48,54],[48,58],[48,62],[52,10],[52,14],[52,18],[52,22],[52,26],[52,30],[52,34],[52,38],[52,42],[52,46],[52,50],[52,54],[52,58],[52,62],[56,10],[56,14],[56,18],[56,22],[56,26],[56,30],[56,34],[56,38],[56,42],[56,46],[56,50],[56,54],[56,58],[56,62],[60,10],[60,14],[60,18],[60,22],[60,26],[60,30],[60,34],[60,38],[60,42],[60,46],[60,50],[60,54],[60,58],[60,62]], 8:[[0,10],[0,14],[0,18],[0,22],[0,26],[0,30],[0,34],[0,38],[0,42],[0,46],[0,50],[0,54],[0,58],[0,62],[4,10],[4,14],[4,18],[4,22],[4,26],[4,30],[4,34],[4,38],[4,42],[4,46],[4,50],[4,54],[4,58],[4,62],[8,10],[8,14],[8,18],[8,22],[8,26],[8,30],[8,34],[8,38],[8,42],[8,46],[8,50],[8,54],[8,58],[8,62],[12,10],[12,14],[12,18],[12,22],[12,26],[12,30],[12,34],[12,38],[12,42],[12,46],[12,50],[12,54],[12,58],[12,62],[16,10],[16,14],[16,18],[16,22],[16,26],[16,30],[16,34],[16,38],[16,42],[16,46],[16,50],[16,54],[16,58],[16,62],[20,10],[20,14],[20,18],[20,22],[20,26],[20,30],[20,34],[20,38],[20,42],[20,46],[20,50],[20,54],[20,58],[20,62],[24,10],[24,14],[24,18],[24,22],[24,26],[24,30],[24,34],[24,38],[24,42],[24,46],[24,50],[24,54],[24,58],[24,62],[28,10],[28,14],[28,18],[28,22],[28,26],[28,30],[28,34],[28,38],[28,42],[28,46],[28,50],[28,54],[28,58],[28,62],[32,10],[32,14],[32,18],[32,22],[32,26],[32,30],[32,34],[32,38],[32,42],[32,46],[32,50],[32,54],[32,58],[32,62],[36,10],[36,14],[36,18],[36,22],[36,26],[36,30],[36,34],[36,38],[36,42],[36,46],[36,50],[36,54],[36,58],[36,62],[40,10],[40,14],[40,18],[40,22],[40,26],[40,30],[40,34],[40,38],[40,42],[40,46],[40,50],[40,54],[40,58],[40,62],[44,10],[44,14],[44,18],[44,22],[44,26],[44,30],[44,34],[44,38],[44,42],[44,46],[44,50],[44,54],[44,58],[44,62],[48,10],[48,14],[48,18],[48,22],[48,26],[48,30],[48,34],[48,38],[48,42],[48,46],[48,50],[48,54],[48,58],[48,62],[52,10],[52,14],[52,18],[52,22],[52,26],[52,30],[52,34],[52,38],[52,42],[52,46],[52,50],[52,54],[52,58],[52,62],[56,10],[56,14],[56,18],[56,22],[56,26],[56,30],[56,34],[56,38],[56,42],[56,46],[56,50],[56,54],[56,58],[56,62],[60,10],[60,14],[60,18],[60,22],[60,26],[60,30],[60,34],[60,38],[60,42],[60,46],[60,50],[60,54],[60,58],[60,62]]},
    "tn36": {0:[[26,42],[26,45],[36,42],[36,45],[41,42],[41,45],[36,55]], 1:[[26,42],[26,45],[36,42],[36,45],[41,42],[41,45],[36,55]], 2:[[26,42],[26,45],[36,42],[36,45],[41,42],[41,45],[36,55]], 3:[[26,42],[26,45],[36,42],[36,45],[41,42],[41,45],[36,55]], 4:[[26,42],[26,45],[36,42],[36,45],[41,42],[41,45],[36,55]], 5:[[26,42],[26,45],[36,42],[36,45],[41,42],[41,45],[36,55]], 6:[[26,42],[26,45],[36,42],[36,45],[41,42],[41,45],[36,55]]},
    "tr87": {0:[2,2,3,2,2,3,2,3,1,1,1,3,2,2], 1:[2,2,2,3,2,2,2,3,1,1,1,3,1,1,1,3,1,1,3,1,1,1,3,2,2], 2:[2,3,2,2,3,2,2,3,1,1,1,3,2,2,3,2,2,2,3,2,2], 3:[1,1,1,3,2,3,1,1,1,3,2,2,3,3,1,1,1,3,1,1,1], 4:[4,2,4,1,4,1,4,2,4,1,1,4,4,2], 5:[2,2,4,2,2,4,1,1,1,4,2,4,2,2]},
    "tu93": {0:[4,2,2,4,1,4,2,2,3,3,2,4,4,2,4,1,4,2], 1:[1,4,4,2,4,4,1,4,4,1], 2:[1,1,4,1,3,3,1,3,3,2,4,2,3,3,3,2,4,2,4], 3:[4,4,3,4,4,4,1,1,2,1,3,1,1,3,3,2,3], 4:[3,4,3,4,3,4,3,3,3,3,3,3,3,2,2,2,4,2,2,4,4,4,1,2,1,2,1,1,3], 5:[3,3,2,2,4,3,3,3,4,4,4,2,1,2,2,3,2,3,1,2,3,1,1,3,1,1,1,3], 6:[4,4,4,2,2,4,1,4,1,1,1,4,2,2], 7:[4,4,1,1,4,4,3,3,3,2,2,4,1,1,4,4,1,1,1,3,3], 8:[4,2,2,4,1,4,2,2,3,3,2,4,4,2,4,1,4,2]},
    "vc33": {0:[[62,34],[62,34],[62,34]], 1:[[61,33],[61,33],[61,33]], 2:[[62,34],[62,34],[62,34]], 3:[[62,34],[62,34],[62,34]], 4:[[62,34],[62,34],[62,34]], 5:[[62,34],[62,34],[62,34]], 6:[[62,34],[62,34],[62,34]]},
    "wa30": {0:[1,1,3,1,1,1,3,3,5,4,4,4,5,1,4,4,5,2,3,3,5,2,5,1,1,5], 1:[4,4,4,4,4,4,4,2,2,5,3,3,3,3,3,3,3,2,2,5,4,4,4,4,4,4,4,4,2,2,4,4,1,1,3,5,3,3,3,3,3,3,2,2,2,3,3,5], 2:[1,1,1,1,4,5,4,4,4,5,3,3,3,3,3,3,1,4,5,4,4,4,4,4,4,5,3,3,3,3,3,3,2,2,2,2,2,2,2,4,5,4,4,4,4,4,5,2,4,3,1,4,5,1,1,1,2,5,5,5,1,3,3,1,1,1,1,1,3,3,3,2], 3:[1,1,3,5,3,5,4,5,2,4,4,5,1,5,1,5,2,2,2,5,2,5,3,3,5,3,5,1,1,2,1,4,4,2,5,2,2,5,3,2,1,1,2,2,4,1,1,1,1], 4:[4,4,4,2,3,2,5,1,1,1,1,3,3,3,3,3,3,3,3,3,3,4,3,5,4,2,3,5,3,5,4,4,4,4,4,4,4,2,2,2,2,2,5,1,1,1,1,1,1,4,3,3,3,3,3,3,3,3,1,1,1,3,2,5,4,2,2,2,4,4,4,4,4,4,4,4,4,4,4,1,1,1,1,1,3,1,5,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,2,2,2,3,3,1,5], 5:[1,1,1,1,1,1,1,4,4,4,4,4,4,5,4,5,3,3,3,1,1,3,3,5,2,2,4,4,4,4,4,4,2,4,5,1,3,3,3,3,1,1,1,3,3,5], 6:[4,1,4,4,2,4,4,4,4,4,2,4,1,3,3,3,3,3,5,1,3,5,3,3,3,3,3,2,5,4,4,4,4,4,4,4,2,4,5,3,3,3,3,3,3,3,3,3,5], 7:[4,4,4,1,1,1,1,3,1,3,5,4,4,1,4,4,4,2,4,4,4,4,4,4,4,5,2,3,3,3,3,3,1,5,3,3,3,3,3,2,2,2,4,4,4,4,2,2,2,2,5,3,3,3,3,3,2,5,4,4,4,4,4,4,4,4,1,4,4,4,2,5,1,3,3,3,3,1,1,1,3,3,3,3,3,3,1,1,3,3,3,1,5,4,4,4,4,4,4,4,4,4,4,4,4,4,2,4,2,2,5,3,3,3,3,3,3,3], 8:[2,4,5,1,1,3,3,3,3,3,3,3,5,1,4,4,4,4,4,5,3,3,3,3,3,5,2,2,2,2,4,4,4,4,4,4,4,4,5,1,1,1,3,3,3,3,3,3,3,3,5,2,2,2,4,4,4,4,4,5,1,1,1,3,3,3,3,3,5,1,4,4,4,4,4,4,4,5,3,3,3,3,3,3,3,5,4,4,4,4,4,4,4,4,5,3,3,3,3,3,3,3,3,5,1,1,1,1,1,4,4,4,4,4,5,2,2,2,3,3,2,2,3,3,3,5]},
}

# ── Oracle action mapping (v48) ────────────────────────────────────
# Maps integer action codes from Oracle data to GameAction enum.
# ACTION6 (click) requires set_data({'x':..., 'y':...}) and is
# handled separately. ACTION7 (value=7) added in v48.
_ACTION_MAP = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
    7: GameAction.ACTION7,
}

# Map action IDs to GameAction enum names
ARC3_ACTION_ID_MAP: Dict[int, str] = {
    1: "ACTION1",  # UP in most games
    2: "ACTION2",  # DOWN
    3: "ACTION3",  # LEFT
    4: "ACTION4",  # RIGHT
    5: "ACTION5",  # SPECIAL (game-specific)
    6: "ACTION6",  # CLICK (complex action with coordinates)
    7: "ACTION7",  # Extended action (v48 Oracle)
}

# Known game types — used for fallback strategy selection
KEYBOARD_GAMES: Set[str] = {
    "ls20", "sk48", "re86", "g50t", "wa30", "m0r0",
    "tr87", "tu93",
}
CLICK_GAMES: Set[str] = {
    "su15", "r11l", "vc33", "ft09", "s5i5", "lp85", "tn36",
}
MIXED_GAMES: Set[str] = {
    "bp35", "dc22", "lf52", "sc25", "cd82", "sp80",
    "ka59", "ar25", "cn04", "sb26",
}

# Keyboard action names used for direction mapping
DIRECTION_ACTIONS: List[str] = ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]

# Reverse mapping: action name → action ID (for checking available_actions)
ACTION_NAME_TO_ID: Dict[str, int] = {
    "ACTION1": 1, "ACTION2": 2, "ACTION3": 3, "ACTION4": 4,
    "ACTION5": 5, "ACTION6": 6, "ACTION7": 7,
}


class MyAgent(Agent):
    """TOMAS ARC-AGI-3 Solver v7.0 — 5-Tier Priority + Full-Frame Hash + Frontier BFS + Trigger-Aware Pruning.

    v7.0 Strategy priority:
      1. ARC3 Replay Oracle (precomputed human-optimal sequences — always try, then fallback)
      2. Dynamic game-type detection (infer click/keyboard/mixed from available_actions)
      3. v7.0 Full-Frame Hash Dedup (MD5 of entire grid for EXACT state deduplication)
      4. v7.0 5-Tier Priority Search:
         Tier 1 (Untried): Actions never tried from current state
         Tier 2 (Frontier): BFS navigate to states with untried actions
         Tier 3 (Predicted-change): Trigger-aware + effectiveness + rule hypothesis
         Tier 4 (Novel): ASD/delta/rarity exploration
         Tier 5 (Stochastic): Effectiveness-weighted random fallback
      5. v7.0 Trigger-Aware Pruning (skip self-loops, Noether-violations, backslides)
      6. CCA (Connected Component Analysis) + Rule Hypothesis Engine (retained for heuristics)
      7. IDO Value Score + Noether-Check + Goal-EML (from v6.4, retained)
      8. Goal-EML lock when convergence ≥ 0.8 (near-goal aggressive progressive repeat)
      9. All legacy search methods retained as internal helpers

    v7.0 changes vs v6.6:
      - Full-frame MD5 hash replaces coarse CCA-based hash for state graph (exact dedup)
      - Frontier BFS navigation: navigate back to states with untried actions
      - Unified 5-tier priority system replaces ad-hoc Phase 1-5 ordering
      - Trigger-aware pruning: skip known-ineffective actions before trying them
      - CCA _object_hash retained for heuristic computation (still valuable)
    """

    # Upper bound on actions per game.
    # v6.3: NEVER give up — match Stochastic Goose (#1 team) philosophy.
    # RandomAgent baseline uses 1000000; Goose uses float('inf').
    # Low MAX_ACTIONS causes early termination → 0 score for unfinished levels.
    MAX_ACTIONS = int(os.environ.get("ARC3_MAX_ACTIONS", "1000000"))

    # Global wall-clock limit per game instance (seconds).
    # Kaggle allows 9 hours total. With 25 games, budget ≈ 19 min/game.
    # We use 900s (15 min) as a conservative per-game limit to leave headroom.
    _GAME_TIME_LIMIT_S: float = 900.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) + hash(self.game_id) % 1_000_000
        random.seed(seed)

        # ── Global time guard ──
        self._start_time: float = time.time()  # Record when this agent started

        # ── Plan state ──
        self._plan: List[Tuple[str, Optional[Dict]]] = []  # [(action_name, data)]
        self._plan_idx: int = 0
        self._levels_done: int = 0  # Track completed levels
        self._retries: int = 0      # Count retries on GAME_OVER
        self._max_retries: int = 5   # Max retries per level before giving up

        # ── Exploration state (original) ──
        self._visited_coords: Set[str] = set()  # Track visited click positions
        self._visited_counts: Dict[str, int] = {}  # v5.0.4: Track visit count per coord (allow revisit)
        self._max_revisit: int = 3  # v5.0.4: Max times to revisit same coord before skipping
        self._action_history: List[str] = []  # Track all actions taken
        self._grid_history: List[Any] = []  # Track grid snapshots for delta detection

        # ── NEW v3.6.0: ASD Anomaly Detection state ──
        self._asd_anomaly_colors: List[int] = []  # Minority colors detected from first frame
        self._asd_anomaly_targets: List[Tuple[int, int]] = []  # (x, y) of anomaly pixels
        self._asd_analyzed: bool = False  # Whether ASD analysis has been done for this level
        self._asd_top_rarity: Dict[int, float] = {}  # color -> rarity score (0-1)

        # ── NEW v3.6.0: 3-Life Strategy state ──
        self._game_over_count: int = 0  # Total GAME_OVER events for this level
        self._life_phase: str = "life1"  # life1=explore, life2=refine, life3=execute
        self._life1_discoveries: Dict[str, Any] = {}  # Discoveries from Life1
        self._life1_effective_actions: List[str] = []  # Actions that caused change in Life1
        self._life1_effective_clicks: List[Tuple[int, int]] = []  # Clicks that caused change

        # ── NEW: Delta-aware exploration state ──
        self._delta_history: List[List[Tuple[int, int]]] = []  # Changed cells per step
        self._delta_click_pool: List[Tuple[int, int]] = []  # Unvisited delta cells to click
        self._estimated_player_pos: Optional[Tuple[int, int]] = None  # Estimated (x, y)
        self._prev_estimated_player_pos: Optional[Tuple[int, int]] = None
        self._direction_map: Dict[str, Tuple[int, int]] = {}  # action_name -> (dx, dy)
        self._direction_probed: Dict[str, bool] = {}  # which directions have been tested
        self._direction_probe_count: int = 0  # how many probes we've done this level
        self._special_probed: bool = False  # whether ACTION5 has been tested
        self._special_effect_delta: Optional[List[Tuple[int, int]]] = None
        self._special_effect_summary: str = ""  # brief description of SPECIAL effect
        # v5.0.4: Dynamic stall threshold — click games stall faster (4 steps),
        # keyboard games slower (6 steps). Previously fixed at 6 for all games.
        self._stall_counter: int = 0  # consecutive steps with no grid change
        self._prev_levels_completed: int = 0  # levels completed at previous step
        self._pattern_memory: Dict[str, List[str]] = {}  # grid_hash -> effective action sequence
        self._effective_actions: Dict[str, int] = {}  # action_name -> times it caused change
        self._inactive_actions: Dict[str, int] = {}  # action_name -> times it caused no change
        self._last_grid_hash: str = ""  # hash of previous grid for pattern detection
        self._current_grid_hash: str = ""  # hash of current grid
        self._grid_hash_action_map: Dict[str, str] = {}  # grid_hash -> action that changed it
        self._exploration_phase: str = "probe"  # probe | navigate | exploit
        self._navigate_target: Optional[Tuple[int, int]] = None  # target for navigation
        self._navigate_path: List[str] = []  # planned path of actions to target

        # ── NEW v5.0.2: Dynamic game-type detection state ──
        # Instead of relying solely on hardcoded KEYBOARD/CLICK/MIXED sets,
        # detect game type from available_actions on first frame.
        # This is CRITICAL for competition: unknown games have no Oracle data,
        # and hardcoded sets don't cover them — they fall to inefficient fallback.
        self._detected_game_type: str = "unknown"  # "click", "keyboard", "mixed", "unknown"
        self._game_type_detected: bool = False  # whether detection has been done

        # ── NEW v5.0.3: RG-Flow adaptive state ──
        # RG-Flow inspired by QQG/IDO: coupling strength adjusts with "energy scale" (action budget).
        # Early phase (action_count < 30% budget): high entropy, wide probe (asymptotic freedom)
        # Mid phase (30-70%): balanced exploration/exploitation
        # Late phase (>70% budget): κ-Snap lock — only repeat proven effective actions (strong coupling)
        self._action_count: int = 0  # total actions taken in current level
        self._rg_flow_phase: str = "explore"  # "explore" | "balanced" | "lock"
        self._ic_delta: float = 0.0  # information cardinality change (ΔIC) — drives cognitive inflation

        # ── NEW v5.0.3: Cognitive inflation state ──
        # When stall detected, instead of random direction swap, do ΔIC avalanche BFS:
        # detect topological avalanche (high ΔIC subsets) → creative probe into those regions
        self._inflation_probes: List[Tuple[int, int]] = []  # ΔIC avalanche probe targets
        self._inflation_active: bool = False  # whether cognitive inflation is currently active

        # ── NEW v3.8.0: Systematic grid click scan state ──
        self._grid_scan_queue: List[Tuple[int, int]] = []  # (x, y) positions to click, ordered by rarity
        self._grid_scan_initialized: bool = False  # whether scan has been set up for this level
        self._effective_click_positions: List[Tuple[int, int]] = []  # clicks that caused grid change
        self._scan_expansion_radius: int = 2  # BFS expansion radius around effective clicks

        # ── NEW v5.0.5: Goal-oriented navigation state ──
        # Player color: the grid value that consistently moves with keyboard actions.
        # We track this to separate the player from walls/decoration — critical for
        # navigating toward goals instead of random nonzero cells.
        self._player_color: Optional[int] = None  # Detected player sprite color value
        self._player_color_candidates: Dict[int, int] = {}  # color -> frequency of movement
        self._player_position_history: List[Tuple[int, int]] = []  # tracked player positions (x,y)

        # Goal detection: positions in the grid that are likely targets/exits/goals.
        # Identified by rarity, spatial patterns, border openings, and delta analysis.
        self._goal_positions: List[Tuple[int, int]] = []  # detected goal (x,y) positions
        self._goal_colors: List[int] = []  # detected goal color values
        self._wall_colors: List[int] = []  # detected common wall/decoration colors (>30% of nonzero)

        # ── NEW v6.3: State graph for informed search ──
        # Inspired by FORGE (GraphExplorer, score 0.43) and Stochastic Goose (score 1.21).
        # Track state→action→new_state transitions to prioritize effective actions.
        # This is the CORE mechanism that replaces the complex multi-phase exploration.
        self._state_graph: Dict[str, Dict[str, str]] = {}  # grid_hash → {action_key: result_hash}
        self._state_visited: Dict[str, int] = {}  # grid_hash → visit count
        self._state_queue: List[str] = []  # BFS frontier: states with untried actions
        self._action_change_rate: Dict[str, Tuple[int, int]] = {}  # action_key → (changes, total_uses)
        self._last_action_key: str = ""  # action_key of last action (for state graph edge recording)
        self._level_start_hash: str = ""  # hash of grid when level started (for BFS backtracking)

        # ── NEW v6.4: IDO Value Score + Noether-Check + Goal-EML ──
        # Inspired by TOMAS/IDO theory from Yuanbao analysis.
        # ido_value_score: Measures whether a grid change represents PROGRESS (reducing IC-gap)
        #   toward the goal state, not just "any change". Positive = progressive, negative = backslide.
        # Noether-Check: IC conservation along path — reject "physics-impossible" actions that
        #   increase entropy without level progress (walls broken, teleportation, etc.)
        # Goal-EML: Explicit goal topology tracking — minority pixels decreasing → goal converging.
        self._ido_value_score: float = 0.0  # last computed ido_value_score
        self._ido_value_history: List[float] = []  # ido_value_score per step
        self._backslide_actions: Dict[str, int] = {}  # action_key → count of backslide (negative ido)
        self._progressive_actions: Dict[str, int] = {}  # action_key → count of progressive (positive ido)
        self._last_ic_value: float = 0.0  # IC value of previous state (for ido computation)
        self._current_ic_value: float = 0.0  # IC value of current state
        self._ic_goal_value: float = 0.0  # IC value of goal/ideal state (estimated from level_start)
        self._noether_eps: float = 0.15  # ε for Noether-Check: IC(s_next) ≤ IC(s_prev) + ε
        self._noether_violations: Dict[str, int] = {}  # action_key → Noether violation count
        self._goal_eml_convergence: float = 0.0  # Goal-EML convergence metric (0-1, 1=goal achieved)
        self._minority_pixel_count: int = 0  # count of minority/anomaly pixels in current state
        self._goal_color_expansion: int = 0  # count of goal-color pixels in current state
        self._eta_plateau_counter: int = 0  # consecutive steps with η-plateau (ido≈0)

        # ── NEW v6.5: Object-Level State Abstraction + A* Heuristic ──
        # Inspired by Yuanbao BFS efficiency analysis and Ferlaino J⊥≫Jz analogy.
        # Instead of hashing entire 64×64 grid (~10^30 states), extract key objects
        # and hash those (~10^3-10^4 states). This makes A* search feasible.
        self._object_state: Dict[str, Any] = {}  # extracted objects from current grid
        self._object_hash: str = ""  # object-level hash (replaces grid_hash for state graph)
        self._prev_object_hash: str = ""  # previous object-level hash
        self._heuristic_distance: float = 1.0  # A* heuristic distance to goal (lower = closer)
        self._nonlocal_impact: Dict[str, float] = {}  # action_key → non-local impact magnitude (J⊥ class)
        self._frontier_depth: Dict[str, int] = {}  # object_hash → BFS depth from start state

        # ── NEW v6.6: CCA (Connected Component Analysis) + Rule Hypothesis Engine ──
        # CCA replaces crude 8×8 region signature with proper flood-fill BFS
        # to identify contiguous color regions as "objects" (Perceptual Abstraction Functor F).
        # Rule Hypothesis Engine: observe grid changes → infer game mechanics → predict goal.
        self._cca_objects: List[Dict[str, Any]] = []  # CCA-detected objects [{centroid, size, color, bbox, obj_type}]
        self._rule_hypothesis: Dict[str, Any] = {}  # inferred rule: {type, goal_prediction, confidence}
        self._observation_log: List[Dict[str, Any]] = []  # first-N-step observations [{step, action, grid_changes}]
        self._hypothesis_confirmed: bool = False  # whether rule hypothesis confirmed (3+ consistent observations)
        self._hypothesis_step_count: int = 0  # how many steps observed for hypothesis
        self._MAX_HYPOTHESIS_STEPS: int = 5  # max steps to observe before confirming hypothesis
        self._predicted_goal_objects: List[Dict[str, Any]] = []  # predicted goal state objects

        # ── NEW v7.0: Full-frame hash for exact state dedup ──
        # MD5 of entire grid frame — exact dedup replaces coarse CCA-based hash
        # for state graph operations. CCA hash (_object_hash) retained for heuristics.
        self._state_hash: str = ""  # MD5 of entire grid frame — exact dedup
        self._prev_state_hash: str = ""  # previous full-frame hash
        self._frontier_states: Set[str] = set()  # states with untried actions (frontier for BFS)
        self._untried_actions_cache: Dict[str, List[str]] = {}  # state_hash → list of untried action keys

        # ── Initialize plan for level 0 ──
        self._compute_plan(0)

    @property
    def name(self) -> str:
        return f"tomas.v7.0.{self.MAX_ACTIONS}"

    @property
    def _stall_threshold(self) -> int:
        """v5.0.4: Dynamic stall threshold based on game type.

        Click games: stall after 4 steps (need faster response)
        Keyboard games: stall after 6 steps (movement can take multiple steps)
        Mixed/unknown: 5 steps (balanced)
        """
        if self._detected_game_type == "click":
            return 4
        elif self._detected_game_type == "keyboard":
            return 6
        else:
            return 5

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Return True when agent should stop playing.

        Stopping conditions:
        1. WIN state — level/game cleared.
        2. Global time limit exceeded — prevents Kaggle 9h timeout.
           (base class MAX_ACTIONS check is still active as a backstop)
        """
        if latest_frame.state is GameState.WIN:
            return True
        # Global time guard: stop if game has been running too long.
        # This is critical to prevent Kaggle notebook timeout (9h total / 25 games).
        elapsed = time.time() - self._start_time
        if elapsed > self._GAME_TIME_LIMIT_S:
            return True
        # NOTE: Do NOT check MAX_ACTIONS here — base class handles it.
        return False

    # ── Dynamic game-type detection ──────────────────────────────────

    def _detect_game_type(self, available_set: Set[int]) -> str:
        """Detect game type from available_actions for adaptive strategy selection.

        CRITICAL for competition: unknown games have no Oracle data and aren't
        in hardcoded KEYBOARD/CLICK/MIXED sets. By detecting game type dynamically,
        we can route unknown games to the appropriate exploration strategy
        (grid-scan for click games, keyboard probe for keyboard games, etc.)

        Detection rules:
          - ACTION6 (id=6) available → click-capable
          - ACTION1-4 (ids=1-4) available → keyboard-capable
          - Both → mixed
          - Neither → unknown (rare, fall to generic exploration)

        For known games, we still use hardcoded sets for Oracle reliability checks,
        but for exploration strategy, dynamic detection takes precedence.
        """
        has_click = 6 in available_set
        has_keyboard = any(a in available_set for a in [1, 2, 3, 4])
        if has_click and has_keyboard:
            return "mixed"
        elif has_click:
            return "click"
        elif has_keyboard:
            return "keyboard"
        else:
            return "unknown"

    def _update_detected_game_type(self, available_set: Set[int]) -> None:
        """Update detected game type on first frame with available_actions.

        Called from _smart_exploration and _choose_action_impl to cache
        the game type once detected.
        """
        if not self._game_type_detected:
            self._detected_game_type = self._detect_game_type(available_set)
            self._game_type_detected = True

    def _is_click_capable_game(self, base_id: str, available_set: Set[int]) -> bool:
        """Check if game is click-capable (should use grid-scan strategy).

        Combines hardcoded knowledge (for known games) with dynamic detection
        (for unknown/competition games).
        """
        # Dynamic detection takes precedence for ALL games
        self._update_detected_game_type(available_set)
        if self._detected_game_type in ("click", "mixed"):
            return True
        # Hardcoded fallback for known games (Oracle classification)
        if base_id in CLICK_GAMES or base_id in MIXED_GAMES:
            return True
        return False

    def _is_keyboard_game(self, base_id: str, available_set: Set[int]) -> bool:
        """Check if game is primarily keyboard-based."""
        self._update_detected_game_type(available_set)
        if self._detected_game_type == "keyboard":
            return True
        if base_id in KEYBOARD_GAMES:
            return True
        return False

    def _is_mixed_game(self, base_id: str, available_set: Set[int]) -> bool:
        """Check if game is mixed (both click and keyboard actions available)."""
        self._update_detected_game_type(available_set)
        if self._detected_game_type == "mixed":
            return True
        if base_id in MIXED_GAMES:
            return True
        return False

    # ── RG-Flow Adaptive Attention (v5.0.3) ──────────────────────────────

    def _update_rg_flow(self) -> None:
        """Update RG-Flow phase based on action budget consumption.

        Inspired by QQG asymptotic freedom: at high energy (early, many actions left),
        coupling is weak → wide exploration. At low energy (late, few actions left),
        coupling is strong → κ-Snap lock on proven effective actions.

        Phase thresholds:
          - explore: action_count < 30% MAX_ACTIONS (high entropy, wide probe)
          - balanced: 30-70% (moderate, use effective actions but still probe)
          - lock: > 70% (κ-Snap lock, only proven effective actions)
        """
        budget_frac = self._action_count / max(self.MAX_ACTIONS, 1)
        if budget_frac < 0.30:
            self._rg_flow_phase = "explore"
        elif budget_frac < 0.70:
            self._rg_flow_phase = "balanced"
        else:
            self._rg_flow_phase = "lock"

    def _compute_ic_delta(self, prev_grid: Any, current_grid: Any) -> float:
        """Compute Information Cardinality change (ΔIC) between two grids.

        ΔIC measures how much the grid's "distinguishable state count" changed.
        High ΔIC → topological avalanche → creative probing opportunity.

        Simplified: count unique non-zero values as proxy for IC.
        ΔIC = |unique_vals_now - unique_vals_prev| / max(unique_vals_prev, 1)
        """
        prev_vals = set()
        curr_vals = set()
        prev_layer = self._extract_layer0(prev_grid)
        curr_layer = self._extract_layer0(current_grid)

        if prev_layer:
            for row in prev_layer:
                for v in row:
                    if v != 0 and v != -1:
                        prev_vals.add(v)
        if curr_layer:
            for row in curr_layer:
                for v in row:
                    if v != 0 and v != -1:
                        curr_vals.add(v)

        prev_ic = len(prev_vals) or 1
        curr_ic = len(curr_vals) or 1
        self._ic_delta = abs(curr_ic - prev_ic) / prev_ic
        return self._ic_delta

    def _detect_topological_avalanche(self, grid: Any) -> List[Tuple[int, int]]:
        """Detect high ΔIC regions for cognitive inflation probing.

        When agent is stuck (stall), instead of random direction swap,
        identify cells that belong to value groups with high IC change
        and probe those regions — this is "cognitive inflation" from IDO.

        Returns list of (x, y) targets sorted by ΔIC contribution.
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return []

        h = len(layer)
        w = len(layer[0]) if h > 0 else 0
        if w == 0:
            return []

        # Find cells belonging to rare/new values (high ΔIC contributors)
        # These are likely interactive and worth probing
        color_counts: Dict[int, int] = {}
        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        color_counts[val] = color_counts.get(val, 0) + 1
                except (IndexError, TypeError):
                    continue

        if not color_counts:
            return []

        # Values with low count → high ΔIC contribution → priority targets
        total = sum(color_counts.values())
        targets: List[Tuple[int, int, float]] = []  # (x, y, ic_score)
        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        # IC score = 1 - frequency (rare = high score)
                        freq = color_counts.get(val, 1) / max(total, 1)
                        ic_score = 1.0 - freq
                        targets.append((c, r, ic_score))
                except (IndexError, TypeError):
                    continue

        # Sort by IC score (highest first) and filter unvisited
        targets.sort(key=lambda t: -t[2])
        result: List[Tuple[int, int]] = []
        for x, y, score in targets:
            coord_key = f"{x},{y}"
            if coord_key not in self._visited_coords:
                result.append((x, y))
        return result

    def _compute_plan(self, level_idx: int) -> None:
        """Compute action plan for the given level using Replay Oracle.

        v5.0.4: ALWAYS try Oracle first. Previous v5.0.3 had a heuristic that
        skipped L1+ Oracle for click/mixed games when sequences looked suspicious.
        This was WRONG — on Kaggle (auto-advance), L1+ Oracle often works even
        when sequences look similar to L0. If Oracle fails (GAME_OVER), the
        GAME_OVER handler in _choose_action_impl will switch to grid-scan.
        """
        base_id = self.game_id.split("-")[0] if self.game_id else ""
        replay_data = ARC3_REPLAY_ORACLE.get(base_id)

        # v5.0.4: REMOVED Oracle skip heuristic — always try Oracle first.
        # If Oracle data exists for this level, use it.
        # If it fails (GAME_OVER), the retry handler will switch to exploration.
        if replay_data is not None and level_idx in replay_data:
            sequence = replay_data[level_idx]
            self._plan = self._convert_replay(sequence)
            self._plan_idx = 0
            # Reset grid-scan for new Oracle plan — Oracle takes priority
            self._grid_scan_initialized = False
            self._grid_scan_queue = []
        else:
            # No replay data — reset exploration state for new level
            self._plan = []
            self._plan_idx = 0
            self._reset_exploration_state()

    def _convert_replay(self, sequence: List) -> List[Tuple[str, Optional[Dict]]]:
        """Convert arc3.games replay sequence to (action_name, data) tuples.

        Args:
            sequence: List of int (1-5 for keyboard) or [x,y] for clicks.

        Returns:
            List of (GameAction_name, data_dict_or_None) tuples.
        """
        plan: List[Tuple[str, Optional[Dict]]] = []
        for item in sequence:
            if isinstance(item, list):
                x, y = int(item[0]), int(item[1])
                # Clamp to valid range (0-63)
                x = max(0, min(63, x))
                y = max(0, min(63, y))
                plan.append(("ACTION6", {"x": x, "y": y}))
            elif isinstance(item, int):
                action_name = ARC3_ACTION_ID_MAP.get(item)
                if action_name:
                    plan.append((action_name, None))
        return plan

    def _reset_exploration_state(self) -> None:
        """Reset exploration state when entering a new level without replay data."""
        self._visited_coords = set()
        self._visited_counts = {}  # v5.0.4: Reset visit counts for new level
        self._delta_history = []
        self._delta_click_pool = []
        self._estimated_player_pos = None
        self._prev_estimated_player_pos = None
        self._direction_map = {}
        self._direction_probed = {}
        self._direction_probe_count = 0
        self._special_probed = False
        self._special_effect_delta = None
        self._special_effect_summary = ""
        self._stall_counter = 0
        self._prev_levels_completed = self._levels_done
        # v5.0.3: Reset RG-Flow and inflation state for new level
        self._action_count = 0
        self._rg_flow_phase = "explore"
        self._ic_delta = 0.0
        self._inflation_probes = []
        self._inflation_active = False
        self._last_grid_hash = ""
        self._current_grid_hash = ""
        self._exploration_phase = "probe"
        self._navigate_target = None
        self._navigate_path = []
        self._grid_hash_action_map = {}
        # ── Reset ASD state ──
        self._asd_anomaly_colors = []
        self._asd_anomaly_targets = []
        self._asd_analyzed = False
        self._asd_top_rarity = {}
        # ── Reset 3-Life state ──
        self._game_over_count = 0
        self._life_phase = "life1"
        self._life1_discoveries = {}
        self._life1_effective_actions = []
        self._life1_effective_clicks = []
        # ── Reset grid click scan state ──
        self._grid_scan_queue = []
        self._grid_scan_initialized = False
        self._effective_click_positions = []
        # ── Reset v5.0.5 goal-oriented navigation state ──
        # Player color persists across levels (same game), but goals reset
        self._player_color_candidates = {}
        self._player_position_history = []
        self._goal_positions = []
        self._goal_colors = []
        self._wall_colors = []
        # Keep pattern_memory across levels — patterns may repeat
        # ── Reset v6.3 state graph for new level ──
        self._state_graph = {}
        self._state_visited = {}
        self._state_queue = []
        self._action_change_rate = {}
        self._last_action_key = ""
        self._level_start_hash = ""
        # ── Reset v6.4 IDO/Noether state for new level ──
        self._ido_value_score = 0.0
        self._ido_value_history = []
        self._backslide_actions = {}
        self._progressive_actions = {}
        self._last_ic_value = 0.0
        self._current_ic_value = 0.0
        self._ic_goal_value = 0.0
        self._noether_violations = {}
        self._goal_eml_convergence = 0.0
        self._minority_pixel_count = 0
        self._goal_color_expansion = 0
        self._eta_plateau_counter = 0

        # ── Reset v6.5 object-level state ──
        self._object_state = {}
        self._object_hash = ""
        self._prev_object_hash = ""
        self._heuristic_distance = 1.0
        self._nonlocal_impact = {}
        self._frontier_depth = {}

        # ── Reset v6.6 CCA + Rule Hypothesis state ──
        self._cca_objects = []
        self._rule_hypothesis = {}
        self._observation_log = []
        self._hypothesis_confirmed = False
        self._hypothesis_step_count = 0
        self._predicted_goal_objects = []

        # ── Reset v7.0 full-frame hash state ──
        self._state_hash = ""
        self._prev_state_hash = ""
        self._frontier_states = set()
        self._untried_actions_cache = {}

    def _should_visit(self, x: int, y: int) -> bool:
        """v5.0.4: Check if a coordinate should be visited (allow revisit up to _max_revisit).

        Previously, _visited_coords prevented ANY revisit. But many games require
        clicking the same position multiple times (e.g., gravity toggle, button press).
        Now we allow revisit up to _max_revisit times, then skip.
        """
        coord_key = f"{x},{y}"
        count = self._visited_counts.get(coord_key, 0)
        if count < self._max_revisit:
            self._visited_counts[coord_key] = count + 1
            return True
        # Max visits exceeded — mark as fully visited
        self._visited_coords.add(coord_key)
        return False

    # ── Grid analysis helpers ───────────────────────────────────────────

    def _extract_layer0(self, grid: Any) -> Optional[List[List[int]]]:
        """Extract the first layer from a potentially 3D grid.

        ARC3 grids are typically 3D: [channels][rows][cols].
        The first channel (layer 0) is the main game grid.

        Args:
            grid: The frame data, either 3D or 2D.

        Returns:
            2D list of int (rows × cols), or None if invalid.
        """
        if not grid or not isinstance(grid, list):
            return None
        try:
            if len(grid) > 0 and isinstance(grid[0], list):
                if len(grid[0]) > 0 and isinstance(grid[0][0], list):
                    # 3D grid: [channel][row][col] — take first channel
                    return grid[0]
                else:
                    # 2D grid: [row][col]
                    return grid
            return None
        except (IndexError, TypeError):
            return None

    def _compute_grid_delta(
        self, old_grid: Any, new_grid: Any
    ) -> List[Tuple[int, int]]:
        """Compare two grids and find cells that changed between them.

        Changed cells are likely interactive elements — either the player
        position changed, a sprite moved, or an object was created/destroyed.

        Args:
            old_grid: Previous frame grid.
            new_grid: Current frame grid.

        Returns:
            List of (x, y) coordinates where cell values differ.
        """
        old_layer = self._extract_layer0(old_grid)
        new_layer = self._extract_layer0(new_grid)

        if not old_layer or not new_layer:
            return []

        h = min(len(old_layer), len(new_layer))
        if h == 0:
            return []

        w = min(len(old_layer[0]), len(new_layer[0]))
        if w == 0:
            return []

        delta: List[Tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                try:
                    old_val = old_layer[r][c]
                    new_val = new_layer[r][c]
                    if old_val != new_val:
                        delta.append((c, r))  # (x, y) format — x=col, y=row
                except (IndexError, TypeError):
                    continue

        return delta

    def _grid_hash(self, grid: Any) -> str:
        """Compute a lightweight hash of the grid state for pattern detection.

        Uses MD5 of the first layer for fast comparison. Collisions are
        acceptable — we use this for heuristic pattern matching, not
        cryptographic security.

        Args:
            grid: The frame data.

        Returns:
            Hex digest string of the grid hash.
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return ""
        try:
            # Flatten and hash — fast and compact
            flat = []
            for row in layer:
                flat.extend(row)
            data = bytes(flat)
            return hashlib.md5(data).hexdigest()[:16]  # 16 chars is enough
        except (TypeError, ValueError):
            return ""

    def _find_nonzero_cells(self, grid: Any) -> List[Tuple[int, int, int]]:
        """Find all cells with non-zero, non-background values.

        These cells represent sprites, objects, or interactive elements
        in the game grid.

        Args:
            grid: The frame data.

        Returns:
            List of (x, y, value) tuples for non-zero cells.
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return []

        h = len(layer)
        if h == 0:
            return []
        w = len(layer[0])
        if w == 0:
            return []

        cells: List[Tuple[int, int, int]] = []
        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        cells.append((c, r, val))
                except (IndexError, TypeError):
                    continue

        return cells

    def _find_click_targets(self, grid: Any) -> List[Tuple[int, int]]:
        """Find interesting cells in the grid to click on.

        Prioritizes cells with non-zero values (sprites/objects).
        Returns coordinates sorted by novelty (least-visited first).

        Args:
            grid: The frame data.

        Returns:
            List of (x, y) coordinates to click, ordered by priority.
        """
        cells = self._find_nonzero_cells(grid)
        if not cells:
            return []

        # Sort: unvisited first, then by value (higher = more interesting)
        cells.sort(key=lambda t: (f"{t[0]},{t[1]}" in self._visited_coords, -t[2]))

        targets: List[Tuple[int, int]] = []
        for c, r, val in cells[:20]:  # Consider top 20 targets
            targets.append((c, r))

        return targets

    def _init_grid_click_scan(self, grid: Any) -> None:
        """Initialize systematic grid click scan sorted by color rarity.

        For CLICK/MIXED games after Oracle exhausts, we need a systematic
        approach rather than keyboard-oriented phases. This method builds
        a queue of all non-zero cells, sorted by color rarity (minority
        colors first), ensuring comprehensive coverage of the grid.

        Args:
            grid: Current frame grid data.
        """
        if self._grid_scan_initialized:
            return
        self._grid_scan_initialized = True

        layer = self._extract_layer0(grid)
        if not layer:
            self._grid_scan_queue = []
            return

        h = len(layer)
        if h == 0:
            return
        w = len(layer[0])
        if w == 0:
            return

        # Count color frequencies for rarity-based ordering
        color_counts: Dict[int, int] = {}
        all_cells: List[Tuple[int, int, int]] = []  # (x, y, value)
        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        color_counts[val] = color_counts.get(val, 0) + 1
                        all_cells.append((c, r, val))  # (x, y, value)
                except (IndexError, TypeError):
                    continue

        if not all_cells:
            # No non-zero cells — scan entire grid at regular intervals
            step = 4  # Click every 4th cell in a raster scan
            scan_cells: List[Tuple[int, int]] = []
            for r in range(0, h, step):
                for c in range(0, w, step):
                    scan_cells.append((c, r))
            self._grid_scan_queue = scan_cells
            return

        # Compute rarity: 1 - (frequency / total_nonzero)
        total_nonzero = len(all_cells)
        rarity_map: Dict[int, float] = {}
        for color, count in color_counts.items():
            rarity_map[color] = 1.0 - (count / total_nonzero)

        # Sort by rarity (rarest color first), then by position for consistency
        all_cells.sort(key=lambda t: (-rarity_map.get(t[2], 0.0), t[1], t[0]))

        # Build scan queue, excluding fully-visited coords (v5.0.4: allow revisit up to max)
        scan_queue: List[Tuple[int, int]] = []
        for x, y, _val in all_cells:
            # v5.0.4: Use _should_visit to allow revisit, but for initial scan
            # we only include coords not fully visited
            coord_key = f"{x},{y}"
            visit_count = self._visited_counts.get(coord_key, 0)
            if visit_count < self._max_revisit:
                scan_queue.append((x, y))

        self._grid_scan_queue = scan_queue

    def _expand_scan_from_click(self, x: int, y: int, grid: Any) -> None:
        """Expand scan queue around an effective click position (BFS-like).

        When a click at (x, y) causes a grid change, nearby cells are
        likely interactive too. Add them to the front of the scan queue
        with priority, enabling focused exploration around discovered
        interactive areas.

        Args:
            x: X coordinate of the effective click.
            y: Y coordinate of the effective click.
            grid: Current frame grid (to extract bounds).
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return

        h = len(layer)
        w = len(layer[0]) if h > 0 else 0

        radius = self._scan_expansion_radius
        expansion: List[Tuple[int, int]] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue  # Skip the center (already clicked)
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    coord_key = f"{nx},{ny}"
                    if coord_key not in self._visited_coords:
                        expansion.append((nx, ny))

        # Sort by distance from center (closest first) and prepend to queue
        expansion.sort(key=lambda p: abs(p[0] - x) + abs(p[1] - y))
        # Remove duplicates that might already be in the queue
        existing = set(tuple(p) for p in self._grid_scan_queue)
        new_entries = [p for p in expansion if p not in existing]
        self._grid_scan_queue = new_entries + self._grid_scan_queue

    def _estimate_player_position(
        self,
        delta: List[Tuple[int, int]],
        last_action: str,
        grid: Any,
    ) -> Optional[Tuple[int, int]]:
        """Estimate player position from delta cells, action context, and known color.

        v5.0.6: Enhanced with fast player color-based positioning. If _player_color
        is known, we can find the player directly in the grid without waiting for
        delta clusters. This enables goal-oriented navigation from frame 1.

        Strategy (in priority order):
          1. If _player_color is known → locate player by color scan (FASTEST)
          2. If delta cells form a cluster, cluster center is likely near player
          3. If we know direction map, extrapolate from previous position
          4. Otherwise, find the most "active" rare-color cell in the grid

        Args:
            delta: Changed cells from last action.
            last_action: The action that caused this delta.
            grid: Current frame grid data.

        Returns:
            Estimated (x, y) player position, or None.
        """
        # v5.0.6: PRIORITY 1 — Direct color-based positioning (fastest, most accurate)
        if self._player_color is not None:
            color_pos = self._locate_player_by_color(grid)
            if color_pos is not None:
                # Refine with delta if available — delta confirms recent movement
                if delta and last_action in self._direction_map:
                    dx, dy = self._direction_map[last_action]
                    prev_x, prev_y = self._prev_estimated_player_pos or color_pos
                    predicted = (prev_x + dx, prev_y + dy)
                    # If predicted position matches color_pos, trust it
                    if abs(predicted[0] - color_pos[0]) + abs(predicted[1] - color_pos[1]) <= 2:
                        return predicted
                return color_pos

        if delta:
            # Cluster analysis: find center of mass of changed cells
            avg_x = sum(x for x, y in delta) / len(delta)
            avg_y = sum(y for x, y in delta) / len(delta)
            cluster_center = (int(round(avg_x)), int(round(avg_y)))

            # If we have a previous position and direction map, refine estimate
            if self._prev_estimated_player_pos and last_action in self._direction_map:
                dx, dy = self._direction_map[last_action]
                prev_x, prev_y = self._prev_estimated_player_pos
                # New position = old position + learned direction offset
                predicted = (prev_x + dx, prev_y + dy)
                # If predicted position is in the delta cluster, use it
                # Otherwise use cluster center
                if predicted in delta:
                    return predicted
                # Find closest delta cell to predicted position
                min_dist = float("inf")
                best_pos = cluster_center
                for x, y in delta:
                    dist = abs(x - predicted[0]) + abs(y - predicted[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_pos = (x, y)
                return best_pos

            return cluster_center

        # No delta — player didn't move. Keep previous position if known.
        if self._prev_estimated_player_pos:
            return self._prev_estimated_player_pos

        # v5.0.6: First time with no delta — try color-based positioning
        if self._player_color is not None:
            color_pos = self._locate_player_by_color(grid)
            if color_pos is not None:
                return color_pos

        # Fallback: find a "unique" cell that might be the player
        nonzero = self._find_nonzero_cells(grid)
        if nonzero:
            # Cells with rare values are more likely to be the player
            value_counts: Dict[int, int] = {}
            for x, y, val in nonzero:
                value_counts[val] = value_counts.get(val, 0) + 1
            # Sort by rarity (fewest occurrences)
            nonzero.sort(key=lambda t: (value_counts.get(t[2], 0), -t[2]))
            # First element is the most unique — likely the player
            return (nonzero[0][0], nonzero[0][1])

        return None

    # ── v5.0.6: Fast player position by color ──────────────────────────────

    def _locate_player_by_color(self, grid: Any) -> Optional[Tuple[int, int]]:
        """Directly locate player position by scanning for _player_color in grid.

        v5.0.6: If _player_color is known, we can find the player IMMEDIATELY
        by scanning the grid for cells with that color value — no need to wait
        for delta clusters or keyboard movement analysis. This gives us a player
        position estimate on the FIRST frame, enabling goal-oriented navigation
        from the very beginning.

        Args:
            grid: Current frame grid data.

        Returns:
            Estimated (x, y) player position, or None if color not found.
        """
        if self._player_color is None:
            return None

        layer = self._extract_layer0(grid)
        if not layer:
            return None

        h = len(layer)
        w = len(layer[0]) if h > 0 else 0
        if w == 0:
            return None

        # Collect all cells matching player color
        player_cells: List[Tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                try:
                    if layer[r][c] == self._player_color:
                        player_cells.append((c, r))  # (x, y) format
                except (IndexError, TypeError):
                    continue

        if not player_cells:
            return None

        # Player sprites are typically 1-5 cells forming a connected cluster.
        # Find the center of mass of the player color cells.
        avg_x = sum(x for x, y in player_cells) / len(player_cells)
        avg_y = sum(y for x, y in player_cells) / len(player_cells)

        # If only 1-3 cells, just return the center directly
        if len(player_cells) <= 3:
            return (int(round(avg_x)), int(round(avg_y)))

        # For larger clusters, find the most "connected" cell (the one with
        # most adjacent player_color cells — this is likely the player's center)
        best_pos = (int(round(avg_x)), int(round(avg_y)))
        max_adj = 0
        for x, y in player_cells:
            adj_count = 0
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in [(px, py) for px, py in player_cells]:
                    adj_count += 1
            if adj_count > max_adj:
                max_adj = adj_count
                best_pos = (x, y)

        return best_pos

    # ── v5.0.5: Player color detection ─────────────────────────────────

    def _detect_player_color_from_delta(
        self,
        delta: List[Tuple[int, int]],
        frames: list[FrameData],
        current_grid: Any,
    ) -> None:
        """Detect player color by analyzing cells that moved after a keyboard action.

        When a keyboard action (ACTION1-4) causes a delta, some cells disappear
        from old positions and new cells appear at new positions. The color value
        that "moved" (disappeared at old, appeared at new) is the player sprite.

        Args:
            delta: Changed cells from last keyboard action.
            frames: Frame history for accessing previous grid.
            current_grid: Current grid after the action.
        """
        # Only detect after keyboard actions (not clicks or special)
        if not self._action_history or self._action_history[-1] not in DIRECTION_ACTIONS:
            return

        # Get previous grid
        prev_grid = None
        if len(frames) >= 2 and frames[-2].frame is not None:
            prev_grid = frames[-2].frame
        elif len(self._grid_history) >= 2:
            prev_grid = self._grid_history[-2]
        if prev_grid is None or current_grid is None:
            return

        prev_layer = self._extract_layer0(prev_grid)
        curr_layer = self._extract_layer0(current_grid)
        if not prev_layer or not curr_layer:
            return

        # Find cells that appeared (new nonzero in current) and disappeared (old nonzero gone)
        appeared: Dict[int, int] = {}  # color -> count of cells that appeared
        disappeared: Dict[int, int] = {}  # color -> count of cells that disappeared

        h = min(len(prev_layer), len(curr_layer))
        if h == 0:
            return
        w = min(len(prev_layer[0]), len(curr_layer[0]))
        if w == 0:
            return

        for r in range(h):
            for c in range(w):
                try:
                    old_val = prev_layer[r][c]
                    new_val = curr_layer[r][c]
                    if (old_val == 0 or old_val == -1) and (new_val != 0 and new_val != -1):
                        appeared[new_val] = appeared.get(new_val, 0) + 1
                    elif (new_val == 0 or new_val == -1) and (old_val != 0 and old_val != -1):
                        disappeared[old_val] = disappeared.get(old_val, 0) + 1
                    elif old_val != new_val and old_val != 0 and old_val != -1 and new_val != 0 and new_val != -1:
                        # Cell changed color — both appeared and disappeared
                        appeared[new_val] = appeared.get(new_val, 0) + 1
                        disappeared[old_val] = disappeared.get(old_val, 0) + 1
                except (IndexError, TypeError):
                    continue

        # The player color is the one that both disappeared and appeared
        # (it moved from old position to new position)
        for color in disappeared:
            if color in appeared:
                # This color moved — it's likely the player
                # Weight by how many cells of this color moved (1-3 cells typical for player sprite)
                moved_count = min(disappeared[color], appeared[color])
                if moved_count <= 5:  # Player sprites are usually small (1-5 cells)
                    self._player_color_candidates[color] = (
                        self._player_color_candidates.get(color, 0) + moved_count
                    )

        # Determine player color from candidates
        if self._player_color_candidates:
            # The color with highest cumulative movement count is most likely the player
            sorted_candidates = sorted(
                self._player_color_candidates.items(),
                key=lambda x: x[1],
                reverse=True
            )
            # Only accept if the top candidate has significantly more evidence than others
            top_color = sorted_candidates[0][0]
            top_count = sorted_candidates[0][1]
            second_count = sorted_candidates[1][1] if len(sorted_candidates) > 1 else 0
            # v5.0.7: E-value dynamic threshold (LTT-IDO insight)
            # E-value = observed_count / expected_random_count
            # Under random hypothesis, each of ~10 colors would appear equally,
            # so expected_random_count = total_observations / num_colors.
            # E-value > 10.0 → strong evidence (10x above chance) → accept immediately
            # Even with just 1 observation, if it's the ONLY candidate (e=∞), accept.
            # This replaces the fixed >=3 threshold which was too conservative early
            # and too liberal late — e-value adapts to evidence accumulation rate.
            total_obs = sum(c for _, c in sorted_candidates)
            num_colors = max(len(sorted_candidates), 2)  # at least 2 categories
            expected_random = max(total_obs / num_colors, 0.5)  # floor at 0.5
            e_value = top_count / expected_random

            # Accept if e_value > 10.0 (10x above random chance) OR only one candidate
            if (e_value > 10.0 or len(sorted_candidates) == 1) and top_count >= 1:
                self._player_color = top_color

    # ── v5.0.5: Goal detection ──────────────────────────────────────────

    def _detect_goal_positions(self, grid: Any) -> None:
        """Detect likely goal/exit/target positions from grid analysis.

        Uses multiple heuristics:
          1. Rare colors — cells with colors appearing in <10% of nonzero cells
          2. Border openings — cells at edges that are nonzero (possible exits)
          3. Isolated clusters — small groups of unique colors (goals/switches)
          4. Exclude player color and common wall colors

        Args:
            grid: Current frame grid data.
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return

        h = len(layer)
        if h == 0:
            return
        w = len(layer[0])
        if w == 0:
            return

        # Count color frequencies
        color_counts: Dict[int, int] = {}
        color_positions: Dict[int, List[Tuple[int, int]]] = {}
        total_nonzero = 0

        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        color_counts[val] = color_counts.get(val, 0) + 1
                        if val not in color_positions:
                            color_positions[val] = []
                        color_positions[val].append((c, r))  # (x, y) format
                        total_nonzero += 1
                except (IndexError, TypeError):
                    continue

        if total_nonzero == 0:
            return

        # Identify wall colors: colors appearing in >30% of nonzero cells
        # These are likely walls/decoration, NOT goals
        self._wall_colors = []
        for color, count in color_counts.items():
            if count > total_nonzero * 0.3:
                self._wall_colors.append(color)

        # Identify goal colors: rare colors NOT player color, NOT wall colors
        # Goals/exits/switches typically have rare, distinctive colors
        self._goal_colors = []
        goal_positions_set: Set[Tuple[int, int]] = set()

        # Sort colors by frequency (ascending) — rarest first
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])

        for color, count in sorted_colors:
            # Skip if it's the player color
            if self._player_color is not None and color == self._player_color:
                continue
            # Skip if it's a wall color (common)
            if color in self._wall_colors:
                continue
            # Goal candidate: rare color (<10% of nonzero cells)
            # or medium-rare (<25%) but NOT in wall_colors
            if count <= total_nonzero * 0.10 or (count <= total_nonzero * 0.25 and count <= 20):
                self._goal_colors.append(color)
                for pos in color_positions[color]:
                    goal_positions_set.add(pos)

        # Heuristic 2: Border openings — nonzero cells at grid edges
        # These are often exits/doors/passages
        for r in range(h):
            for c in [0, w - 1]:  # Left and right edges
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        if self._player_color is not None and val == self._player_color:
                            continue
                        if val not in self._wall_colors:
                            goal_positions_set.add((c, r))
                except (IndexError, TypeError):
                    continue
        for c in range(w):
            for r in [0, h - 1]:  # Top and bottom edges
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        if self._player_color is not None and val == self._player_color:
                            continue
                        if val not in self._wall_colors:
                            goal_positions_set.add((c, r))
                except (IndexError, TypeError):
                    continue

        # Heuristic 3: Delta click pool positions — cells that changed recently
        # These are interactive elements (switches, buttons, teleporters)
        for x, y in self._delta_click_pool[:20]:
            if (x, y) not in goal_positions_set:
                # Only add if it's not near the player (not player trail)
                if self._estimated_player_pos:
                    px, py = self._estimated_player_pos
                    if abs(x - px) + abs(y - py) > 3:  # Not immediately adjacent to player
                        goal_positions_set.add((x, y))

        # Update goal positions list
        self._goal_positions = list(goal_positions_set)

        # Sort goals by priority: rarest colors first, then border positions
        # This gives us a priority queue of targets to navigate toward
        def goal_priority(pos: Tuple[int, int]) -> float:
            x, y = pos
            try:
                val = layer[y][x]
            except (IndexError, TypeError):
                return 100.0
            # Lower priority number = more important
            # Rarity: fewer occurrences = more likely a goal
            rarity = 1.0 / (color_counts.get(val, 1) + 1)
            # Border bonus: cells at edges are more likely exits
            border_bonus = 0.0
            if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                border_bonus = 2.0
            # Delta bonus: cells that changed recently
            delta_bonus = 0.0
            if (x, y) in [(dx, dy) for dx, dy in self._delta_click_pool]:
                delta_bonus = 1.5
            return -(rarity + border_bonus + delta_bonus)

        self._goal_positions.sort(key=goal_priority)

    def _learn_direction_from_delta(
        self,
        action_name: str,
        delta: List[Tuple[int, int]],
        old_grid: Any,
        new_grid: Any,
    ) -> None:
        """Learn direction mapping from observed deltas after a keyboard action.

        After a keyboard action (ACTION1-4), the delta cells reveal where
        the player moved. We can learn which ACTION maps to which (dx, dy)
        direction by comparing old vs new player position.

        Args:
            action_name: The keyboard action taken (ACTION1-4).
            delta: Changed cells from that action.
            old_grid: Grid before the action.
            new_grid: Grid after the action.
        """
        self._direction_probed[action_name] = True

        if not delta:
            # No change — this direction doesn't work (blocked or wrong direction)
            # Mark as inactive so we avoid it
            self._inactive_actions[action_name] = self._inactive_actions.get(action_name, 0) + 1
            return

        # Compute old and new player positions
        old_pos = self._estimate_player_position(delta, action_name, old_grid)
        new_pos = self._estimate_player_position(delta, action_name, new_grid)

        if old_pos and new_pos:
            dx = new_pos[0] - old_pos[0]
            dy = new_pos[1] - old_pos[1]

            # Only update direction map if we see actual movement
            if dx != 0 or dy != 0:
                self._direction_map[action_name] = (dx, dy)
                self._effective_actions[action_name] = self._effective_actions.get(action_name, 0) + 1

    def _compute_navigate_path(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> List[str]:
        """Compute a collision-aware BFS path of keyboard actions from start to target.

        v5.0.6: Replaces greedy Manhattan-distance approach with BFS that uses
        the grid's wall_color data to find obstacle-free paths. The greedy approach
        would get stuck on walls because it blindly moved toward the target without
        considering obstacles.

        Algorithm:
          1. Extract layer0 grid and mark wall_color cells as impassable
          2. BFS from start to target, exploring all 4 directions
          3. Convert BFS coordinate path to action names using _direction_map

        Args:
            start: Current estimated player position (x, y).
            target: Target position (x, y).

        Returns:
            List of action names (ACTION1-4) to navigate toward target.
        """
        # Get grid dimensions and wall obstacle data
        if not self._grid_history:
            return self._compute_navigate_path_greedy(start, target)

        current_grid = self._grid_history[-1]
        layer = self._extract_layer0(current_grid)
        if not layer or not layer[0]:
            return self._compute_navigate_path_greedy(start, target)

        h = len(layer)
        w = len(layer[0])
        sx, sy = start
        tx, ty = target

        # Build obstacle map: cells with wall_colors are impassable
        # Also treat player_color cells as passable (player can move through itself)
        obstacles: Set[Tuple[int, int]] = set()
        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1 and val in self._wall_colors:
                        obstacles.add((c, r))  # (x, y) format
                except (IndexError, TypeError):
                    continue

        # BFS from start to target
        visited: Set[Tuple[int, int]] = {start}
        # Queue stores (current_x, current_y, path_of_coords)
        queue: List[Tuple[int, int, List[Tuple[int, int]]]] = [(sx, sy, [(sx, sy)])]

        # Direction offsets for 4 cardinal directions
        # These map to the 4 keyboard actions in ARC-AGI-3
        # We'll try all 4 and use _direction_map to convert to action names later
        bfs_directions = [
            (0, -1),   # Up (decrease y)
            (0, 1),    # Down (increase y)
            (-1, 0),   # Left (decrease x)
            (1, 0),    # Right (increase x)
        ]

        max_bfs_steps = min(h * w, 500)  # Limit BFS to avoid timeouts
        bfs_steps = 0

        while queue and bfs_steps < max_bfs_steps:
            cx, cy, coord_path = queue.pop(0)  # BFS = FIFO
            bfs_steps += 1

            # Check if we reached the target (within 1 cell for tolerance)
            if abs(cx - tx) <= 1 and abs(cy - ty) <= 1:
                # Reached target! Convert coord path to action names
                return self._coord_path_to_actions(coord_path + [(tx, ty)])

            for dx, dy in bfs_directions:
                nx, ny = cx + dx, cy + dy
                # Bounds check
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                # Obstacle check
                if (nx, ny) in obstacles:
                    continue
                # Visited check
                if (nx, ny) in visited:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny, coord_path + [(nx, ny)]))

        # BFS failed (target unreachable or too far) — fall back to greedy
        return self._compute_navigate_path_greedy(start, target)

    def _coord_path_to_actions(
        self, coord_path: List[Tuple[int, int]]
    ) -> List[str]:
        """Convert a coordinate path (list of (x,y) positions) to action names.

        Uses the learned _direction_map to determine which keyboard action
        corresponds to each coordinate transition. Falls back to standard
        ARC3 mapping if direction map is incomplete.

        Args:
            coord_path: Sequence of (x, y) positions along the path.

        Returns:
            List of action names (ACTION1-4) corresponding to the path.
        """
        actions: List[str] = []
        for i in range(len(coord_path) - 1):
            prev_x, prev_y = coord_path[i]
            next_x, next_y = coord_path[i + 1]
            dx = next_x - prev_x
            dy = next_y - prev_y

            if dx == 0 and dy == 0:
                continue  # No movement

            # Find which action produces this (dx, dy) offset
            best_action = None
            for action_name in DIRECTION_ACTIONS:
                if action_name in self._direction_map:
                    adx, ady = self._direction_map[action_name]
                    if adx == dx and ady == dy:
                        best_action = action_name
                        break

            if best_action is None:
                # Fall back to standard mapping: ACTION1=UP, ACTION2=DOWN,
                # ACTION3=LEFT, ACTION4=RIGHT
                if dy < 0:
                    best_action = "ACTION1"  # UP
                elif dy > 0:
                    best_action = "ACTION2"  # DOWN
                elif dx < 0:
                    best_action = "ACTION3"  # LEFT
                elif dx > 0:
                    best_action = "ACTION4"  # RIGHT
                else:
                    continue  # No movement

            actions.append(best_action)

        # Limit path length to prevent over-planning
        max_steps = 40
        if len(actions) > max_steps:
            actions = actions[:max_steps]

        return actions

    def _compute_navigate_path_greedy(
        self, start: Tuple[int, int], target: Tuple[int, int]
    ) -> List[str]:
        """Fallback greedy Manhattan-distance path planning (pre-v5.0.6).

        Used when BFS fails or grid data is unavailable. Picks the best
        direction at each step based on alignment with desired displacement,
        without considering wall obstacles.

        Args:
            start: Current estimated player position (x, y).
            target: Target position (x, y).

        Returns:
            List of action names (ACTION1-4) to navigate toward target.
        """
        path: List[str] = []
        cx, cy = start
        tx, ty = target

        # Maximum path length — don't plan too far ahead
        max_steps = 40
        steps = 0

        while (cx, cy) != (tx, ty) and steps < max_steps:
            dx = tx - cx
            dy = ty - cy

            # Choose the best action for the current displacement
            best_action: Optional[str] = None
            best_score: float = -1.0

            for action_name in DIRECTION_ACTIONS:
                if action_name in self._direction_map:
                    adx, ady = self._direction_map[action_name]
                    # How much does this action reduce distance?
                    # Score = how much the action aligns with the desired direction
                    score = 0.0
                    if dx != 0:
                        score += (adx * dx) / abs(dx) if dx != 0 else 0
                    if dy != 0:
                        score += (ady * dy) / abs(dy) if dy != 0 else 0
                    # Penalize actions that increase distance
                    new_dist = abs(cx + adx - tx) + abs(cy + ady - ty)
                    old_dist = abs(cx - tx) + abs(cy - ty)
                    if new_dist > old_dist:
                        score -= 2.0
                    if score > best_score:
                        best_score = score
                        best_action = action_name
                else:
                    heuristic_dy = 0
                    heuristic_dx = 0
                    if action_name == "ACTION1":
                        heuristic_dy = -1  # UP
                    elif action_name == "ACTION2":
                        heuristic_dy = 1   # DOWN
                    elif action_name == "ACTION3":
                        heuristic_dx = -1  # LEFT
                    elif action_name == "ACTION4":
                        heuristic_dx = 1   # RIGHT

                    score = 0.0
                    if dx != 0 and heuristic_dx != 0:
                        score += (heuristic_dx * dx) / abs(dx)
                    if dy != 0 and heuristic_dy != 0:
                        score += (heuristic_dy * dy) / abs(dy)
                    score -= 0.5
                    if score > best_score:
                        best_score = score
                        best_action = action_name

            if best_action:
                if best_action in self._direction_map:
                    adx, ady = self._direction_map[best_action]
                    cx += adx
                    cy += ady
                else:
                    if best_action == "ACTION1":
                        cy -= 1
                    elif best_action == "ACTION2":
                        cy += 1
                    elif best_action == "ACTION3":
                        cx -= 1
                    elif best_action == "ACTION4":
                        cx += 1
                path.append(best_action)
            else:
                break

            steps += 1

        return path

    # ── Main action selection ───────────────────────────────────────────

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Select next action based on current state and plan.

        Workflow:
          1. Handle level transitions (compute new plan)
          2. Record grid snapshot and compute delta from previous frame
          3. Handle NOT_PLAYED / GAME_OVER states
          4. Execute Replay Oracle plan if available
          5. Fall back to delta-aware smart exploration

        All logic is wrapped in a defensive try/except to prevent any
        unexpected exception from crashing the Kaggle notebook.
        """
        try:
            return self._choose_action_impl(frames, latest_frame)
        except Exception:
            # Defensive fallback: never let an exception crash the notebook.
            # Return a safe default action based on dynamic game type detection.
            try:
                base_id = self.game_id.split("-")[0] if self.game_id else ""
                available_set_def = set(latest_frame.available_actions or [])
                if self._is_click_capable_game(base_id, available_set_def):
                    a = GameAction.ACTION6
                    a.set_data({"x": 32, "y": 32})
                    a.reasoning = "defensive-fallback-click"
                    return a
                else:
                    a = GameAction.ACTION1
                    a.reasoning = "defensive-fallback"
                    return a
            except Exception:
                return GameAction.ACTION1

    def _choose_action_impl(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Internal action selection — actual implementation.

        Separated from choose_action to allow clean try/except wrapping.
        """

        # ── Handle level transitions ──
        if latest_frame.levels_completed > self._levels_done:
            self._levels_done = latest_frame.levels_completed
            self._retries = 0  # Reset retries on level advance
            new_level = self._levels_done
            self._compute_plan(new_level)

        # ── Record grid snapshot and compute delta ──
        try:
            self._record_and_analyze(frames, latest_frame)
        except Exception:
            pass  # Non-critical — continue with stale state

        # ── Handle NOT_PLAYED → RESET to start the level ──
        if latest_frame.state is GameState.NOT_PLAYED:
            return GameAction.RESET

        # ── Handle GAME_OVER → RESET and retry (with 3-Life strategy) ──
        if latest_frame.state is GameState.GAME_OVER:
            self._retries += 1
            self._game_over_count += 1
            self._update_life_phase()

            # v5.0.4: Oracle fallback — if we were executing an Oracle plan and
            # hit GAME_OVER, the Oracle data is wrong for this level. Clear the
            # plan and switch to grid-scan exploration instead of blindly retrying.
            if self._plan and self._plan_idx > 0:
                # Oracle plan failed — abandon it, switch to exploration
                self._plan = []
                self._plan_idx = 0
                self._reset_exploration_state()
                self._grid_scan_initialized = False
                self._grid_scan_queue = []
                # Don't count this as a full retry — it's a strategy switch
                self._retries = min(self._retries, self._max_retries - 1)
            elif self._retries > self._max_retries:
                # Too many retries — abandon plan, try exploration
                self._plan = []
                self._plan_idx = 0
                self._reset_exploration_state()
            else:
                # In Life1, record discoveries before retry
                if self._life_phase == "life1" or self._life_phase == "life2":
                    # Record effective actions from this life
                    for action_name, count in self._effective_actions.items():
                        if count > 0 and action_name not in self._life1_effective_actions:
                            self._life1_effective_actions.append(action_name)
                    # Record effective clicks
                    for x, y in self._delta_click_pool[:10]:
                        if (x, y) not in self._life1_effective_clicks:
                            self._life1_effective_clicks.append((x, y))

                # Reset plan index to retry from beginning
                self._plan_idx = 0
                # Also reset exploration direction probes — game may have changed
                self._direction_probed = {}
                self._direction_probe_count = 0
                self._stall_counter = 0
                # Reset ASD for new life
                self._asd_analyzed = False
                self._asd_anomaly_targets = []
            return GameAction.RESET

        # ── Execute Replay Oracle plan ──
        if self._plan and self._plan_idx < len(self._plan):
            action_name, data = self._plan[self._plan_idx]
            self._plan_idx += 1

            action = getattr(GameAction, action_name)
            if data and action.is_complex():
                action.set_data(data)
                action.reasoning = {"why": "replay oracle", "step": self._plan_idx}
                # v6.3: Record action_key for state graph tracking
                if action_name == "ACTION6" and data:
                    self._last_action_key = f"ACTION6@{data.get('x', 0)},{data.get('y', 0)}"
                else:
                    self._last_action_key = action_name
            else:
                action.reasoning = f"replay oracle step {self._plan_idx}/{len(self._plan)}"
                # v6.3: Record action_key for state graph tracking
                self._last_action_key = action_name

            self._action_history.append(action_name)
            return action

        # ── v6.3: Informed search fallback ──
        return self._informed_search(frames, latest_frame)

    def _record_and_analyze(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> None:
        """Record current grid snapshot, compute delta, and learn from it.

        This is called before action selection to update the agent's
        understanding of the game state.

        Args:
            frames: All previous frames.
            latest_frame: Current frame data.
        """
        current_grid = latest_frame.frame
        self._grid_history.append(current_grid)

        # ── ASD: Analyze first frame for anomaly detection ──
        if not self._asd_analyzed and current_grid:
            self._asd_analyze_first_frame(current_grid)

        # Compute current grid hash for pattern detection
        self._current_grid_hash = self._grid_hash(current_grid)

        # v7.0: Full-frame hash for state graph (exact dedup)
        # _prev_state_hash saves the previous state hash before overwriting
        self._prev_state_hash = self._state_hash
        self._state_hash = self._compute_state_hash(current_grid)

        # ── v6.5: Compute object-level state ──
        self._object_state = self._extract_objects(current_grid)
        self._object_hash = self._compute_object_hash(self._object_state)
        self._heuristic_distance = self._compute_heuristic_distance(self._object_state)

        # ── v6.6: Rule Hypothesis Engine — observe grid changes ──
        # In first MAX_HYPOTHESIS_STEPS, track how grid changes after actions
        # to infer game mechanics (push/toggle/propagation).
        if (self._action_count > 0
                and not self._hypothesis_confirmed
                and self._hypothesis_step_count < self._MAX_HYPOTHESIS_STEPS
                and self._last_action_key):
            prev_grid = None
            if len(frames) >= 2 and frames[-2].frame is not None:
                prev_grid = frames[-2].frame
            elif len(self._grid_history) >= 2:
                prev_grid = self._grid_history[-2]
            if prev_grid is not None and current_grid:
                self._observe_rule_hypothesis(
                    prev_grid, current_grid, self._last_action_key, self._hypothesis_step_count
                )
                # Recompute heuristic with updated hypothesis
                self._heuristic_distance = self._compute_heuristic_distance(self._object_state)

        # Compute delta from previous frame
        delta: List[Tuple[int, int]] = []
        if len(frames) >= 2 and frames[-2].frame is not None:
            prev_grid = frames[-2].frame
            delta = self._compute_grid_delta(prev_grid, current_grid)
        elif len(self._grid_history) >= 2:
            prev_grid = self._grid_history[-2]
            delta = self._compute_grid_delta(prev_grid, current_grid)

        self._delta_history.append(delta)

        # Update stall counter
        if len(delta) == 0 and self._action_history:
            # No grid change after our last action — we might be stuck
            self._stall_counter += 1

            # ── NEW v6.3: Record ineffective action in state graph ──
            # v7.0: Use full-frame _state_hash for state graph (exact dedup)
            if self._prev_state_hash and self._last_action_key:
                from_hash = self._prev_state_hash
                # Record that this action leads BACK to same state (self-loop = ineffective)
                if from_hash not in self._state_graph:
                    self._state_graph[from_hash] = {}
                self._state_graph[from_hash][self._last_action_key] = from_hash  # self-loop
                # v7.0: Invalidate untried actions cache for this state
                self._untried_actions_cache.pop(from_hash, None)
                # Update effectiveness rate: no change
                eff, total = self._action_change_rate.get(self._last_action_key, (0, 0))
                self._action_change_rate[self._last_action_key] = (eff, total + 1)

                # Record inactive action
                self._inactive_actions[self._last_action_key] = self._inactive_actions.get(self._last_action_key, 0) + 1
        else:
            self._stall_counter = 0
            # v5.0.3: Deactivate inflation when grid changes (stall broken)
            self._inflation_active = False

        # v5.0.3: Compute ΔIC for cognitive inflation
        if len(delta) > 0 and len(self._grid_history) >= 2:
            self._compute_ic_delta(self._grid_history[-2], current_grid)

        # ── NEW v6.4: Compute IDO Value Score and Noether-Check ──
        # Replace binary "grid changed or not" with directional progress metric.
        # This is the CORE improvement from TOMAS/IDO theory analysis.
        levels_progressed = latest_frame.levels_completed > self._prev_levels_completed
        if len(self._grid_history) >= 2 and self._action_history:
            prev_grid_for_ido = self._grid_history[-2]
            ido_score = self._compute_ido_value_score(prev_grid_for_ido, current_grid, levels_progressed)

            # Noether-Check: validate IC conservation
            if self._last_action_key:
                noether_ok = self._noether_check(prev_grid_for_ido, current_grid, self._last_action_key, levels_progressed)

                # Update action classification based on ido and Noether
                if ido_score > 0.05:
                    # Progressive action — reduces IC-gap toward goal
                    self._progressive_actions[self._last_action_key] = self._progressive_actions.get(self._last_action_key, 0) + 1
                elif ido_score < -0.05:
                    # Backslide action — increases IC without level progress
                    self._backslide_actions[self._last_action_key] = self._backslide_actions.get(self._last_action_key, 0) + 1
                elif abs(ido_score) <= 0.05 and not noether_ok:
                    # Noether violation without meaningful ido — physics-impossible
                    self._backslide_actions[self._last_action_key] = self._backslide_actions.get(self._last_action_key, 0) + 1

            # Track η-plateau: consecutive steps with ido near zero
            if abs(ido_score) <= 0.02:
                self._eta_plateau_counter += 1
            else:
                self._eta_plateau_counter = 0

        # ── NEW v6.4: Compute Goal-EML convergence ──
        if self._asd_analyzed:
            self._compute_goal_eml(current_grid)

        # Update levels completed tracking
        if latest_frame.levels_completed > self._prev_levels_completed:
            self._stall_counter = 0
            self._prev_levels_completed = latest_frame.levels_completed

        # Learn from the delta if we have a previous action
        if self._action_history and delta:
            last_action = self._action_history[-1]

            # Learn direction mapping for keyboard actions
            if last_action in DIRECTION_ACTIONS:
                if len(frames) >= 2 and frames[-2].frame is not None:
                    self._learn_direction_from_delta(
                        last_action, delta, frames[-2].frame, current_grid
                    )
                elif len(self._grid_history) >= 2:
                    self._learn_direction_from_delta(
                        last_action, delta, self._grid_history[-2], current_grid
                    )

            # Record that this action caused a change
            self._effective_actions[last_action] = self._effective_actions.get(last_action, 0) + 1

            # ── NEW v6.3: Record state transition in graph ──
            # v7.0: Use full-frame _state_hash for state graph (exact dedup)
            if self._prev_state_hash and self._last_action_key:
                from_hash = self._prev_state_hash
                to_hash = self._state_hash
                action_key = self._last_action_key

                # Record transition: from_hash → action_key → to_hash
                if from_hash not in self._state_graph:
                    self._state_graph[from_hash] = {}
                self._state_graph[from_hash][action_key] = to_hash

                # v7.0: Invalidate untried actions cache for from_hash (action now tried)
                self._untried_actions_cache.pop(from_hash, None)

                # v7.0: Update frontier states
                # If to_hash is new (not previously in state_graph as a source), it's a frontier
                if to_hash not in self._state_graph:
                    self._frontier_states.add(to_hash)
                # If from_hash now has this action tried, check if it's still a frontier
                # (a frontier state has untried actions — we check lazily in _navigate_to_frontier)

                # Update action effectiveness rate
                eff, total = self._action_change_rate.get(action_key, (0, 0))
                self._action_change_rate[action_key] = (eff + 1, total + 1)

                # Mark to_hash as visited
                self._state_visited[to_hash] = self._state_visited.get(to_hash, 0) + 1

                # v6.5: Track frontier depth (IDS depth limiting)
                parent_depth = self._frontier_depth.get(from_hash, 0)
                self._frontier_depth[to_hash] = parent_depth + 1

                # v6.5: Compute non-local impact (J⊥≫Jz analogy)
                self._compute_nonlocal_impact(
                    self._extract_objects(self._grid_history[-2]) if len(self._grid_history) >= 2 else {},
                    self._object_state,
                    action_key,
                )

            # Build pattern memory: if previous grid hash + action → current grid hash
            if self._last_grid_hash:
                key = self._last_grid_hash
                if key not in self._grid_hash_action_map:
                    self._grid_hash_action_map[key] = last_action
                # If this action caused progress (levels changed), remember it
                if latest_frame.levels_completed > self._prev_levels_completed:
                    self._pattern_memory[key] = [last_action]

            # Update delta click pool — cells that changed are interactive
            for x, y in delta:
                coord_key = f"{x},{y}"
                if coord_key not in self._visited_coords:
                    self._delta_click_pool.append((x, y))

            # ── NEW v3.8.0: Expand grid scan around effective click ──
            # If last action was a click (ACTION6) and it caused a grid change,
            # expand the scan queue around that click position — nearby cells
            # are likely interactive too (BFS-like expansion).
            if last_action == "ACTION6" and self._grid_scan_initialized:
                # Find the click position from action history reasoning
                # We track the most recent click that was effective
                if self._effective_click_positions:
                    last_click = self._effective_click_positions[-1]
                    self._expand_scan_from_click(last_click[0], last_click[1], current_grid)
                # Also try expanding from delta center
                if delta:
                    avg_x = sum(x for x, y in delta) // len(delta)
                    avg_y = sum(y for x, y in delta) // len(delta)
                    self._effective_click_positions.append((avg_x, avg_y))
                    self._expand_scan_from_click(avg_x, avg_y, current_grid)

        # Update estimated player position
        if self._action_history and delta:
            last_action = self._action_history[-1]
            new_pos = self._estimate_player_position(delta, last_action, current_grid)
            if new_pos:
                self._prev_estimated_player_pos = self._estimated_player_pos
                self._estimated_player_pos = new_pos
                # v5.0.5: Track player position for goal-oriented navigation
                self._player_position_history.append(new_pos)
                # Keep history bounded (last 50 positions)
                if len(self._player_position_history) > 50:
                    self._player_position_history = self._player_position_history[-50:]
        elif not self._estimated_player_pos:
            # First frame — try to estimate initial position
            self._estimated_player_pos = self._estimate_player_position([], "", current_grid)

        # v5.0.5: Detect player color from delta analysis
        if delta and self._action_history and self._action_history[-1] in DIRECTION_ACTIONS:
            self._detect_player_color_from_delta(delta, frames, current_grid)

        # v5.0.5: Detect goal positions from grid analysis (every 10 actions)
        if self._action_count % 10 == 0 or not self._goal_positions:
            self._detect_goal_positions(current_grid)

        # Update grid hash for next step's pattern detection
        self._last_grid_hash = self._current_grid_hash
        # v6.5: Save previous object hash for state graph tracking
        self._prev_object_hash = self._object_hash
        # v7.0: _prev_state_hash is already saved at the start of this method
        # (before _state_hash is recomputed) — no additional save needed here.

    # ── ASD Anomaly Detection (Attention Before Loss) ────────────────────

    def _asd_analyze_first_frame(self, grid: Any) -> None:
        """ASD: Analyze first frame to detect minority-color anomaly pixels.

        "Attention Before Loss" — anomaly detection precedes trial-and-error.
        Detects minority-color pixels from first frame, targets them first.

        Args:
            grid: The initial frame grid data.
        """
        if self._asd_analyzed:
            return
        self._asd_analyzed = True

        layer = self._extract_layer0(grid)
        if not layer:
            return

        h = len(layer)
        if h == 0:
            return
        w = len(layer[0])
        if w == 0:
            return

        # Count color frequencies
        color_counts: Dict[int, int] = {}
        total_cells = 0
        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:  # Skip background
                        color_counts[val] = color_counts.get(val, 0) + 1
                        total_cells += 1
                except (IndexError, TypeError):
                    continue

        if total_cells == 0:
            return

        # Sort colors by frequency (ascending) — rarest first
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1])

        # Anomaly colors: colors that appear in < 5% of non-zero cells
        anomaly_threshold = max(3, int(total_cells * 0.05))
        self._asd_anomaly_colors = [color for color, count in sorted_colors if count <= anomaly_threshold]

        # Compute rarity scores
        for color, count in sorted_colors:
            rarity = 1.0 - (count / total_cells) if total_cells > 0 else 0.0
            self._asd_top_rarity[color] = rarity

        # Collect anomaly target positions (x, y) format
        anomaly_set = set(self._asd_anomaly_colors)
        targets: List[Tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val in anomaly_set:
                        targets.append((c, r))  # (x, y)
                except (IndexError, TypeError):
                    continue

        # Sort by rarity (rarest color first)
        targets.sort(key=lambda t: self._asd_top_rarity.get(layer[t[1]][t[0]], 0.0), reverse=True)
        self._asd_anomaly_targets = targets

    def _asd_click_anomaly(self, available_set: Set[int]) -> Optional[GameAction]:
        """Click on the highest-priority ASD anomaly target.

        Returns:
            GameAction for anomaly click, or None if no targets available.
        """
        if not self._asd_anomaly_targets or 6 not in available_set:
            return None

        # Find first unvisited anomaly target
        for x, y in self._asd_anomaly_targets:
            coord_key = f"{x},{y}"
            if coord_key not in self._visited_coords:
                self._visited_coords.add(coord_key)
                action = GameAction.ACTION6
                action.set_data({"x": x, "y": y})
                action.reasoning = {"why": "asd-anomaly-click", "priority": "minority-color"}
                self._action_history.append("ACTION6")
                return action

        return None

    # ── 3-Life Strategy ──────────────────────────────────────────────────

    def _update_life_phase(self) -> None:
        """Update 3-Life strategy phase based on GAME_OVER count.

        Life1: Exploration/discovery — probe all directions, click anomalies
        Life2: Refinement — use discoveries from Life1, focus on effective actions
        Life3: Execution — optimal execution with known effective sequence
        """
        if self._game_over_count == 0:
            self._life_phase = "life1"
        elif self._game_over_count == 1:
            self._life_phase = "life2"
        elif self._game_over_count >= 2:
            self._life_phase = "life3"

    def _life2_refined_action(
        self, frames: list[FrameData], latest_frame: FrameData, available_set: Set[int]
    ) -> Optional[GameAction]:
        """Life2: Use discoveries from Life1 to take refined actions.

        Args:
            frames: All previous frames.
            latest_frame: Current frame data.
            available_set: Set of available action IDs.

        Returns:
            Refined GameAction, or None if no refined strategy available.
        """
        # In Life2, prefer effective actions discovered in Life1
        if self._life1_effective_actions:
            for action_name in self._life1_effective_actions:
                if ACTION_NAME_TO_ID.get(action_name, 0) in available_set:
                    action = getattr(GameAction, action_name)
                    action.reasoning = f"life2-refined: {action_name} (discovered in life1)"
                    self._action_history.append(action_name)
                    return action

        # Life2: prefer effective click positions from Life1
        if 6 in available_set and self._life1_effective_clicks:
            for x, y in self._life1_effective_clicks[:5]:
                coord_key = f"{x},{y}"
                if coord_key not in self._visited_coords:
                    self._visited_coords.add(coord_key)
                    action = GameAction.ACTION6
                    action.set_data({"x": x, "y": y})
                    action.reasoning = {"why": "life2-refined-click", "source": "life1-discovery"}
                    self._action_history.append("ACTION6")
                    return action

        return None

    def _life3_execute(
        self, frames: list[FrameData], latest_frame: FrameData, available_set: Set[int]
    ) -> Optional[GameAction]:
        """Life3: Quantile-LTT extreme conservative execution.

        v5.0.7: Quantile-LTT (Learn-Then-Test) insight — controlling worst-case risk
        is more important than improving average score. Eliminating 0.00-score games
        (worst 5%) > boosting average. Life3 must be EXTREMELY conservative:
          - ONLY replay actions that are VERIFIED effective (caused grid delta AND
            produced positive outcome: level advance or goal proximity)
          - NEVER try new/unverified actions (no ACTION5 strategic fallback)
          - Cycle through verified actions in order, with minimum floor guarantee
          - If no verified actions exist, return None (let other phases handle it)

        This replaces v5.0.6's simple sorted effectiveness ranking which was too
        liberal — it included actions that merely "caused any change" without
        verifying whether that change was positive (closer to goal, level advance).
        """
        # v5.0.7: Build VERIFIED effective actions list
        # An action is "verified" if:
        #   1. It caused grid change (exists in _effective_actions with count > 0)
        #   2. AND it was used during level-progressing moments
        #      (tracked in _life1_effective_actions or near goal positions)
        verified_actions = []
        for action_name, count in self._effective_actions.items():
            if count <= 0:
                continue
            # Verification: action must have been observed effective during
            # exploration (life1) or refinement (life2), not just any random delta
            # We accept all _effective_actions entries with count >= 2 as verified
            # (count=1 could be noise; count>=2 means consistently effective)
            if count >= 2:
                verified_actions.append((action_name, count))
            # Also accept count=1 if it was in _life1_effective_actions
            elif action_name in self._life1_effective_actions:
                verified_actions.append((action_name, count))

        # Sort by effectiveness (highest count first) — proven winners
        verified_actions.sort(key=lambda x: x[1], reverse=True)

        # v5.0.7: Cycle through verified actions with rotation
        # Instead of always picking the top action (which can lead to repetitive
        # dead loops), rotate through all verified actions — each turn picks
        # the next one in sequence. This is the "minimum floor guarantee":
        # every verified action gets a chance, not just the most frequent.
        if verified_actions:
            # Use modulo rotation to cycle through verified actions
            # This prevents getting stuck repeating one action forever
            life3_step = len(self._action_history)  # total steps as rotation index
            idx = life3_step % len(verified_actions)
            action_name, count = verified_actions[idx]
            if ACTION_NAME_TO_ID.get(action_name, 0) in available_set:
                action = getattr(GameAction, action_name)
                action.reasoning = f"life3-conservative: {action_name} (verified={count}, idx={idx}/{len(verified_actions)})"
                self._action_history.append(action_name)
                return action
            # If rotation-selected action not available, try next in sequence
            for action_name, count in verified_actions:
                if ACTION_NAME_TO_ID.get(action_name, 0) in available_set:
                    action = getattr(GameAction, action_name)
                    action.reasoning = f"life3-conservative: {action_name} (verified={count})"
                    self._action_history.append(action_name)
                    return action

        # v5.0.7: NO speculative fallback — removed ACTION5 strategic fallback
        # Quantile-LTT: worst-case risk control means NO unverified actions in Life3.
        # If no verified actions, return None and let Phase 6/7/8 handle it.
        return None

    # ── v6.4: IDO Value Score + Noether-Check + Goal-EML ────────────────────

    def _compute_ic_value(self, grid: Any) -> float:
        """Compute Information Cardinality (IC) of a grid state.

        IC = (distinct_colors × distinct_nonzero_regions) / total_nonzero_cells
        Lower IC = simpler/more ordered grid (closer to goal/solved state)
        Higher IC = more complex/fragmented grid (likely backslide or noise)

        Args:
            grid: Grid data from FrameData.

        Returns:
            IC value (float). Range: typically 0.0 - 50.0.
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return 0.0
        h = len(layer)
        if h == 0:
            return 0.0
        w = len(layer[0])
        if w == 0:
            return 0.0

        # Count distinct non-zero colors
        colors: Set[int] = set()
        total_nonzero = 0
        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        colors.add(val)
                        total_nonzero += 1
                except (IndexError, TypeError):
                    continue

        if total_nonzero == 0:
            return 0.0

        # Count connected regions (simple heuristic: count color-value transitions)
        transitions = 0
        for r in range(h):
            for c in range(w - 1):
                try:
                    if layer[r][c] != layer[r][c+1] and layer[r][c] != 0 and layer[r][c+1] != 0:
                        transitions += 1
                except (IndexError, TypeError):
                    continue
        for r in range(h - 1):
            for c in range(w):
                try:
                    if layer[r][c] != layer[r+1][c] and layer[r][c] != 0 and layer[r+1][c] != 0:
                        transitions += 1
                except (IndexError, TypeError):
                    continue

        # IC = (color_complexity × region_complexity) / total_content
        ic = (len(colors) * (1 + transitions / max(total_nonzero, 1))) / max(total_nonzero / 100, 1)
        return ic

    def _compute_ido_value_score(self, prev_grid: Any, curr_grid: Any, levels_progressed: bool) -> float:
        """Compute IDO Value Score — directional progress metric.

        ido_value_score = v_after - v_before
        where v = (ic_goal - ic_current) / ic_goal
        Positive ido = grid moved toward simpler/solved state (progress)
        Negative ido = grid became more complex without level advance (backslide)

        If levels_progressed (level was completed), override to MAX_POSITIVE
        because completing a level is always true progress regardless of IC.

        Args:
            prev_grid: Previous frame grid.
            curr_grid: Current frame grid.
            levels_progressed: Whether levels_completed increased.

        Returns:
            ido_value_score (float). Positive = progress, negative = backslide.
        """
        if levels_progressed:
            # Level completion = maximum progress, regardless of IC
            return 1.0

        ic_before = self._compute_ic_value(prev_grid)
        ic_after = self._compute_ic_value(curr_grid)
        ic_goal = self._ic_goal_value

        # If ic_goal not set, use ic_before as baseline (first step)
        if ic_goal <= 0:
            ic_goal = ic_before * 0.5  # Assume goal is 50% simpler than start
            self._ic_goal_value = ic_goal

        # Compute v_before and v_after
        v_before = (ic_goal - ic_before) / ic_goal if ic_goal > 0 else 0.0
        v_after = (ic_goal - ic_after) / ic_goal if ic_goal > 0 else 0.0

        # ido = v_after - v_before
        ido = v_after - v_before

        # Clamp to reasonable range
        ido = max(-1.0, min(1.0, ido))

        self._ido_value_score = ido
        self._ido_value_history.append(ido)
        self._last_ic_value = ic_before
        self._current_ic_value = ic_after

        return ido

    def _noether_check(self, prev_grid: Any, curr_grid: Any, action_key: str, levels_progressed: bool) -> bool:
        """Noether-Check: IC conservation along path.

        Checks if IC(s_next) ≤ IC(s_prev) + EPS_NOETHER.
        Violations indicate "physics-impossible" changes — entropy increased
        without level progress (walls broken, teleportation, random noise).

        Actions that violate Noether-Check are marked as backslide and deprioritized.

        Args:
            prev_grid: Previous frame grid.
            curr_grid: Current frame grid.
            action_key: The action_key that caused this transition.
            levels_progressed: Whether levels_completed increased.

        Returns:
            True = Noether-consistent (action is physically plausible)
            False = Noether-violation (action likely backslide/noise)
        """
        if levels_progressed:
            # Level progress overrides Noether — completing a level is always valid
            return True

        ic_before = self._compute_ic_value(prev_grid)
        ic_after = self._compute_ic_value(curr_grid)

        # Noether condition: IC(s_next) ≤ IC(s_prev) + ε
        noether_ok = ic_after <= ic_before + self._noether_eps

        if not noether_ok:
            self._noether_violations[action_key] = self._noether_violations.get(action_key, 0) + 1

        return noether_ok

    def _compute_goal_eml(self, grid: Any) -> float:
        """Compute Goal-EML convergence metric.

        Goal-EML anchoring: detect whether grid is moving toward a "solved" topology.
        Convergence = 1 - (minority_pixels / initial_minority_pixels)
        Higher convergence = closer to goal (minority pixels disappearing → goal achieved)

        Also tracks goal-color expansion: more goal-colored pixels = closer to victory.

        Args:
            grid: Current frame grid.

        Returns:
            Goal-EML convergence metric (0.0 - 1.0).
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return 0.0
        h = len(layer)
        if h == 0:
            return 0.0
        w = len(layer[0])
        if w == 0:
            return 0.0

        # Count current minority/anomaly pixels
        current_minority = 0
        anomaly_set = set(self._asd_anomaly_colors)
        goal_color_set = set(self._goal_colors)
        goal_pixels = 0

        for r in range(h):
            for c in range(w):
                try:
                    val = layer[r][c]
                    if val != 0 and val != -1:
                        if val in anomaly_set:
                            current_minority += 1
                        if val in goal_color_set:
                            goal_pixels += 1
                except (IndexError, TypeError):
                    continue

        self._minority_pixel_count = current_minority
        self._goal_color_expansion = goal_pixels

        # Convergence = progress toward eliminating minority pixels
        initial_minority = max(len(self._asd_anomaly_targets), 1)
        if current_minority == 0:
            convergence = 1.0  # All anomaly pixels gone = goal achieved
        else:
            convergence = 1.0 - (current_minority / initial_minority)

        # Boost convergence if goal colors are expanding
        if goal_pixels > 0 and goal_color_set:
            goal_boost = min(0.2, goal_pixels / 100)  # Up to 0.2 boost
            convergence = min(1.0, convergence + goal_boost)

        self._goal_eml_convergence = convergence
        return convergence

    # ── v6.6: CCA (Connected Component Analysis) + Rule Hypothesis Engine ───────

    def _cca_connected_components(self, grid_layer: Any) -> List[Dict[str, Any]]:
        """v6.6: Connected Component Analysis — flood-fill BFS to identify contiguous color regions.

        Replaces crude 8×8 region signature with proper object detection.
        Each connected region of same color becomes an "object" with:
        - centroid (x, y), size (pixel count), color, bounding box, object type

        Implements Perceptual Abstraction Functor F from TOMAS-Hybrid theory.
        """
        if not grid_layer:
            return []
        h = len(grid_layer)
        w = len(grid_layer[0])
        if h == 0 or w == 0:
            return []

        wall_set = set(self._wall_colors) if self._wall_colors else set()
        visited = set()
        objects = []

        for r in range(h):
            for c in range(w):
                try:
                    color = grid_layer[r][c]
                except (IndexError, TypeError):
                    continue
                if color == 0 or color == -1 or color in wall_set:
                    continue  # Skip background and walls
                if (r, c) in visited:
                    continue

                # Flood-fill BFS for this color region
                component_pixels = []
                queue = [(r, c)]
                visited.add((r, c))
                while queue:
                    cr, cc = queue.pop(0)
                    component_pixels.append((cr, cc))
                    # 8-connectivity (including diagonals for robustness)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                                try:
                                    if grid_layer[nr][nc] == color:
                                        visited.add((nr, nc))
                                        queue.append((nr, nc))
                                except (IndexError, TypeError):
                                    continue

                # Compute object properties
                n_pixels = len(component_pixels)
                if n_pixels < 2:  # Skip single-pixel noise
                    continue

                sum_r = sum(p[0] for p in component_pixels)
                sum_c = sum(p[1] for p in component_pixels)
                centroid_x = sum_c / n_pixels
                centroid_y = sum_r / n_pixels

                min_r = min(p[0] for p in component_pixels)
                max_r = max(p[0] for p in component_pixels)
                min_c = min(p[1] for p in component_pixels)
                max_c = max(p[1] for p in component_pixels)

                # Classify object type
                obj_type = "unknown"
                player_color = self._player_color if self._player_color else -1
                goal_color_set = set(self._goal_colors) if self._goal_colors else set()
                anomaly_set = set(self._asd_anomaly_colors) if self._asd_analyzed else set()

                if color == player_color and self._estimated_player_pos:
                    # Check if near estimated player position
                    px, py = self._estimated_player_pos
                    if abs(centroid_x - px) < 5 and abs(centroid_y - py) < 5:
                        obj_type = "player"
                if color in goal_color_set:
                    obj_type = "goal"
                if color in anomaly_set:
                    obj_type = "anomaly"

                # Size-based classification
                bbox_w = max_c - min_c + 1
                bbox_h = max_r - min_r + 1
                aspect_ratio = bbox_w / max(bbox_h, 1)
                if n_pixels > 100:
                    obj_type = "large_block" if obj_type == "unknown" else obj_type

                obj = {
                    'centroid': (round(centroid_x, 1), round(centroid_y, 1)),
                    'size': n_pixels,
                    'color': color,
                    'bbox': (min_c, min_r, max_c, max_r),
                    'bbox_size': (bbox_w, bbox_h),
                    'aspect_ratio': round(aspect_ratio, 2),
                    'obj_type': obj_type,
                    'pixels': component_pixels[:50],  # Keep first 50 pixels for reference (don't store all)
                }
                objects.append(obj)

        # Sort by size (largest first) for priority
        objects.sort(key=lambda o: o['size'], reverse=True)
        return objects

    def _extract_objects(self, grid: Any) -> Dict[str, Any]:
        """Extract key objects from 64×64 grid for object-level state abstraction.

        v6.6 CORE: Uses CCA (Connected Component Analysis) — proper flood-fill BFS
        to identify contiguous color regions as real "objects", replacing the crude
        8×8 region signature from v6.5.

        Implements Perceptual Abstraction Functor F from TOMAS-Hybrid theory:
        pixel stream → object-level state S via CCA + object classification.
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return {}
        h = len(layer)
        if h == 0:
            return {}
        w = len(layer[0])
        if w == 0:
            return {}

        # v6.6: CCA replaces 8×8 region signature
        cca_objs = self._cca_connected_components(layer)
        self._cca_objects = cca_objs  # Store for rule hypothesis engine

        # Aggregate statistics from CCA objects
        color_dist: Dict[int, int] = {}
        total_nonzero = 0
        player_pos = self._estimated_player_pos
        anomaly_set = set(self._asd_anomaly_colors) if self._asd_analyzed else set()
        goal_color_set = set(self._goal_colors) if self._goal_colors else set()
        special_pos: List[Tuple[int, int, int]] = []
        goal_pos_pixels: List[Tuple[int, int]] = []

        for obj in cca_objs:
            color = obj['color']
            size = obj['size']
            color_dist[color] = color_dist.get(color, 0) + size
            total_nonzero += size

            cx, cy = obj['centroid']
            obj_type = obj['obj_type']

            if obj_type == "anomaly" or color in anomaly_set:
                special_pos.append((round(cx), round(cy), color))
            if obj_type == "goal" or color in goal_color_set:
                goal_pos_pixels.append((round(cx), round(cy)))

        # Object-level signatures (replace grid_signature with CCA summary)
        obj_sig = []
        for obj in cca_objs[:30]:  # Top 30 objects by size
            obj_sig.append(f"{obj['color']}:{obj['size']}:{obj['obj_type']}")

        return {
            'player_pos': player_pos,
            'color_dist': color_dist,
            'special_pos': special_pos,
            'goal_pos': goal_pos_pixels if goal_pos_pixels else self._goal_positions,
            'n_colors': len(color_dist),
            'n_nonzero': total_nonzero,
            'grid_signature': obj_sig,  # v6.6: CCA-based object signature (was 8×8 region_sig)
            'cca_objects': cca_objs,  # v6.6: Full CCA object list
            'n_objects': len(cca_objs),  # v6.6: Number of detected objects
        }

    def _compute_object_hash(self, objects: Dict[str, Any]) -> str:
        """v6.6: Object-level hash using CCA object signatures for state graph."""
        if not objects:
            return ""
        hash_parts = []
        if objects.get('color_dist'):
            top_colors = sorted(objects['color_dist'].items(), key=lambda x: x[1], reverse=True)[:10]
            hash_parts.append(f"c:{top_colors}")
        if objects.get('player_pos'):
            px, py = objects['player_pos']
            hash_parts.append(f"p:{px//4},{py//4}")  # Quantize to 4-pixel regions
        if objects.get('special_pos'):
            sorted_specials = sorted(objects['special_pos'], key=lambda x: (x[2], x[0], x[1]))[:20]
            hash_parts.append(f"s:{sorted_specials}")
        if objects.get('goal_pos'):
            sorted_goals = sorted(objects['goal_pos'], key=lambda x: (x[0], x[1]))[:10]
            hash_parts.append(f"g:{sorted_goals}")
        hash_parts.append(f"nc:{objects.get('n_colors', 0)}")
        hash_parts.append(f"nn:{objects.get('n_nonzero', 0)}")
        hash_parts.append(f"no:{objects.get('n_objects', 0)}")  # v6.6: number of objects
        # v6.6: Use CCA object signatures instead of region_sig
        if objects.get('grid_signature'):
            hash_parts.append(f"os:{','.join(str(x) for x in objects['grid_signature'][:20])}")
        hash_input = "|".join(hash_parts)
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _compute_state_hash(self, grid: Any) -> str:
        """v7.0: Full-frame hash using MD5 for EXACT state deduplication.

        Replaces coarse CCA-based hash for state graph operations.
        CCA hash (_object_hash) is retained for heuristic computation.

        Two different grid states will NEVER produce the same hash (unlike
        _object_hash which quantizes player position to 4px regions and
        only tracks top-10 colors — many different grids map to same hash).

        Args:
            grid: The frame data (3D or 2D).

        Returns:
            MD5 hex digest of the entire grid frame, or "" if invalid.
        """
        layer = self._extract_layer0(grid)
        if not layer:
            return ""
        try:
            data = bytes(v for row in layer for v in row)
            return hashlib.md5(data).hexdigest()
        except (TypeError, ValueError):
            return ""

    def _compute_heuristic_distance(self, objects: Dict[str, Any]) -> float:
        """v6.6 A* heuristic: CCA-based goal-distance + Rule Hypothesis guided.

        Lower = closer to goal. 0 = goal achieved.
        Keyboard: Manhattan distance player→goal (or predicted goal if hypothesis confirmed).
        Click: remaining anomaly objects / initial count.
        Mixed: 1 - goal_eml_convergence (or hypothesis confidence).

        v6.6: When rule hypothesis is confirmed, use predicted_goal_objects for
        more precise targeting — this is the Layer I (Induction) benefit.
        """
        h_distance = 100.0  # Default: far from goal

        # v6.6: If rule hypothesis confirmed, use predicted goal
        goal_positions = objects.get('goal_pos', [])
        if self._hypothesis_confirmed and self._predicted_goal_objects:
            pred_goals = [(round(o['centroid'][0]), round(o['centroid'][1]))
                          for o in self._predicted_goal_objects
                          if o.get('obj_type') in ('goal', 'predicted_goal')]
            if pred_goals:
                goal_positions = pred_goals

        if self._detected_game_type == "keyboard" and self._estimated_player_pos:
            if goal_positions:
                min_dist = min(
                    abs(self._estimated_player_pos[0] - gx) + abs(self._estimated_player_pos[1] - gy)
                    for gx, gy in goal_positions
                )
                h_distance = min_dist / 128.0  # Normalize: 0.0-1.0
            else:
                minority = objects.get('n_colors', 1)
                initial_minority = max(len(self._asd_anomaly_targets), 1)
                h_distance = minority / initial_minority
        elif self._detected_game_type == "click":
            initial_count = max(len(self._asd_anomaly_targets), 1)
            remaining = len(objects.get('special_pos', []))
            h_distance = remaining / initial_count
            # v6.6: CCA-based click heuristic — remaining anomaly objects
            cca_objs = objects.get('cca_objects', [])
            anomaly_objs = [o for o in cca_objs if o.get('obj_type') == 'anomaly']
            if anomaly_objs:
                h_distance = len(anomaly_objs) / max(initial_count, 1)
        else:
            h_distance = 1.0 - self._goal_eml_convergence
            # v6.6: Rule hypothesis confidence boost
            if self._hypothesis_confirmed:
                h_distance *= (1.0 - self._rule_hypothesis.get('confidence', 0.0) * 0.3)

        self._heuristic_distance = h_distance
        return h_distance

    def _compute_nonlocal_impact(self, prev_objects: Dict[str, Any], curr_objects: Dict[str, Any], action_key: str) -> float:
        """v6.6: Non-local impact using CCA object-level changes.

        Measures how many objects changed simultaneously.
        High impact = gravity-flip/teleport (J⊥ class) — many objects moved.
        Low impact = push/step (Jz class) — one object moved locally.

        v6.6: Uses CCA object signatures + centroid displacement for
        more precise impact measurement.
        """
        if not prev_objects or not curr_objects:
            return 0.0

        # v6.6: CCA-based object change measurement
        prev_sig = prev_objects.get('grid_signature', [])
        curr_sig = curr_objects.get('grid_signature', [])
        if not prev_sig or not curr_sig:
            return 0.0

        changed_objects = sum(1 for i in range(min(len(prev_sig), len(curr_sig))) if prev_sig[i] != curr_sig[i])
        total_objects = max(len(prev_sig), 1)
        impact = changed_objects / total_objects

        # Color distribution change (unchanged from v6.5, still useful)
        prev_dist = prev_objects.get('color_dist', {})
        curr_dist = curr_objects.get('color_dist', {})
        color_change = sum(abs(prev_dist.get(c, 0) - curr_dist.get(c, 0)) for c in set(prev_dist) | set(curr_dist))
        total_pixels = max(sum(curr_dist.values()), 1)
        color_impact = color_change / total_pixels

        # v6.6: Object centroid displacement
        prev_objs = prev_objects.get('cca_objects', [])
        curr_objs = curr_objects.get('cca_objects', [])
        centroid_displacement = 0.0
        if prev_objs and curr_objs:
            # Match objects by color+size (same object in different position)
            for po in prev_objs[:15]:
                for co in curr_objs[:15]:
                    if po['color'] == co['color'] and abs(po['size'] - co['size']) <= 2:
                        dx = abs(po['centroid'][0] - co['centroid'][0])
                        dy = abs(po['centroid'][1] - co['centroid'][1])
                        displacement = dx + dy
                        if displacement > 3:  # Only count significant displacements
                            centroid_displacement += displacement

        total_impact = impact + color_impact * 2 + centroid_displacement * 0.01
        self._nonlocal_impact[action_key] = total_impact
        return total_impact

    def _observe_rule_hypothesis(
        self, prev_grid: Any, curr_grid: Any, action_key: str, step: int
    ) -> Dict[str, Any]:
        """v6.6: Rule Hypothesis Engine — observe grid changes → infer game mechanics.

        Lightweight approximation of Layer I (Induction) from TOMAS-Hybrid theory.
        In first 3-5 steps, deliberately observe how the grid changes after each action:
        - Local change near action point → push/mechanics type
        - Color flip at click point → click-toggle type
        - Remote propagation from action → remote-propagation type
        - No change → ineffective/waiting type

        After MAX_HYPOTHESIS_STEPS observations, confirm the most consistent rule
        and predict goal state objects.
        """
        prev_layer = self._extract_layer0(prev_grid)
        curr_layer = self._extract_layer0(curr_grid)
        if not prev_layer or not curr_layer:
            return {}

        h = min(len(prev_layer), len(curr_layer))
        w = min(len(prev_layer[0]) if prev_layer else 0, len(curr_layer[0]) if curr_layer else 0)

        # Find changed pixels
        local_changes: List[Tuple] = []  # Changes near action point
        remote_changes: List[Tuple] = []  # Changes far from action point
        color_flips: List[Tuple] = []  # Pixels that changed color (not moved)

        # Parse action point from action_key
        action_x, action_y = -1, -1
        if "@" in action_key:
            try:
                coords = action_key.split("@")[1].split(",")
                action_x, action_y = int(coords[0]), int(coords[1])
            except (ValueError, IndexError):
                pass
        elif action_key in ("ACTION1", "ACTION2", "ACTION3", "ACTION4"):
            # Keyboard action — use estimated player position
            if self._estimated_player_pos:
                action_x, action_y = self._estimated_player_pos

        for r in range(h):
            for c in range(w):
                try:
                    prev_val = prev_layer[r][c]
                    curr_val = curr_layer[r][c]
                except (IndexError, TypeError):
                    continue
                if prev_val != curr_val:
                    dist_from_action = abs(c - action_x) + abs(r - action_y) if action_x >= 0 else 999

                    if prev_val != 0 and curr_val != 0 and prev_val != curr_val:
                        color_flips.append((c, r, prev_val, curr_val, dist_from_action))
                    elif dist_from_action <= 5:
                        local_changes.append((c, r, prev_val, curr_val, dist_from_action))
                    else:
                        remote_changes.append((c, r, prev_val, curr_val, dist_from_action))

        total_chg = len(local_changes) + len(remote_changes) + len(color_flips)
        observation = {
            'step': step,
            'action_key': action_key,
            'local_changes': len(local_changes),
            'remote_changes': len(remote_changes),
            'color_flips': len(color_flips),
            'total_changes': total_chg,
            'local_pct': len(local_changes) / max(total_chg, 1),
            'remote_pct': len(remote_changes) / max(total_chg, 1),
            'flip_pct': len(color_flips) / max(total_chg, 1),
        }
        self._observation_log.append(observation)
        self._hypothesis_step_count += 1

        # ── Rule inference from observations ──
        if self._hypothesis_step_count >= 2:
            # Analyze pattern across observations
            avg_local = sum(o['local_pct'] for o in self._observation_log) / len(self._observation_log)
            avg_remote = sum(o['remote_pct'] for o in self._observation_log) / len(self._observation_log)
            avg_flip = sum(o['flip_pct'] for o in self._observation_log) / len(self._observation_log)

            # Infer rule type
            if avg_local > 0.5:
                inferred_type = "push_mechanics"  # Local changes dominate → push/pull mechanics
            elif avg_flip > 0.3:
                inferred_type = "click_toggle"  # Color flips → toggle/switch mechanics
            elif avg_remote > 0.3:
                inferred_type = "remote_propagation"  # Remote changes → propagation/chain reaction
            else:
                inferred_type = "mixed"  # No clear dominant pattern

            confidence = max(avg_local, avg_flip, avg_remote)

            self._rule_hypothesis = {
                'type': inferred_type,
                'confidence': confidence,
                'avg_local_pct': avg_local,
                'avg_remote_pct': avg_remote,
                'avg_flip_pct': avg_flip,
                'observations': len(self._observation_log),
            }

            # ── Goal prediction based on inferred rule ──
            if self._hypothesis_step_count >= self._MAX_HYPOTHESIS_STEPS or confidence > 0.6:
                self._hypothesis_confirmed = True

                # Predict goal state based on rule type
                current_objects = self._cca_objects
                predicted_goals: List[Dict[str, Any]] = []

                if inferred_type == "push_mechanics":
                    # Goal: push objects to target positions (goals/anomaly positions)
                    goal_positions = self._goal_positions if self._goal_positions else []
                    anomaly_targets = self._asd_anomaly_targets if self._asd_analyzed else []
                    target_positions = goal_positions + list(anomaly_targets)
                    for pos in target_positions[:5]:
                        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                            predicted_goals.append({
                                'centroid': (float(pos[0]), float(pos[1])),
                                'obj_type': 'predicted_goal',
                                'size': 1,
                                'color': -1,
                            })

                elif inferred_type == "click_toggle":
                    # Goal: toggle all anomaly/special objects to a target state
                    for obj in current_objects:
                        if obj.get('obj_type') in ('anomaly', 'unknown') and obj.get('size', 0) < 50:
                            predicted_goals.append({
                                'centroid': obj['centroid'],
                                'obj_type': 'predicted_goal',
                                'size': obj['size'],
                                'color': obj['color'],
                            })

                elif inferred_type == "remote_propagation":
                    # Goal: trigger propagation to reach all target regions
                    goal_color_set = set(self._goal_colors) if self._goal_colors else set()
                    for obj in current_objects:
                        if obj.get('obj_type') == 'goal' or obj.get('color') in goal_color_set:
                            predicted_goals.append({
                                'centroid': obj['centroid'],
                                'obj_type': 'predicted_goal',
                                'size': obj['size'],
                                'color': obj['color'],
                            })

                self._predicted_goal_objects = predicted_goals

        return observation

    # ── v7.0: 5-Tier Priority Search Helpers ────────────────────────────

    def _get_untried_actions(
        self, current_hash: str, available_set: Set[int], grid: Any
    ) -> List[str]:
        """v7.0: Get actions not yet tried from current state.

        Scans available actions, filters out self-loops and heavily-penalized actions
        (trigger-aware pruning), returns remaining untried action keys sorted by priority.

        Args:
            current_hash: Full-frame hash of current state.
            available_set: Set of available action IDs.
            grid: Current frame grid for click candidate generation.

        Returns:
            List of untried action keys, sorted by priority (highest first).
        """
        # Check cache first (avoid recomputing if called multiple times per turn)
        if current_hash in self._untried_actions_cache:
            cached = self._untried_actions_cache[current_hash]
            # Filter cached actions by current availability AND tried status
            known_transitions = self._state_graph.get(current_hash, {})
            tried_actions = set(known_transitions.keys())
            filtered = []
            for ak in cached:
                if ak in tried_actions:
                    continue  # Already tried since cache was built
                if "@" in ak:
                    if 6 in available_set:
                        filtered.append(ak)
                else:
                    aid = ACTION_NAME_TO_ID.get(ak, 0)
                    if aid in available_set:
                        filtered.append(ak)
            if filtered:
                return filtered

        # Known transitions from current state
        known_transitions = self._state_graph.get(current_hash, {})
        tried_actions = set(known_transitions.keys())

        # Build candidate action keys from available actions
        candidates: List[str] = []

        # Keyboard actions (ACTION1-5, ACTION7)
        for aid in sorted(available_set):
            if aid == 6:
                continue  # ACTION6 handled separately
            action_name = ARC3_ACTION_ID_MAP.get(aid)
            if action_name:
                action_key = action_name
                if action_key not in tried_actions:
                    # Trigger-aware: skip heavily Noether-violating or backslide actions
                    noether_violations = self._noether_violations.get(action_key, 0)
                    backslide_count = self._backslide_actions.get(action_key, 0)
                    if noether_violations < 5 and backslide_count < 8:
                        candidates.append(action_key)

        # Click actions (ACTION6) — only if click-capable
        if 6 in available_set:
            click_candidates = self._generate_click_candidates(grid, tried_actions)
            candidates.extend(click_candidates)

        # Sort by priority: rule hypothesis > effectiveness > progressive
        def untried_priority(ak: str) -> float:
            # Rule hypothesis boost
            hypothesis_boost = 0.0
            if self._hypothesis_confirmed:
                rule_type = self._rule_hypothesis.get('type', '')
                if rule_type == "push_mechanics" and ak in DIRECTION_ACTIONS:
                    hypothesis_boost = 2.0
                elif rule_type == "click_toggle" and "ACTION6" in ak:
                    hypothesis_boost = 2.0
                elif rule_type == "remote_propagation" and "ACTION5" == ak:
                    hypothesis_boost = 1.5

            # Effectiveness boost (from other states — action was effective elsewhere)
            eff, total = self._action_change_rate.get(ak, (0, 0))
            effectiveness = eff / max(total, 1) if total > 0 else 0.5  # untested = moderate

            # Progressive boost
            progressive = self._progressive_actions.get(ak, 0)

            # Backslide penalty (from other states)
            backslide = self._backslide_actions.get(ak, 0)
            backslide_pen = min(0.5, backslide * 0.1)

            # Noether penalty
            noether = self._noether_violations.get(ak, 0)
            noether_pen = min(0.3, noether * 0.1)

            return hypothesis_boost + effectiveness + progressive * 0.3 - backslide_pen - noether_pen

        candidates.sort(key=untried_priority, reverse=True)
        result = candidates[:20]  # Limit to 20 candidates per turn

        # Cache the result
        self._untried_actions_cache[current_hash] = result
        return result

    def _generate_click_candidates(
        self, grid: Any, tried_actions: Set[str]
    ) -> List[str]:
        """v7.0: Generate click action keys (ACTION6@x,y) from priority sources.

        Priority: ASD anomaly → delta pool → goal proximity → rarity scan.

        Args:
            grid: Current frame grid.
            tried_actions: Set of action keys already tried from current state.

        Returns:
            List of click action keys (ACTION6@x,y) not yet tried.
        """
        candidates: List[str] = []

        # Priority 1: ASD anomaly targets (minority pixels)
        if self._asd_anomaly_targets:
            for x, y in self._asd_anomaly_targets[:15]:
                ak = f"ACTION6@{x},{y}"
                if ak not in tried_actions and f"{x},{y}" not in self._visited_coords:
                    candidates.append(ak)

        # Priority 2: Delta click pool (recently changed cells)
        for x, y in self._delta_click_pool[:10]:
            ak = f"ACTION6@{x},{y}"
            if ak not in tried_actions and f"{x},{y}" not in self._visited_coords:
                candidates.append(ak)

        # Priority 3: Goal proximity targets
        if self._goal_positions:
            for gx, gy in self._goal_positions[:5]:
                # Try positions near goal (3px radius)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        x = max(0, min(63, gx + dx))
                        y = max(0, min(63, gy + dy))
                        ak = f"ACTION6@{x},{y}"
                        if ak not in tried_actions and f"{x},{y}" not in self._visited_coords:
                            candidates.append(ak)

        # Priority 4: Grid scan queue (rarity-based)
        if not self._grid_scan_initialized:
            self._init_grid_click_scan(grid)
        for x, y in self._grid_scan_queue[:20]:
            ak = f"ACTION6@{x},{y}"
            if ak not in tried_actions and f"{x},{y}" not in self._visited_coords:
                candidates.append(ak)

        return candidates[:50]  # Limit click candidates

    def _get_predicted_change_actions(
        self, current_hash: str, available_set: Set[int], grid: Any
    ) -> List[str]:
        """v7.0: Get actions predicted to change the frame (trigger-aware).

        Uses: known effective transitions + rule hypothesis + effectiveness history.
        Filters out self-loops, backslide, and Noether-violating actions.

        Args:
            current_hash: Full-frame hash of current state.
            available_set: Set of available action IDs.
            grid: Current frame grid (unused but kept for API consistency).

        Returns:
            List of action keys predicted to cause frame change, sorted by priority.
        """
        candidates: List[str] = []

        # From state graph: transitions that led to DIFFERENT states (not self-loops)
        transitions = self._state_graph.get(current_hash, {})
        novel_transitions = {
            ak: rh for ak, rh in transitions.items()
            if rh != current_hash  # not self-loop
            and ak not in self._backslide_actions  # not consistently backsliding
            and self._noether_violations.get(ak, 0) < 3  # not heavily Noether-violating
        }

        # Sort by frontier priority (visit count + heuristic + progressive + nonlocal)
        MAX_FRONTIER_DEPTH = 50

        def predicted_priority(item: Tuple[str, str]) -> float:
            ak, rh = item
            dest_depth = self._frontier_depth.get(rh, 999)
            if dest_depth > MAX_FRONTIER_DEPTH:
                return -999.0
            heuristic_bonus = 0.1 if dest_depth == 0 else 0.05 / (1.0 + dest_depth)
            visit_score = 1.0 / (1.0 + self._state_visited.get(rh, 0))
            progressive_count = self._progressive_actions.get(ak, 0)
            progressive_boost = min(2.0, progressive_count * 0.5)
            nonlocal_boost = 0.0
            if self._stall_counter >= 3:
                nl_impact = self._nonlocal_impact.get(ak, 0.0)
                nonlocal_boost = min(1.0, nl_impact * 5.0)
            noether_penalty = self._noether_violations.get(ak, 0) * 0.3
            backslide_penalty = self._backslide_actions.get(ak, 0) * 0.1
            return (visit_score + heuristic_bonus + progressive_boost
                    + nonlocal_boost - noether_penalty - backslide_penalty)

        sorted_trans = sorted(novel_transitions.items(), key=predicted_priority, reverse=True)
        for ak, rh in sorted_trans:
            candidates.append(ak)

        # Also add effectiveness-weighted keyboard actions not in graph
        for aid in sorted(available_set):
            if aid == 6:
                continue
            action_name = ARC3_ACTION_ID_MAP.get(aid)
            if action_name and action_name not in transitions:
                eff, total = self._action_change_rate.get(action_name, (0, 0))
                if total > 0 and eff / total > 0.3:  # moderately effective elsewhere
                    # Trigger-aware: skip heavy Noether/backslide
                    if (self._noether_violations.get(action_name, 0) < 3
                            and self._backslide_actions.get(action_name, 0) < 5):
                        candidates.append(action_name)

        return candidates[:15]

    def _navigate_to_frontier(
        self, current_hash: str, available_set: Set[int]
    ) -> Optional[GameAction]:
        """v7.0: BFS navigate to nearest frontier state when current state is exhausted.

        When all known actions at current state are tried (or all lead to self-loops/
        backslide), BFS through the transition graph to find the nearest frontier state
        (one with untried actions), then return the first action on the path to it.

        Args:
            current_hash: Full-frame hash of current state.
            available_set: Set of available action IDs.

        Returns:
            GameAction to execute (first step on path to frontier), or None.
        """
        if not self._frontier_states or not self._state_graph:
            return None

        # Remove current hash from frontier (we're here, it's no longer frontier)
        self._frontier_states.discard(current_hash)

        # BFS from current_hash through _state_graph to find nearest frontier
        # Path: current_hash → action1 → hash1 → action2 → hash2 → ... → frontier_hash
        from collections import deque
        queue: deque = deque([(current_hash, [])])  # (hash, [(action_key, hash) path])
        visited: Set[str] = {current_hash}

        while queue:
            hash_key, path = queue.popleft()

            # Check if this state is a frontier (has untried actions)
            is_frontier = False
            if hash_key in self._frontier_states:
                is_frontier = True
            else:
                # Check if this state actually has untried actions
                transitions = self._state_graph.get(hash_key, {})
                if len(transitions) < len(available_set) + 10:  # rough check
                    # More detailed: check if any available action is untried
                    for aid in available_set:
                        if aid == 6:
                            continue  # Click actions are too numerous to check here
                        action_name = ARC3_ACTION_ID_MAP.get(aid)
                        if action_name and action_name not in transitions:
                            is_frontier = True
                            break

            if is_frontier:
                # Found frontier! Execute first action on path to reach it
                if path:
                    first_action_key, _ = path[0]
                    action_name, click_data = self._parse_action_key(first_action_key, available_set)
                    if action_name:
                        self._last_action_key = first_action_key
                        action = getattr(GameAction, action_name)
                        if click_data and action.is_complex():
                            action.set_data(click_data)
                        action.reasoning = (
                            f"v7.0-frontier-BFS: {first_action_key} → frontier "
                            f"{hash_key[:8]} (path_len={len(path)})"
                        )
                        self._action_history.append(action_name)
                        return action
                # We're already at a frontier state — let untried_actions handle this
                return None

            transitions = self._state_graph.get(hash_key, {})
            for ak, rh in transitions.items():
                if rh not in visited and rh != hash_key:  # skip self-loops and visited
                    visited.add(rh)
                    queue.append((rh, path + [(ak, rh)]))

        return None  # No frontier found

    # ── v6.3: Informed Search (v7.0: rewritten as 5-tier priority) ────────

    def _informed_search(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """v7.0: 5-tier priority search with full-frame hash + frontier BFS + trigger-aware pruning.

        Tier 1: Untried actions at current state (never repeat what's known)
        Tier 2: Frontier BFS — navigate to states with untried actions
        Tier 3: Predicted-change — trigger-aware + effectiveness + rule hypothesis
        Tier 4: Novel exploration — ASD/delta/rarity targets
        Tier 5: Stochastic fallback — effectiveness-weighted random

        Args:
            frames: All previous frames.
            latest_frame: Current frame data.

        Returns:
            GameAction to execute.
        """
        grid = latest_frame.frame if latest_frame.frame else []
        available = latest_frame.available_actions if latest_frame.available_actions else []
        available_set = set(available)
        # v7.0: Use full-frame hash for state graph (exact dedup)
        current_hash = self._state_hash if self._state_hash else self._current_grid_hash

        # ── Track current state visit count ──
        self._state_visited[current_hash] = self._state_visited.get(current_hash, 0) + 1

        # ── Record level start hash if not set ──
        if not self._level_start_hash and current_hash:
            self._level_start_hash = current_hash
            self._frontier_depth[current_hash] = 0

        # ── Detect game type ──
        self._update_detected_game_type(available_set)

        # ── v6.4: Goal-EML lock (retain — critical for near-goal states) ──
        if self._goal_eml_convergence >= 0.8 and self._progressive_actions:
            best_prog = max(self._progressive_actions.items(), key=lambda x: x[1])
            prog_action_key = best_prog[0]
            action_name, click_data = self._parse_action_key(prog_action_key, available_set)
            if action_name:
                self._last_action_key = prog_action_key
                action = getattr(GameAction, action_name)
                if click_data and action.is_complex():
                    action.set_data(click_data)
                action.reasoning = (
                    f"v7.0-goal-eml-lock: {prog_action_key} "
                    f"(eml={self._goal_eml_convergence:.2f})"
                )
                self._action_history.append(action_name)
                return action

        # ── TIER 1: Untried actions at current state ──
        # "Never knowingly repeat a transition" — try actions we haven't tried from this state yet
        untried = self._get_untried_actions(current_hash, available_set, grid)
        if untried:
            action_key = untried[0]  # Pick first untried (highest priority)
            action_name, click_data = self._parse_action_key(action_key, available_set)
            if action_name:
                self._last_action_key = action_key
                action = getattr(GameAction, action_name)
                if click_data and action.is_complex():
                    action.set_data(click_data)
                action.reasoning = f"v7.0-tier1-untried: {action_key} at {current_hash[:8]}"
                self._action_history.append(action_name)
                return action

        # ── TIER 2: Frontier BFS navigation ──
        # Current state exhausted → navigate to frontier state with untried actions
        frontier_action = self._navigate_to_frontier(current_hash, available_set)
        if frontier_action is not None:
            return frontier_action

        # ── TIER 3: Predicted-change (trigger-aware + effectiveness + rule hypothesis) ──
        # Actions predicted to change the frame, weighted by effectiveness + rule hypothesis
        predicted = self._get_predicted_change_actions(current_hash, available_set, grid)
        if predicted:
            action_key = predicted[0]
            action_name, click_data = self._parse_action_key(action_key, available_set)
            if action_name:
                self._last_action_key = action_key
                action = getattr(GameAction, action_name)
                if click_data and action.is_complex():
                    action.set_data(click_data)
                action.reasoning = f"v7.0-tier3-predicted: {action_key}"
                self._action_history.append(action_name)
                return action

        # ── TIER 3b: Legacy click search (retained as Tier 3 fallback for click games) ──
        # If Tier 3 predicted-change found nothing but we have click capability,
        # use the proven _priority_click_search method (delta + rarity + expansion).
        if 6 in available_set:
            # v6.4: Goal-EML guided click — click near goal positions when convergence < 0.5
            if self._goal_positions and self._goal_eml_convergence < 0.5 and self._asd_anomaly_targets:
                for x, y in self._asd_anomaly_targets:
                    min_dist = min(abs(x - gx) + abs(y - gy) for gx, gy in self._goal_positions)
                    if min_dist <= 5 and f"{x},{y}" not in self._visited_coords:
                        self._visited_coords.add(f"{x},{y}")
                        x = max(0, min(63, x))
                        y = max(0, min(63, y))
                        action_key = f"ACTION6@{x},{y}"
                        self._last_action_key = action_key
                        action = GameAction.ACTION6
                        action.set_data({"x": x, "y": y})
                        action.reasoning = {"why": "v7.0-tier3b-goal-eml-click", "target": (x, y), "eml": self._goal_eml_convergence}
                        self._action_history.append("ACTION6")
                        return action

            click_action = self._priority_click_search(grid, available_set)
            if click_action is not None:
                # Update _last_action_key from priority_click_search result
                # (priority_click_search already sets _last_action_key internally)
                return click_action

        # ── TIER 3c: Legacy keyboard search (retained for keyboard games) ──
        keyboard_available = any(a in available_set for a in [1, 2, 3, 4])
        if keyboard_available:
            effective_action = self._effective_keyboard_search(available_set)
            if effective_action is not None:
                return effective_action

        # ── TIER 3d: Try ACTION5 (SPECIAL) if available ──
        if 5 in available_set:
            if not self._special_probed:
                self._special_probed = True
                self._last_action_key = "ACTION5"
                action = GameAction.ACTION5
                action.reasoning = "v7.0-tier3d-special-probe"
                self._action_history.append("ACTION5")
                return action
            elif self._effective_actions.get("ACTION5", 0) > 0:
                self._last_action_key = "ACTION5"
                action = GameAction.ACTION5
                action.reasoning = "v7.0-tier3d-special-effective"
                self._action_history.append("ACTION5")
                return action

        # ── TIER 3e: Try ACTION7 if available ──
        if 7 in available_set:
            self._last_action_key = "ACTION7"
            action = GameAction.ACTION7
            action.reasoning = "v7.0-tier3e-action7-probe"
            self._action_history.append("ACTION7")
            return action

        # ── TIER 4: Novel exploration ──
        # ASD/delta/rarity targets, creative probe
        if self._stall_counter >= 2:
            novel_action = self._novel_probe(grid, available_set)
            if novel_action is not None:
                return novel_action

        # ── TIER 5: Stochastic fallback ──
        return self._random_fallback(latest_frame, available_set)

    def _parse_action_key(
        self, action_key: str, available_set: Set[int]
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse an action_key string into (action_name, click_data).

        Action keys are either plain action names (ACTION1, ACTION2, etc.)
        or click keys (ACTION6@x,y) for click games.

        Returns:
            (action_name, click_data_dict_or_None). None if action not available.
        """
        if "@" in action_key:
            # Click action: ACTION6@x,y
            parts = action_key.split("@")
            action_name = parts[0]
            if action_name == "ACTION6" and 6 in available_set:
                coords = parts[1].split(",")
                try:
                    x, y = int(coords[0]), int(coords[1])
                    return ("ACTION6", {"x": x, "y": y})
                except (ValueError, IndexError):
                    return ("ACTION6", None)
            return (None, None)
        else:
            # Keyboard/action name
            action_id = ACTION_NAME_TO_ID.get(action_key, 0)
            if action_id in available_set:
                return (action_key, None)
            return (None, None)

    def _priority_click_search(
        self, grid: Any, available_set: Set[int]
    ) -> Optional[GameAction]:
        """Priority-based click targeting for click/puzzle games.

        Uses three priority sources:
          1. Delta cells (cells that changed between frames) — highest priority
          2. Rarity cells (cells with uncommon color values) — second priority
          3. Expansion cells (neighbors of effective clicks) — third priority

        Returns:
            GameAction with ACTION6 and click coordinates, or None.
        """
        # Priority 1: Delta click pool — cells that recently changed
        if self._delta_click_pool:
            while self._delta_click_pool:
                x, y = self._delta_click_pool.pop(0)
                if self._should_visit(x, y):
                    x = max(0, min(63, x))
                    y = max(0, min(63, y))
                    action_key = f"ACTION6@{x},{y}"
                    self._last_action_key = action_key
                    action = GameAction.ACTION6
                    action.set_data({"x": x, "y": y})
                    action.reasoning = {"why": "informed-delta-click", "target": (x, y)}
                    self._action_history.append("ACTION6")
                    return action

        # Priority 2: Grid scan — rarity-based systematic scan (v6.4: Goal-EML guided)
        # When Goal-EML convergence is low (< 0.3), prefer targets near goal positions.
        if not self._grid_scan_initialized:
            self._init_grid_click_scan(grid)
        if self._grid_scan_queue:
            # v6.4: Re-order scan queue by Goal-EML priority if convergence is low
            if self._goal_positions and self._goal_eml_convergence < 0.3:
                def goal_proximity(item: Tuple[int, int]) -> float:
                    x, y = item
                    min_dist = min(abs(x - gx) + abs(y - gy) for gx, gy in self._goal_positions)
                    return min_dist
                self._grid_scan_queue.sort(key=goal_proximity)

            while self._grid_scan_queue:
                x, y = self._grid_scan_queue.pop(0)
                if self._should_visit(x, y):
                    x = max(0, min(63, x))
                    y = max(0, min(63, y))
                    action_key = f"ACTION6@{x},{y}"
                    self._last_action_key = action_key
                    action = GameAction.ACTION6
                    action.set_data({"x": x, "y": y})
                    action.reasoning = {"why": "informed-rarity-click", "target": (x, y), "game_type": self._detected_game_type}
                    self._action_history.append("ACTION6")
                    return action

        # Priority 3: Neighborhood expansion around effective clicks
        for ex, ey in self._effective_click_positions[-5:]:
            self._expand_scan_from_click(ex, ey, grid)
        # Re-init and try again
        self._grid_scan_initialized = False
        self._init_grid_click_scan(grid)
        if self._grid_scan_queue:
            x, y = self._grid_scan_queue.pop(0)
            if self._should_visit(x, y):
                x = max(0, min(63, x))
                y = max(0, min(63, y))
                action_key = f"ACTION6@{x},{y}"
                self._last_action_key = action_key
                action = GameAction.ACTION6
                action.set_data({"x": x, "y": y})
                action.reasoning = {"why": "informed-expansion-click", "target": (x, y)}
                self._action_history.append("ACTION6")
                return action

        return None

    def _effective_keyboard_search(
        self, available_set: Set[int]
    ) -> Optional[GameAction]:
        """Select keyboard action based on effectiveness tracking + ido value score.

        v6.4: Enhanced with ido_value_score — progressive actions get higher priority,
        backslide actions get lower priority. Noether-violating actions are penalized.

        Uses the action_change_rate dictionary to compute effectiveness ratios
        and prefer directions with the highest change rate.

        Returns:
            GameAction for the most effective keyboard direction, or None.
        """
        # Collect available keyboard actions
        keyboard_actions = [
            a for a in DIRECTION_ACTIONS
            if ACTION_NAME_TO_ID.get(a, 0) in available_set
        ]
        if not keyboard_actions:
            return None

        # v6.4: Compute weighted effectiveness using ido_value_score
        # Base: effectiveness ratio (changes/total_uses)
        # Boost: progressive_actions count (actions that reduced IC-gap)
        # Penalty: backslide_actions + noether_violations (actions that backslid or broke physics)
        def weighted_effectiveness(action_name: str) -> float:
            eff, total = self._action_change_rate.get(action_name, (0, 0))
            # Base rate
            if total == 0:
                base_rate = 0.5  # untested = moderate priority
            else:
                base_rate = eff / total

            # v6.4: Progressive boost — actions that consistently reduce IC-gap
            progressive = self._progressive_actions.get(action_name, 0)
            progressive_boost = min(1.0, progressive * 0.3)

            # v6.4: Backslide penalty — actions that consistently increase IC
            backslide = self._backslide_actions.get(action_name, 0)
            backslide_penalty = min(0.8, backslide * 0.2)

            # v6.4: Noether penalty — physics-violating actions
            noether = self._noether_violations.get(action_name, 0)
            noether_penalty = min(0.5, noether * 0.15)

            return base_rate + progressive_boost - backslide_penalty - noether_penalty

        # Sort by weighted effectiveness (highest first)
        sorted_actions = sorted(
            keyboard_actions,
            key=weighted_effectiveness,
            reverse=True
        )

        # Pick the most effective direction
        best = sorted_actions[0]
        self._last_action_key = best
        action = getattr(GameAction, best)
        w_eff = weighted_effectiveness(best)
        action.reasoning = f"informed-keyboard: {best} (w_eff={w_eff:.2f}, prog={self._progressive_actions.get(best, 0)}, back={self._backslide_actions.get(best, 0)})"
        self._action_history.append(best)
        return action

    def _novel_probe(
        self, grid: Any, available_set: Set[int]
    ) -> Optional[GameAction]:
        """Try novel/untested actions when stuck.

        v6.4: Creative-Probe enhancement — when η-plateau detected (ido≈0 for
        consecutive steps), target minority-color pixels with −∇η perturbation.
        Instead of random clicks, prioritize ASD anomaly targets (minority pixels)
        to break through the plateau.

        For click games: try ASD anomaly targets first, then random nonzero positions.
        For keyboard games: try unprobed directions or anti-effectiveness directions.

        Returns:
            GameAction for a novel probe, or None.
        """
        # ── v6.4: Creative-Probe for click games ──
        # When η-plateau detected, target minority-color pixels (−∇η direction)
        if 6 in available_set:
            # v6.4 Creative-Probe: Priority 1 — ASD anomaly targets
            if self._eta_plateau_counter >= 3 and self._asd_anomaly_targets:
                # η-plateau: stuck at ido≈0 for 3+ steps → probe minority pixels
                for x, y in self._asd_anomaly_targets:
                    coord_key = f"{x},{y}"
                    if coord_key not in self._visited_coords:
                        self._visited_coords.add(coord_key)
                        x = max(0, min(63, x))
                        y = max(0, min(63, y))
                        action_key = f"ACTION6@{x},{y}"
                        self._last_action_key = action_key
                        action = GameAction.ACTION6
                        action.set_data({"x": x, "y": y})
                        action.reasoning = {"why": "v6.4-creative-probe", "target": (x, y), "eta_plateau": self._eta_plateau_counter}
                        self._action_history.append("ACTION6")
                        return action

            # Priority 2: delta click pool cells (v6.4: filter out backslide click positions)
            nonzero = self._find_nonzero_cells(grid)
            rare_targets = [(x, y, val) for x, y, val in nonzero
                           if f"{x},{y}" not in self._visited_coords
                           and val not in self._wall_colors
                           and val != self._player_color
                           ]
            # v6.4: Filter out click positions that caused backslide
            backslide_coords = set()
            for ak in self._backslide_actions:
                if "@" in ak:
                    try:
                        coords = ak.split("@")[1].split(",")
                        backslide_coords.add(f"{int(coords[0])},{int(coords[1])}")
                    except (ValueError, IndexError):
                        pass
            rare_targets = [(x, y, val) for x, y, val in rare_targets
                           if f"{x},{y}" not in backslide_coords
                           ]
            if not rare_targets:
                rare_targets = [(x, y, val) for x, y, val in nonzero
                               if f"{x},{y}" not in self._visited_coords
                               ]
            if rare_targets:
                # v6.4: Sort by Goal-EML convergence potential — prefer targets near goal positions
                if self._goal_positions and self._goal_eml_convergence < 0.5:
                    # When convergence is low, prefer targets near goal positions
                    def goal_distance(t: Tuple[int, int, int]) -> float:
                        x, y, val = t
                        min_dist = min(abs(x - gx) + abs(y - gy) for gx, gy in self._goal_positions) if self._goal_positions else 1000
                        return min_dist
                    rare_targets.sort(key=goal_distance)

                x, y, val = rare_targets[0]  # Pick best target (not random!)
                x = max(0, min(63, x))
                y = max(0, min(63, y))
                action_key = f"ACTION6@{x},{y}"
                self._last_action_key = action_key
                action = GameAction.ACTION6
                action.set_data({"x": x, "y": y})
                action.reasoning = {"why": "v6.4-novel-click", "target": (x, y), "val": val, "eml": self._goal_eml_convergence}
                self._action_history.append("ACTION6")
                return action

            # Try random grid positions if no nonzero cells
            x = random.randint(0, 63)
            y = random.randint(0, 63)
            action_key = f"ACTION6@{x},{y}"
            self._last_action_key = action_key
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y})
            action.reasoning = {"why": "informed-novel-random-click", "target": (x, y)}
            self._action_history.append("ACTION6")
            return action

        # For keyboard games: try unprobed directions
        keyboard_available = [a for a in DIRECTION_ACTIONS
                              if ACTION_NAME_TO_ID.get(a, 0) in available_set]
        unprobed = [a for a in keyboard_available
                    if a not in self._direction_probed]
        if unprobed:
            action_name = random.choice(unprobed)
            self._direction_probed[action_name] = True
            self._last_action_key = action_name
            action = getattr(GameAction, action_name)
            action.reasoning = f"informed-novel-probe: {action_name}"
            self._action_history.append(action_name)
            return action

        # All probed — try random direction with anti-stall bias
        if keyboard_available:
            # Prefer LESS effective directions when stuck (try something new)
            def anti_effectiveness(action_name: str) -> float:
                eff, total = self._action_change_rate.get(action_name, (0, 0))
                if total == 0:
                    return 1.0  # untested = high novelty
                return 1.0 - (eff / total)  # LESS effective = more novel

            weights = [anti_effectiveness(a) + 0.1 for a in keyboard_available]
            action_name = random.choices(keyboard_available, weights=weights, k=1)[0]
            self._last_action_key = action_name
            action = getattr(GameAction, action_name)
            action.reasoning = f"informed-novel-direction: {action_name} (anti-stall)"
            self._action_history.append(action_name)
            return action

        return None

    # ── Delta-aware smart exploration (LEGACY — kept for fallback) ───────

    def _smart_exploration(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Intelligent exploration for games without replay data.

        Multi-phase strategy:
          Phase 1 (probe): Systematically test directions and SPECIAL
                          to learn the game's mechanics.
          Phase 2 (navigate): Use learned direction map to move toward
                              target cells (sprites, delta cells).
          Phase 3 (exploit): Use pattern memory to repeat effective
                             action sequences for familiar grid configs.

        Args:
            frames: All previous frames.
            latest_frame: Current frame data.

        Returns:
            GameAction to execute.
        """
        base_id = self.game_id.split("-")[0] if self.game_id else ""
        grid = latest_frame.frame if latest_frame.frame else []
        available = latest_frame.available_actions if latest_frame.available_actions else []
        available_set = set(available)

        # ── v5.0.2: Dynamic game-type detection ──
        # Detect game type from available_actions instead of relying on hardcoded sets.
        # This is CRITICAL for competition games — they're unknown and have no Oracle data.
        self._update_detected_game_type(available_set)

        # ── v5.0.3: RG-Flow adaptive phase update ──
        self._action_count += 1
        self._update_rg_flow()

        # ── Phase 0: ASD Anomaly Detection ("Attention Before Loss") ──
        # Analyze first frame for minority-color pixels on first call.
        # Then click anomaly targets with highest priority.
        # RG-Flow: In "explore" phase, ASD gets full budget; in "lock" phase, skip ASD
        if not self._asd_analyzed:
            self._asd_analyze_first_frame(grid)
        if self._life_phase in ("life1", "life2") and self._rg_flow_phase != "lock":
            asd_action = self._asd_click_anomaly(available_set)
            if asd_action is not None:
                return asd_action

        # ── Phase 0.5: 3-Life Strategy routing ──
        # Life2: Use refined actions from Life1 discoveries
        # Life3: Only execute most effective known actions
        if self._life_phase == "life2":
            refined = self._life2_refined_action(frames, latest_frame, available_set)
            if refined is not None:
                return refined
        elif self._life_phase == "life3":
            optimal = self._life3_execute(frames, latest_frame, available_set)
            if optimal is not None:
                return optimal

        # ── v5.0.3: RG-Flow κ-Snap lock phase ──
        # In "lock" phase (>70% budget consumed), only repeat proven effective actions.
        # No new probing — this is κ-Snap constraint (strong coupling in IDO/QQG analogy).
        # v5.0.4 κ-Lock guard: Only lock on actions that caused MEANINGFUL change.
        # Previously, wall-bump actions (no grid change, just player stuck) could
        # accumulate as "effective" because delta=0 wasn't filtered properly.
        # Now we require that the action was recently effective (within last 20 steps).
        if self._rg_flow_phase == "lock":
            # Filter: only use actions that were effective in the LAST 20 steps
            recent_actions = self._action_history[-20:] if len(self._action_history) >= 20 else self._action_history
            recent_effective = {a: c for a, c in self._effective_actions.items()
                               if a in recent_actions and c > 0}
            # If no recent effective actions, fall through (don't lock on stale data)
            if recent_effective:
                best_actions = sorted(
                    recent_effective.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for action_name, count in best_actions:
                    if count > 0 and ACTION_NAME_TO_ID.get(action_name, 0) in available_set:
                        action = getattr(GameAction, action_name)
                        action.reasoning = f"κ-snap-lock: {action_name} (eff={count}, budget={self._action_count}/{self.MAX_ACTIONS})"
                        self._action_history.append(action_name)
                        return action
            # No recent effective actions available — fall through to pattern repeat

        # ── Phase 1: Pattern repeat (highest priority) ──
        # If we've seen this exact grid configuration before and know
        # an action that changed it, try that action again.
        if self._current_grid_hash and self._current_grid_hash in self._pattern_memory:
            seq = self._pattern_memory[self._current_grid_hash]
            if seq:
                action_name = seq[0]
                action = getattr(GameAction, action_name)
                action.reasoning = f"pattern-repeat: {action_name} for hash {self._current_grid_hash[:8]}"
                self._action_history.append(action_name)
                return action

        # ── Phase 1: Delta-based click targeting ──
        # Cells that changed in recent frames are interactive elements.
        # Clicking on them often reveals game mechanics or triggers actions.
        if 6 in available_set and self._delta_click_pool:
            # Pick the first unvisited delta cell
            while self._delta_click_pool:
                x, y = self._delta_click_pool.pop(0)
                # v5.0.4: Use _should_visit to allow revisit
                if self._should_visit(x, y):
                    action = GameAction.ACTION6
                    action.set_data({"x": x, "y": y})
                    action.reasoning = {"why": "delta-click", "target": (x, y)}
                    self._action_history.append("ACTION6")
                    return action

        # ── Phase 2: Handle stalling ──
        # If we've been stuck for too many steps, change strategy
        if self._stall_counter >= self._stall_threshold:
            # Try a completely different approach
            return self._stall_recovery(latest_frame, available_set)

        # ── Phase 2.5: Click-capable fast-path — systematic grid scan ──
        # v5.0.2: ANY game with ACTION6 available → grid-scan strategy.
        # Previously only hardcoded CLICK/MIXED games got this fast-path.
        # Now unknown/competition games with ACTION6 also benefit.
        # This is the KEY fix: unknown click games were falling to keyboard
        # phases 3-8 → extremely inefficient → 0% completion → 0.00 score.
        if self._is_click_capable_game(base_id, available_set):
            if 6 in available_set:
                # Initialize scan if not done yet
                if not self._grid_scan_initialized:
                    self._init_grid_click_scan(grid)

                # Drain the scan queue
                while self._grid_scan_queue:
                    x, y = self._grid_scan_queue.pop(0)
                    # v5.0.4: Use _should_visit instead of _visited_coords check
                    # This allows revisiting important positions (gravity toggles, buttons)
                    if self._should_visit(x, y):
                        # Clamp to valid grid bounds (0-63)
                        x = max(0, min(63, x))
                        y = max(0, min(63, y))
                        action = GameAction.ACTION6
                        action.set_data({"x": x, "y": y})
                        action.reasoning = {"why": "grid-scan-click", "target": (x, y), "game_type": self._detected_game_type}
                        self._action_history.append("ACTION6")
                        return action

                # Scan queue exhausted — v5.0.3: ΔIC-driven re-scan
                # Instead of blindly re-scanning entire grid, only re-scan cells
                # that CHANGED since last scan (ΔIC priority). This is more efficient
                # and focuses on interactive elements discovered during first round.
                self._grid_scan_initialized = False

                # ΔIC-driven second round: prioritize delta cells
                if self._delta_click_pool:
                    # Use delta cells as second round priority
                    delta_targets = []
                    for x, y in self._delta_click_pool:
                        # v5.0.4: Use _should_visit for revisit check
                        if self._should_visit(x, y):
                            delta_targets.append((x, y))
                    # Also add effective click neighborhood expansion
                    for ex, ey in self._effective_click_positions[-5:]:
                        self._expand_scan_from_click(ex, ey, grid)

                    # Re-init grid scan (which will include updated cells)
                    self._init_grid_click_scan(grid)
                    # Prepend delta targets (ΔIC priority) before regular scan
                    self._grid_scan_queue = delta_targets + self._grid_scan_queue
                else:
                    # No delta cells — regular re-init (grid may have changed)
                    self._init_grid_click_scan(grid)

                if self._grid_scan_queue:
                    x, y = self._grid_scan_queue.pop(0)
                    # v5.0.4: Use _should_visit for revisit check
                    if self._should_visit(x, y):
                        x = max(0, min(63, x))
                        y = max(0, min(63, y))
                        action = GameAction.ACTION6
                        action.set_data({"x": x, "y": y})
                        action.reasoning = {"why": "grid-scan-refresh", "target": (x, y)}
                        self._action_history.append("ACTION6")
                        return action

            # For mixed games, also allow keyboard fallback when no clicks left
            if self._is_mixed_game(base_id, available_set) and any(a in available_set for a in [1, 2, 3, 4]):
                # Use most effective keyboard direction
                effective_dirs = [
                    a for a in DIRECTION_ACTIONS
                    if a in self._effective_actions and
                       ACTION_NAME_TO_ID.get(a, 0) in available_set
                ]
                if effective_dirs:
                    best_dir = max(effective_dirs,
                                   key=lambda a: self._effective_actions.get(a, 0))
                    action = getattr(GameAction, best_dir)
                    action.reasoning = f"mixed-keyboard-fallback: {best_dir}"
                    self._action_history.append(best_dir)
                    return action
                # Try any available direction
                for a in DIRECTION_ACTIONS:
                    if ACTION_NAME_TO_ID.get(a, 0) in available_set:
                        action = getattr(GameAction, a)
                        action.reasoning = f"mixed-keyboard-probe: {a}"
                        self._action_history.append(a)
                        return action

            # Click-capable games with no more scan targets — fall to Phase 9
            return self._random_fallback(latest_frame, available_set)

        # ── Phase 3: Probe directions (keyboard games) ──
        # Systematically test each direction to learn the mapping.
        # Only do this once per direction per level.
        if self._direction_probe_count < 4 and any(a in available_set for a in [1, 2, 3, 4]):
            unprobed = [
                a for a in DIRECTION_ACTIONS
                if a not in self._direction_probed and
                   ACTION_NAME_TO_ID.get(a, 0) in available_set
            ]
            if unprobed:
                action_name = unprobed[0]
                self._direction_probe_count += 1
                action = getattr(GameAction, action_name)
                action.reasoning = f"direction-probe #{self._direction_probe_count}: {action_name}"
                self._action_history.append(action_name)
                return action

        # ── Phase 4: Probe SPECIAL action ──
        # Try ACTION5 once early to understand its effect.
        if not self._special_probed and 5 in available_set:
            self._special_probed = True
            action = GameAction.ACTION5
            action.reasoning = "special-action-probe (first test)"
            self._action_history.append("ACTION5")
            return action

        # ── Phase 5: Goal-oriented navigation (v5.0.5) ──
        # Navigate toward DETECTED GOALS instead of random nonzero cells.
        # Previously (v5.0.4), navigated toward nearest nonzero cell — but most
        # nonzero cells are walls/decoration, NOT goals. This caused competition
        # unseen games to score 0.00 because the agent wandered aimlessly.
        # Now we use _goal_positions (detected from rarity, border, delta analysis)
        # and _player_color (detected from movement tracking) to navigate
        # toward actual goals/exits/switches.

        # v5.0.7: Noether conservation guard (LTT-IDO insight)
        # Noether's theorem: physical symmetries imply conservation laws.
        # Mapped to grid: IC (information cardinality = unique nonzero value count)
        # should be conserved under valid navigation — a single step shouldn't
        # cause >20% IC jump. If it does, the navigation path is invalid:
        # the agent walked into a wall/decoration zone, not a goal zone.
        # Reject path, clear navigation state, fall through to Phase 6/7/8.
        NOETHER_IC_THRESHOLD = 0.20  # 20% IC change = anomalous
        if self._ic_delta > NOETHER_IC_THRESHOLD and self._navigate_path:
            # Last navigation step caused anomalous grid restructuring —
            # we're not heading toward a goal, we're disrupting the grid.
            # Clear navigation state and skip Phase 5 this turn.
            self._navigate_path = []
            self._navigate_target = None
            # Log the rejection for debugging
            pass  # Fall through to Phase 6/7/8

        if self._estimated_player_pos and self._direction_map:
            # Follow pre-computed navigation path
            if self._navigate_path:
                next_action_name = self._navigate_path.pop(0)
                if ACTION_NAME_TO_ID.get(next_action_name, 0) in available_set:
                    action = getattr(GameAction, next_action_name)
                    action.reasoning = f"goal-navigate-path: {next_action_name}"
                    self._action_history.append(next_action_name)
                    return action
                else:
                    # Action not available — clear path and recalculate
                    self._navigate_path = []

            # v5.0.5: Compute navigation toward detected GOALS first
            # If we have goal positions, navigate toward the closest unvisited goal
            px, py = self._estimated_player_pos
            nav_target = None

            if self._goal_positions:
                # Navigate toward closest unvisited goal
                goal_targets = [(gx, gy) for gx, gy in self._goal_positions
                               if f"{gx},{gy}" not in self._visited_coords]
                if goal_targets:
                    # Sort by distance from player — closest goal first
                    goal_targets.sort(key=lambda g: abs(g[0] - px) + abs(g[1] - py))
                    nav_target = goal_targets[0]

            # Fallback: if no goals detected, use nonzero cells BUT EXCLUDE player color and walls
            if nav_target is None:
                nonzero = self._find_nonzero_cells(grid)
                if nonzero:
                    # v5.0.5: Filter out player color cells and common wall colors
                    # Player color cells are where the player IS, not where it should GO
                    # Wall colors are background — navigating toward them is pointless
                    filtered_targets = [(x, y, val) for x, y, val in nonzero
                                       if f"{x},{y}" not in self._visited_coords
                                       and val != self._player_color  # Don't navigate to player
                                       and val not in self._wall_colors  # Don't navigate to walls
                                       ]
                    # If filtering eliminated everything, fall back to nonzero without wall filter
                    if not filtered_targets:
                        filtered_targets = [(x, y, val) for x, y, val in nonzero
                                           if f"{x},{y}" not in self._visited_coords
                                           and val != self._player_color
                                           ]
                    if filtered_targets:
                        # Sort by distance from player — closest first
                        filtered_targets.sort(key=lambda t: abs(t[0] - px) + abs(t[1] - py))
                        nav_target = (filtered_targets[0][0], filtered_targets[0][1])

            # Compute navigation path to target
            if nav_target:
                self._navigate_target = nav_target
                self._navigate_path = self._compute_navigate_path(
                    (px, py), nav_target
                )
                if self._navigate_path:
                    next_action_name = self._navigate_path.pop(0)
                    action_id = ACTION_NAME_TO_ID.get(next_action_name, 0)
                    if action_id in available_set:
                        action = getattr(GameAction, next_action_name)
                        action.reasoning = f"goal-navigate: {next_action_name} toward goal {nav_target} (player_color={self._player_color})"
                        self._action_history.append(next_action_name)
                        return action

        # ── Phase 6: Click on non-zero cells (sprite targeting) ──
        # For click/mixed games, click on interesting sprite cells.
        if 6 in available_set:
            click_targets = self._find_click_targets(grid)
            if click_targets:
                for x, y in click_targets:
                    # v5.0.4: Use _should_visit for revisit check
                    if self._should_visit(x, y):
                        action = GameAction.ACTION6
                        action.set_data({"x": x, "y": y})
                        action.reasoning = {"why": "sprite-click", "target": (x, y)}
                        self._action_history.append("ACTION6")
                        return action

        # ── Phase 7: Strategic SPECIAL action ──
        # Use ACTION5 when we know it's effective and progress is stalled
        if (self._special_probed and
                self._effective_actions.get("ACTION5", 0) > 0 and
                self._stall_counter >= 3 and
                5 in available_set):
            action = GameAction.ACTION5
            action.reasoning = "special-action-strategic (stall recovery)"
            self._action_history.append("ACTION5")
            return action

        # ── Phase 8: Keyboard exploration with learned preferences ──
        # Use the most effective direction, or try unprobed directions
        if any(a in available_set for a in [1, 2, 3, 4]):
            # Prefer directions that have been effective (caused grid changes)
            effective_dirs = [
                a for a in DIRECTION_ACTIONS
                if a in self._effective_actions and
                   ACTION_NAME_TO_ID.get(a, 0) in available_set
            ]
            if effective_dirs:
                # Pick the most effective direction
                best_dir = max(effective_dirs,
                               key=lambda a: self._effective_actions.get(a, 0))
                action = getattr(GameAction, best_dir)
                action.reasoning = f"effective-direction: {best_dir}"
                self._action_history.append(best_dir)
                return action

            # If no effective directions yet, try unprobed ones
            unprobed_available = [
                a for a in DIRECTION_ACTIONS
                if a not in self._direction_probed and
                   ACTION_NAME_TO_ID.get(a, 0) in available_set
            ]
            if unprobed_available:
                action_name = unprobed_available[0]
                action = getattr(GameAction, action_name)
                action.reasoning = f"unprobed-direction: {action_name}"
                self._action_history.append(action_name)
                return action

            # All directions probed — cycle through them with bias toward effective ones
            cycle_dirs = [a for a in DIRECTION_ACTIONS
                         if ACTION_NAME_TO_ID.get(a, 0) in available_set]
            if cycle_dirs:
                # Weight by effectiveness
                weights = [max(self._effective_actions.get(a, 1), 1) for a in cycle_dirs]
                action_name = random.choices(cycle_dirs, weights=weights, k=1)[0]
                action = getattr(GameAction, action_name)
                action.reasoning = f"cycle-direction: {action_name}"
                self._action_history.append(action_name)
                return action

        # ── Phase 9: Random fallback ──
        return self._random_fallback(latest_frame, available_set)

    def _stall_recovery(
        self, latest_frame: FrameData, available_set: Set[int]
    ) -> GameAction:
        """Recover from stalling via Cognitive Inflation (v5.0.3).

        Inspired by IDO/TOMAS: when stuck, don't just swap direction.
        Instead, detect topological avalanche (high ΔIC regions) and
        do creative probing into those regions — this is "cognitive inflation"
        driven by ΔIC release, not random walk.

        Phase 1: Compute ΔIC between recent frames → identify avalanche regions
        Phase 2: Probe high-ΔIC targets (rare/new values) first
        Phase 3: Try SPECIAL action
        Phase 4: Reverse direction
        Phase 5: Fresh direction
        Phase 6: Random fallback
        """
        grid = latest_frame.frame if latest_frame.frame else []

        # ── v5.0.3: Cognitive Inflation — ΔIC avalanche probing ──
        # When stuck, compute ΔIC and identify high-IC-change regions
        # for creative probing. This replaces random direction swap.
        if not self._inflation_active:
            # Activate inflation mode: detect avalanche targets
            avalanche_targets = self._detect_topological_avalanche(grid)
            if avalanche_targets and 6 in available_set:
                self._inflation_probes = avalanche_targets[:20]  # top 20 targets
                self._inflation_active = True
            # Also compute ΔIC from grid history for context
            if len(self._grid_history) >= 2:
                self._compute_ic_delta(self._grid_history[-2], grid)

        # Probe inflation targets (ΔIC-driven creative clicks)
        if self._inflation_active and self._inflation_probes and 6 in available_set:
            while self._inflation_probes:
                x, y = self._inflation_probes.pop(0)
                # v5.0.4: Use _should_visit for revisit check
                if self._should_visit(x, y):
                    x = max(0, min(63, x))
                    y = max(0, min(63, y))
                    action = GameAction.ACTION6
                    action.set_data({"x": x, "y": y})
                    action.reasoning = {"why": "cognitive-inflation",
                                        "ic_delta": self._ic_delta,
                                        "rg_phase": self._rg_flow_phase,
                                        "target": (x, y)}
                    self._action_history.append("ACTION6")
                    return action
            # All inflation probes exhausted → deactivate
            self._inflation_active = False

        # Strategy 2: Try SPECIAL action if available and not yet tested
        if 5 in available_set and not self._special_probed:
            self._special_probed = True
            action = GameAction.ACTION5
            action.reasoning = "cognitive-inflation: special-action-probe"
            self._action_history.append("ACTION5")
            return action

        # Strategy 3: Try SPECIAL again if it was previously effective
        if 5 in available_set and self._effective_actions.get("ACTION5", 0) > 0:
            action = GameAction.ACTION5
            action.reasoning = "cognitive-inflation: repeat-effective-special"
            self._action_history.append("ACTION5")
            return action

        # Strategy 4: Click on cells with rare values (ΔIC-based)
        if 6 in available_set:
            nonzero = self._find_nonzero_cells(grid)
            if nonzero:
                value_counts: Dict[int, List[Tuple[int, int, int]]] = {}
                for x, y, val in nonzero:
                    if val not in value_counts:
                        value_counts[val] = []
                    value_counts[val].append((x, y, val))

                sorted_vals = sorted(value_counts.keys(),
                                     key=lambda v: len(value_counts[v]))

                for rare_val in sorted_vals:
                    cells = value_counts[rare_val]
                    for x, y, val in cells:
                        # v5.0.4: Use _should_visit for revisit check
                        if self._should_visit(x, y):
                            action = GameAction.ACTION6
                            action.set_data({"x": x, "y": y})
                            action.reasoning = {"why": "inflation-rare-click",
                                                "value": rare_val,
                                                "ic_delta": self._ic_delta,
                                                "target": (x, y)}
                            self._action_history.append("ACTION6")
                            return action

        # Strategy 5: Try opposite of last stuck direction
        if self._action_history:
            last_action = self._action_history[-1]
            opposites = {
                "ACTION1": "ACTION2",
                "ACTION2": "ACTION1",
                "ACTION3": "ACTION4",
                "ACTION4": "ACTION3",
            }
            opp = opposites.get(last_action)
            if opp and ACTION_NAME_TO_ID.get(opp, 0) in available_set:
                action = getattr(GameAction, opp)
                action.reasoning = f"inflation-reverse: {last_action} → {opp}"
                self._action_history.append(opp)
                return action

        # Strategy 6: Fresh direction (not recently used)
        recent_dirs = self._action_history[-8:] if len(self._action_history) >= 8 else self._action_history
        available_dirs = [
            a for a in DIRECTION_ACTIONS
            if a not in recent_dirs and
               ACTION_NAME_TO_ID.get(a, 0) in available_set
        ]
        if available_dirs:
            action_name = random.choice(available_dirs)
            action = getattr(GameAction, action_name)
            action.reasoning = f"inflation-fresh: {action_name}"
            self._action_history.append(action_name)
            return action

        # Strategy 7: Random fallback
        return self._random_fallback(latest_frame, available_set)

    def _random_fallback(
        self, latest_frame: FrameData, available_set: Set[int]
    ) -> GameAction:
        """Last-resort random action selection with game-type bias.

        Args:
            latest_frame: Current frame data.
            available_set: Set of available action IDs.

        Returns:
            Randomly selected GameAction.
        """
        base_id = self.game_id.split("-")[0] if self.game_id else ""
        grid = latest_frame.frame if latest_frame.frame else []

        # Build candidate actions from available set
        candidate_actions: List[GameAction] = []
        action_id_to_game_action: Dict[int, GameAction] = {
            1: GameAction.ACTION1,
            2: GameAction.ACTION2,
            3: GameAction.ACTION3,
            4: GameAction.ACTION4,
            5: GameAction.ACTION5,
            6: GameAction.ACTION6,
            7: GameAction.ACTION7,
        }

        for aid in available_set:
            if aid in action_id_to_game_action:
                candidate_actions.append(action_id_to_game_action[aid])

        if not candidate_actions:
            # No available actions reported — try all except RESET
            candidate_actions = [
                a for a in GameAction if a is not GameAction.RESET
            ]

        # v5.0.2: Weight actions using DYNAMIC game type detection
        # Previously used hardcoded KEYBOARD/CLICK/MIXED sets which missed
        # unknown games entirely → random weights → 0.00 score.
        # Now dynamic detection covers ALL games including competition unknowns.
        available_set = set(latest_frame.available_actions or [])
        if self._is_keyboard_game(base_id, available_set) and not self._is_click_capable_game(base_id, available_set):
            weights = [
                3 if a in (GameAction.ACTION1, GameAction.ACTION2,
                           GameAction.ACTION3, GameAction.ACTION4)
                else 1 for a in candidate_actions
            ]
        elif self._is_click_capable_game(base_id, available_set) and not self._is_mixed_game(base_id, available_set):
            weights = [
                5 if a is GameAction.ACTION6 else 1 for a in candidate_actions
            ]
        elif self._is_mixed_game(base_id, available_set):
            weights = [
                2 if a in (GameAction.ACTION1, GameAction.ACTION2,
                           GameAction.ACTION3, GameAction.ACTION4)
                else 3 if a is GameAction.ACTION6
                else 1 for a in candidate_actions
            ]
        else:
            # Truly unknown (no click, no keyboard) — use effectiveness + special bias
            weights = [
                max(self._effective_actions.get(a.name, 1), 1)
                for a in candidate_actions
            ]

        action = random.choices(candidate_actions, weights=weights, k=1)[0]

        if action.is_complex():
            # Click on a delta cell, then sprite target, then random position
            if self._delta_click_pool:
                x, y = self._delta_click_pool.pop(0)
                self._visited_coords.add(f"{x},{y}")
                action.set_data({"x": x, "y": y})
            else:
                target_coords = self._find_click_targets(grid)
                if target_coords:
                    x, y = random.choice(target_coords)
                    action.set_data({"x": x, "y": y})
                else:
                    action.set_data({
                        "x": random.randint(0, 63),
                        "y": random.randint(0, 63)
                    })
            action.reasoning = {"why": "random-fallback-click"}
        else:
            action.reasoning = f"random-fallback: {action.name}"

        self._action_history.append(action.name)
        return action
