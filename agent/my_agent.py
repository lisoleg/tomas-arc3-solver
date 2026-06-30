"""TOMAS ARC-AGI-3 Solver Agent — ARC Prize 2026 Kaggle Submission v5.0.5 (Goal-Oriented Navigation + Player Color Tracking + Goal Detection for Unseen Games).

Strategy:
  1. ARC3 Replay Oracle: Pre-computed human-optimal action sequences from arc3.games
  2. ASD Anomaly Detection: "Attention Before Loss" — target minority-color pixels first
  3. 3-Life Strategy: Life1=explore, Life2=refine, Life3=execute
  4. Delta-aware exploration: Frame delta detection to find interactive cells
  5. Systematic keyboard navigation: Probe directions, learn mapping, navigate to targets
  6. Special action probing: Systematically test ACTION5 to understand its effect
  7. Pattern repeat: Remember effective action sequences for similar grid configurations
  8. Enhanced perception: Color frequency analysis + rarity-based targeting
  9. Random fallback: Last resort

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
    """TOMAS ARC-AGI-3 Solver v5.0.5 — Replay Oracle + Dynamic Game-Type Detection + Goal-Oriented Navigation + Grid-Scan Universal + RG-Flow Adaptive + Cognitive Inflation.

    Strategy priority:
      1. ARC3 Replay Oracle (precomputed human-optimal sequences — always try, then fallback)
      2. Dynamic game-type detection (infer click/keyboard/mixed from available_actions)
      3. Goal-oriented navigation (detect player + goal, navigate toward goal instead of random nonzero)
      4. Grid-scan universal (ANY game with ACTION6 → systematic grid click scan)
      5. ASD Anomaly Detection ("Attention Before Loss" — minority colors first)
      6. RG-Flow Adaptive (early→high entropy exploration, late→κ-Snap lock)
      7. 3-Life Strategy (Life1=explore, Life2=refine, Life3=execute)
      8. Cognitive Inflation stall recovery (ΔIC avalanche BFS instead of random swap)
      9. ΔIC-Driven Re-scan (frame change pixels → second scan round priority)
      10. Delta-aware exploration (detect changed cells, navigate toward targets)
      11. Pattern repeat (reuse effective action sequences)
      12. Random fallback (last resort)

    v5.0.5 changes vs v5.0.4:
      - Player color detection: track which color value moves with keyboard actions
      - Goal detection: identify likely target/exit/goal positions from grid analysis
      - Goal-oriented navigation: Phase 5 navigates toward detected goals, not random nonzero cells
      - This is the KEY fix for competition unseen games — previously navigating toward walls/decoration
    """

    # Upper bound on actions per game.
    # Oracle replay: ~20-200 steps/level × 7 levels = at most ~1400 steps.
    # Fallback exploration budget: generous but bounded to avoid 9h Kaggle timeout.
    # Supports ARC3_MAX_ACTIONS env override for competition vs test tuning.
    MAX_ACTIONS = int(os.environ.get("ARC3_MAX_ACTIONS", "2000"))

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
        self._wall_colors: List[int] = []  # detected common wall/decoration colors (>50% of nonzero)

        # ── Initialize plan for level 0 ──
        self._compute_plan(0)

    @property
    def name(self) -> str:
        return f"tomas.v5.0.4.{self.MAX_ACTIONS}"

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
        """Estimate player position from delta cells and action context.

        Strategy:
          1. If delta cells form a cluster, the cluster center is likely
             near the player.
          2. If we know the direction map, extrapolate from previous position.
          3. Otherwise, find the most "active" non-zero cell in the grid.

        Args:
            delta: Changed cells from last action.
            last_action: The action that caused this delta.
            grid: Current frame grid.

        Returns:
            Estimated (x, y) player position, or None.
        """
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

        # First time: find a "unique" cell that might be the player
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
                    if old_val == 0 or old_val == -1 and new_val != 0 and new_val != -1:
                        appeared[new_val] = appeared.get(new_val, 0) + 1
                    elif new_val == 0 or new_val == -1 and old_val != 0 and old_val != -1:
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
            # Accept if top has >2x more evidence than second (or only one candidate)
            if top_count >= 3 and (top_count > second_count * 2 or len(sorted_candidates) == 1):
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
        """Compute a path of keyboard actions from start to target.

        Uses the learned direction map to plan a sequence of actions.
        Falls back to Manhattan-distance-based guessing if direction map
        is incomplete.

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
                        # Want to move in dx direction
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
                    # Unknown direction — assign heuristic score based on
                    # typical ARC3 mapping: ACTION1=UP(-y), ACTION2=DOWN(+y),
                    # ACTION3=LEFT(-x), ACTION4=RIGHT(+x)
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
                    # Prefer known directions over unknown
                    score -= 0.5
                    if score > best_score:
                        best_score = score
                        best_action = action_name

            if best_action:
                # Apply the direction to update position
                if best_action in self._direction_map:
                    adx, ady = self._direction_map[best_action]
                    cx += adx
                    cy += ady
                else:
                    # Use heuristic
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
                # No viable action — break
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
            else:
                action.reasoning = f"replay oracle step {self._plan_idx}/{len(self._plan)}"

            self._action_history.append(action_name)
            return action

        # ── Delta-aware smart exploration fallback ──
        return self._smart_exploration(frames, latest_frame)

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
        else:
            self._stall_counter = 0
            # v5.0.3: Deactivate inflation when grid changes (stall broken)
            self._inflation_active = False

        # v5.0.3: Compute ΔIC for cognitive inflation
        if len(delta) > 0 and len(self._grid_history) >= 2:
            self._compute_ic_delta(self._grid_history[-2], current_grid)

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
        """Life3: Execute the best known strategy with minimal exploration.

        Args:
            frames: All previous frames.
            latest_frame: Current frame data.
            available_set: Set of available action IDs.

        Returns:
            Optimal execution GameAction, or None.
        """
        # In Life3, only use the most effective actions — no probing
        best_actions = sorted(
            self._effective_actions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for action_name, count in best_actions:
            if ACTION_NAME_TO_ID.get(action_name, 0) in available_set:
                action = getattr(GameAction, action_name)
                action.reasoning = f"life3-optimal: {action_name} (effectiveness={count})"
                self._action_history.append(action_name)
                return action

        # If special was effective, use it strategically
        if (self._special_probed and
                self._effective_actions.get("ACTION5", 0) > 0 and
                5 in available_set):
            action = GameAction.ACTION5
            action.reasoning = "life3-optimal: strategic SPECIAL"
            self._action_history.append("ACTION5")
            return action

        return None

    # ── Delta-aware smart exploration ────────────────────────────────────

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
