"""
L4 Fuse Gate: Connects L2 ProgramNode → L3 State → env.step(action).

Supports:
- MOVE_TO_GRID_OBJECT_ID (L2 intent → L3 resolution → L5 action)
- INTERACT (pickup / unlock / drop)
- DONE / WAIT
- Dead-Zero interrupt (protects RHAE)
"""

import random
from typing import Dict, Optional, Tuple, List

# Import L2 and L3 modules
from src.core.program_node import ProgramNode
from src.perception.l3_perceiver import L3Perceiver


class ActionDecider:
    """
    L4 Fuse Gate: Connects L2 ProgramNode → L3 State → env.step(action).
    
    Key upgrade: Receives structured intent from L2, combines with L3 state,
    outputs atomic action for L5.
    
    Features:
    - MOVE_TO_GRID_OBJECT_ID support
    - Dead-Zero interrupt (stuck detection)
    - Manhattan navigation
    """
    
    def __init__(self):
        """Initialize ActionDecider."""
        self.last_pos: Optional[Tuple[int, int]] = None
        self.stuck_count: int = 0
        self.MAX_STUCK: int = 4  # Dead-Zero: terminate after 4 stuck steps
        self.last_intent_type: Optional[str] = None
    
    def decide(self,
               l3_state: Dict,
               program: ProgramNode,
               available_actions: List[str]) -> Optional[str]:
        """
        Decide the next action.
        
        Args:
            l3_state: L3 perception output (player_pos, object_positions, etc.)
            program: L2 ProgramNode (structured intent)
            available_actions: List of available actions from env
        
        Returns:
            Action string for env.step(), or None to terminate
        """
        current_pos = l3_state.get("player_pos")
        
        # ---- L4: Stuck Detection (Dead-Zero) ----
        if current_pos == self.last_pos and l3_state.get("moved") is False:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
        self.last_pos = current_pos
        
        # ---- L2: Get structured intent ----
        intent = program.next_intent(l3_state)
        
        # ---- L4: Termination ----
        if intent.get("done") or self.stuck_count >= self.MAX_STUCK:
            return None
        
        # ---- Resolve to atomic action ----
        action = self._resolve_intent(intent, l3_state, available_actions)
        
        self.last_intent_type = intent.get("type")
        return action
    
    def _resolve_intent(self, intent: Dict, l3_state: Dict, available: List[str]) -> str:
        """
        Resolve L2 intent + L3 state to atomic action.
        
        This is the KEY method: converts structured intent to env.step() action.
        """
        itype = intent.get("type")
        
        # === MOVE: navigate toward target position ===
        if itype == "MOVE":
            target = intent.get("target_pos")
            if target is None:
                return self._random_move(available)
            
            cur = l3_state.get("player_pos")
            if cur == target:
                return "WAIT"  # Arrived
            
            return self._move_toward(cur, target, available)
        
        # === INTERACT: use / pickup / unlock ===
        if itype == "INTERACT":
            # If we just interacted, wait a tick
            if l3_state.get("interacted"):
                return "WAIT"
            if "A" in available:
                return "A"
            return "WAIT"
        
        # === WAIT ===
        if itype == "WAIT":
            return "WAIT"
        
        # === DONE ===
        if itype == "DONE":
            return None
        
        # === Fallback ===
        return self._random_move(available)
    
    def _move_toward(self, cur, target, available: List[str]) -> str:
        """
        Simple Manhattan navigation.
        
        Priority: vertical movement first (can be configured).
        """
        if cur is None or target is None:
            return self._random_move(available)
        
        dr = target[0] - cur[0]
        dc = target[1] - cur[1]
        
        # Priority: larger distance first
        if abs(dr) >= abs(dc):
            if dr > 0 and "DOWN" in available:
                return "DOWN"
            if dr < 0 and "UP" in available:
                return "UP"
        else:
            if dc > 0 and "RIGHT" in available:
                return "RIGHT"
            if dc < 0 and "LEFT" in available:
                return "LEFT"
        
        # If can't move in preferred direction, try the other
        if abs(dr) < abs(dc):
            if dr > 0 and "DOWN" in available:
                return "DOWN"
            if dr < 0 and "UP" in available:
                return "UP"
        else:
            if dc > 0 and "RIGHT" in available:
                return "RIGHT"
            if dc < 0 and "LEFT" in available:
                return "LEFT"
        
        return "WAIT"
    
    def _random_move(self, available: List[str]) -> str:
        """Fallback: random valid move."""
        moves = [a for a in available if a in ("UP", "DOWN", "LEFT", "RIGHT")]
        return random.choice(moves) if moves else "WAIT"
    
    def reset(self):
        """Reset stuck detection (for new episode)."""
        self.last_pos = None
        self.stuck_count = 0
        self.last_intent_type = None
