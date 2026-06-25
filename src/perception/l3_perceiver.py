"""
L3 Differential Perception.

Upgraded: tracks all grid objects (not just player),
supports MOVE_TO_GRID_OBJECT_ID resolution.

This module connects L2 ProgramNode to real-time game state.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class L3Perceiver:
    """
    L3 Differential Perception.
    
    Upgraded: tracks all grid objects (not just player),
    supports MOVE_TO_GRID_OBJECT_ID resolution.
    
    Key upgrade: object_positions dictionary for runtime object tracking.
    """
    
    def __init__(self):
        """Initialize L3Perceiver."""
        self.prev_obs: Optional[np.ndarray] = None
        self.player_pos: Optional[Tuple[int, int]] = None
        self.object_positions: Dict[int, Tuple[int, int]] = {}  # {color_id: (r, c)}
        self.last_interaction: Optional[Dict] = None
    
    def perceive(self, obs: np.ndarray, info: Dict) -> Dict:
        """
        Perceive the current game state.
        
        Returns enhanced L3 state with object tracking.
        
        Returns:
            {
                "event": "init" | "step",
                "player_pos": (r, c) | None,
                "moved": bool,
                "interacted": {"type": "pickup"|"drop", "obj_id": int, "pos": (r,c)} | None,
                "object_positions": {obj_id: (r, c)},
                "changed_cells": [(r, c), ...],
                "grid_shape": (H, W),
            }
        """
        if self.prev_obs is None:
            # First frame: initialize
            self.prev_obs = obs.copy()
            self.player_pos = self._find_player(obs)
            self._update_object_positions(obs)
            
            return {
                "event": "init",
                "player_pos": self.player_pos,
                "object_positions": dict(self.object_positions),
                "grid_shape": obs.shape,
            }
        
        # ---- Core L3: Delta ----
        diff_mask = (obs != self.prev_obs)
        changed_cells = list(map(tuple, np.argwhere(diff_mask)))
        
        # Player
        new_player = self._find_player(obs)
        moved = new_player != self.player_pos
        
        # Objects
        self._update_object_positions(obs)
        interaction = self._detect_interaction(self.prev_obs, obs, changed_cells)
        
        # Update state
        self.prev_obs = obs.copy()
        self.player_pos = new_player
        
        return {
            "event": "step",
            "player_pos": new_player,
            "moved": moved,
            "interacted": interaction,
            "object_positions": dict(self.object_positions),
            "changed_cells": changed_cells,
            "grid_shape": obs.shape,
        }
    
    def _find_player(self, obs: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find player avatar.
        
        Heuristic: color 8 (standard ARC player color)
        TODO: Make this configurable per game
        """
        coords = np.argwhere(obs == 8)
        if len(coords) > 0:
            return tuple(coords[0])
        return None
    
    def _update_object_positions(self, obs: np.ndarray):
        """
        Scan grid for all non-zero, non-player objects.
        
        Key: map color/ID → position for L2 resolution.
        This enables MOVE_TO_GRID_OBJECT_ID at runtime.
        """
        self.object_positions.clear()
        H, W = obs.shape
        
        for r in range(H):
            for c in range(W):
                val = obs[r, c]
                if val != 0 and val != 8:  # 0=empty, 8=player
                    self.object_positions[val] = (r, c)
    
    def _detect_interaction(self, prev: np.ndarray, curr: np.ndarray, changes: List) -> Optional[Dict]:
        """
        Detect pickups/drops by observing cell state transitions.
        
        Enhanced: tag with object ID.
        """
        for r, c in changes:
            prev_v = prev[r, c]
            curr_v = curr[r, c]
            
            if prev_v != 0 and curr_v == 0:
                # Object disappeared → picked up
                return {"type": "pickup", "obj_id": int(prev_v), "pos": (r, c)}
            
            if prev_v == 0 and curr_v != 0:
                # Object appeared → dropped
                return {"type": "drop", "obj_id": int(curr_v), "pos": (r, c)}
        
        return None
    
    def get_object_position(self, obj_id: int) -> Optional[Tuple[int, int]]:
        """
        Get the current position of an object by ID.
        
        This is the KEY method for MOVE_TO_GRID_OBJECT_ID:
        - L2 specifies target_id=5
        - L4 calls this method to get (r, c)
        - L4 navigates player to that position
        """
        return self.object_positions.get(obj_id)
    
    def reset(self):
        """Reset the perceiver state (for new episode)."""
        self.prev_obs = None
        self.player_pos = None
        self.object_positions.clear()
        self.last_interaction = None
