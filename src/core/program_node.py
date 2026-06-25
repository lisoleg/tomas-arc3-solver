"""
L2 Conduction output: a DSL program with executable semantics.
Upgraded to support grid-object binding for Oracle Mode.

This is the core data structure connecting:
- L2 κ-Snap Searcher (program induction)
- L3 Perceiver (state perception)
- L4 ActionDecider (intent → action)
- L5 Environment (execution)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ProgramNode:
    """
    L2 Conduction output: a DSL program with executable semantics.
    Upgraded to support grid-object binding for Oracle Mode.
    
    Key upgrade: Structured Intent API for L4 ActionDecider.
    """
    
    def __init__(self, dsl_name: str, args: Dict = None, children: List['ProgramNode'] = None):
        """
        Initialize a ProgramNode.
        
        Args:
            dsl_name: DSL operation name (e.g., "move_to", "unlock", "fill", "done")
            args: Arguments for the operation (e.g., {"target_id": 5, "color": 3})
            children: Child nodes for composite programs
        """
        self.dsl_name = dsl_name
        self.args = args or {}
        self.children = children or []
        
        # ---- Oracle Mode fields ----
        # If set, the L3 perception module can resolve this to pixel coords
        self._target_grid_object_id: Optional[int] = self.args.get("target_id")
        self._cached_target_pos: Optional[Tuple[int, int]] = None
    
    # ========== L2: Intent API (used by ActionDecider) ==========
    
    def next_intent(self, l3_state: Dict) -> Dict:
        """
        Returns a structured intent dict for L4 ActionDecider.
        
        Returns:
            {
                "type": "MOVE" | "INTERACT" | "WAIT" | "DONE",
                "target_pos": (r, c) | None,
                "target_id": int | None,       # grid object ID
                "action_hint": "A" | None,     # if interaction needed
                "done": bool,
            }
        
        This is the KEY upgrade: structured intent instead of raw args.
        """
        if self.dsl_name == "move_to":
            return {
                "type": "MOVE",
                "target_pos": self._resolve_position(l3_state),
                "target_id": self._target_grid_object_id,
                "action_hint": None,
                "done": False,
            }
        
        if self.dsl_name in ("unlock", "interact", "pickup", "drop"):
            return {
                "type": "INTERACT",
                "target_pos": self._resolve_position(l3_state),
                "target_id": self._target_grid_object_id,
                "action_hint": "A",  # Default: press A to interact
                "done": False,
            }
        
        if self.dsl_name == "fill":
            # Fill is a "move + interact" composite
            if self.children:
                return self.children[0].next_intent(l3_state)
            return {"type": "WAIT", "done": False}
        
        if self.dsl_name == "done" or self.dsl_name == "terminate":
            return {"type": "DONE", "done": True}
        
        # Default: wait (e.g., for animation to finish)
        return {"type": "WAIT", "done": False}
    
    def _resolve_position(self, l3_state: Dict) -> Optional[Tuple[int, int]]:
        """
        L3-assisted resolution:
        1. Try cache
        2. Ask L3 perception for object location
        3. Fallback: use args['target_pos'] if provided
        
        This enables MOVE_TO_GRID_OBJECT_ID:
        - L2 specifies target_id=5
        - L3 resolves to (r, c) at runtime
        - L4 navigates to that position
        """
        # Check cache first
        if self._cached_target_pos is not None:
            return self._cached_target_pos
        
        # Ask L3: "Where is object X?"
        target_id = self._target_grid_object_id
        if target_id is not None and "object_positions" in l3_state:
            obj_pos = l3_state["object_positions"].get(target_id)
            if obj_pos is not None:
                self._cached_target_pos = tuple(obj_pos)
                return self._cached_target_pos
        
        # Fallback: static target from args
        if "target_pos" in self.args:
            return tuple(self.args["target_pos"])
        
        return None
    
    # ========== L2: Execution API (used by Grid-Only Mode) ==========
    
    def apply(self, grid: np.ndarray, l3_state: Dict = None) -> np.ndarray:
        """
        Execute on a grid (Grid-Only Mode).
        
        This is the original execution method for Grid-only mode.
        Oracle mode uses next_intent() instead.
        """
        # Placeholder - implement based on specific DSL operations
        # For now, return grid unchanged
        return grid
    
    def is_done(self) -> bool:
        """Check if this program is a termination command."""
        return self.dsl_name in ("done", "terminate")
    
    def __repr__(self):
        return f"ProgramNode({self.dsl_name}, args={self.args})"
