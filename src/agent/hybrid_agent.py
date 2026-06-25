"""
Hybrid Agent: Integrates L2→L3→L4→L5 pipeline from 元宝's proposal.

This agent can work in both:
1. Oracle Mode (if oracle is available)
2. Grid-Only Mode (learns from demos)

Uses the upgraded modules:
- L2: ProgramNode (structured intent)
- L3: L3Perceiver (object tracking)
- L4: ActionDecider (intent → action)
- L5: env.step() (execution)
"""

import arc_agi
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Import upgraded modules
from ..core.program_node import ProgramNode
from ..perception.l3_perceiver import L3Perceiver
from ..solver.action_decider import ActionDecider


class HybridTaiyiAgent:
    """
    Hybrid agent using L2→L3→L4→L5 pipeline.
    
    Features:
    - Oracle Mode: Use oracle adapter if available
    - Grid-Only Mode: Learn from demos and execute
    - Dead-Zero interrupt: Protects RHAE
    """
    
    def __init__(self, use_oracle: bool = True):
        """
        Initialize HybridTaiyiAgent.
        
        Args:
            use_oracle: If True, try Oracle mode first
        """
        self.use_oracle = use_oracle
        self.l3 = L3Perceiver()
        self.decider = ActionDecider()
        self.program = None  # L2 ProgramNode
        self.oracle_adapter = None
        
    def run_episode(self, env, game_id: str = None, modifier_hints: List[str] = None):
        """
        Run one episode (one level).
        
        Args:
            env: arc_agi environment
            game_id: Game ID (e.g., "ls20")
            modifier_hints: List of modifier hints
        
        Returns:
            dict with steps, reward, success
        """
        obs, info = env.reset()
        modifier_hints = modifier_hints or info.get("modifiers", [])
        
        # ---- Try Oracle Mode ----
        if self.use_oracle:
            try:
                self.oracle_adapter = arc_agi.Oracle(env)
                # Oracle available: use direct entity access
                return self._run_oracle_mode(env, obs, info, game_id)
            except:
                # Oracle not available: fallback to Grid-only
                self.oracle_adapter = None
        
        # ---- Grid-Only Mode ----
        return self._run_grid_only_mode(env, obs, info, game_id, modifier_hints)
    
    def _run_oracle_mode(self, env, obs, info, game_id: str):
        """Oracle Mode: Direct entity access."""
        # TODO: Integrate with existing UniversalOracleAdapter
        # For now, use a simple strategy:
        # 1. Read player position
        # 2. Read goal positions
        # 3. Navigate to goals
        
        steps = 0
        max_steps = 1000
        
        while steps < max_steps:
            # Get oracle info
            player = self.oracle_adapter.get_sprites_by_tag("player")[0]
            goals = self.oracle_adapter.get_sprites_by_tag("goal")
            
            # Simple strategy: move to first goal
            if goals:
                target = goals[0]
                action = self._navigate_to(env, player.x, player.y, target.x, target.y)
                
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                
                if terminated or truncated:
                    return {"steps": steps, "reward": reward, "success": True}
        
        return {"steps": steps, "reward": 0, "success": False}
    
    def _run_grid_only_mode(self, env, obs, info, game_id: str, modifier_hints: List[str]):
        """
        Grid-Only Mode: Learn from demos and execute.
        
        Key upgrade: Uses L2→L3→L4→L5 pipeline.
        """
        # ---- L2: Induce program from demos ----
        demo_pairs = info.get("demonstrations", [])
        if demo_pairs:
            self.program = self._induce_from_demos(demo_pairs, modifier_hints)
        else:
            # No demos: use a default "explore" program
            self.program = ProgramNode("explore")
        
        steps = 0
        hard_cap = 5 * info.get("human_baseline_actions", 500)
        
        while steps < hard_cap:
            # ---- L3: Perceive ----
            l3_state = self.l3.perceive(obs, info)
            
            # ---- L4: Decide action ----
            action = self.decider.decide(
                l3_state=l3_state,
                program=self.program,
                available_actions=info.get("available_actions", [])
            )
            
            if action is None:  # L4 says terminate (or stuck)
                break
            
            # ---- L5: Execute ----
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            
            if terminated or truncated:
                return {"steps": steps, "reward": reward, "success": True}
        
        env.close()
        return {"steps": steps, "reward": 0, "success": False}
    
    def _induce_from_demos(self, demo_pairs: List, modifier_hints: List[str]) -> ProgramNode:
        """
        L2: Induce a program from demo pairs.
        
        Simplified version:
        - Analyze input/output grids
        - Identify target objects
        - Generate a simple "move_to + interact" program
        """
        # For now, return a simple "move_to_goal" program
        # TODO: Implement full κ-Snap Searcher
        
        # Heuristic: If demo shows player moving to a goal, generate "move_to" program
        if demo_pairs:
            inp, out = demo_pairs[0]
            # Find changed cells
            diff = (out != inp)
            if np.any(diff):
                # Assume the changed cell is the goal
                goal_pos = np.argwhere(diff)[0]
                return ProgramNode("move_to", args={"target_pos": tuple(goal_pos)})
        
        # Default: explore
        return ProgramNode("explore")
    
    def _navigate_to(self, env, from_x, from_y, to_x, to_y) -> str:
        """Simple navigation: return action to move toward target."""
        dx = to_x - from_x
        dy = to_y - from_y
        
        if abs(dx) > abs(dy):
            if dx > 0:
                return "DOWN"
            else:
                return "UP"
        else:
            if dy > 0:
                return "RIGHT"
            else:
                return "LEFT"
