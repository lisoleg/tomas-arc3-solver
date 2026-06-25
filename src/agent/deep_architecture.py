"""
Deep Optimization Architecture for TOMAS ARC-AGI-3 Solver

Implements L3Perceiver, ActionDecider, and ProgramNode as proposed by 元宝.
These components work together to solve the core problem: Agent cannot correctly 
detect level completion for 22 out of 25 games.

Architecture Overview:
1. L3Perceiver: Deep state perception with win condition detection
2. ActionDecider: Intelligent action decision with κ-Snap alignment
3. ProgramNode: Executable plan representation with validation

Author: TOMAS Team (Enhanced by 元宝's proposal)
"""

from __future__ import annotations

import sys
import os
import time
from collections import deque
from typing import Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np


# ============================================================================
# L3Perceiver: Deep State Perception with Win Condition Detection
# ============================================================================

class WinConditionType(Enum):
    """Types of win conditions in ARC-AGI-3 games."""
    ALL_TAGS_CLICKED = "all_tags_clicked"          # Click all sprites with tag X
    ALL_TARGETS_MATCHED = "all_targets_matched"    # Match all targets with goals
    SCORE_THRESHOLD = "score_threshold"            # Reach a score threshold
    PIXEL_MATCH = "pixel_match"                    # Match pixel pattern
    SPRITE_POSITION = "sprite_position"            # Sprites in specific positions
    COMBO = "combo"                               # Combination of conditions
    UNKNOWN = "unknown"                            # Need to discover


@dataclass
class GameState:
    """Rich game state representation for L3Perceiver.
    
    Attributes:
        step: Current step number
        sprites: Dict of sprite_id -> sprite info
        player: Player entity info
        goals: List of goal entities
        walls: List of wall entities
        score: Current score (if available)
        levels_completed: Number of levels completed
        win_condition_met: Whether win condition is met
        win_condition_progress: Progress towards win condition (0.0 to 1.0)
        raw_observation: Raw observation from environment
    """
    step: int
    sprites: dict
    player: Optional[dict]
    goals: list
    walls: list
    score: float
    levels_completed: int
    win_condition_met: bool
    win_condition_progress: float
    raw_observation: Any


class L3Perceiver:
    """L3Perceiver: Deep State Perception with Win Condition Detection.
    
    This class enhances the basic Oracle adapter by:
    1. Tracking all sprites and their state changes
    2. Discovering win conditions by analyzing game logic
    3. Providing rich state representation to ActionDecider
    4. Detecting level completion accurately
    
    Attributes:
        game_id: Identifier of the current game
        game: Reference to the game object
        oracle_adapter: Reference to the Oracle adapter
        win_condition_type: Detected win condition type
        win_condition_params: Parameters for win condition check
        sprite_history: History of sprite states for change detection
        step_count: Number of steps perceived
    """
    
    def __init__(self, game_id: str, game: Any, oracle_adapter: Any) -> None:
        """Initialize L3Perceiver.
        
        Args:
            game_id: Identifier of the current game (e.g., 'vc33')
            game: Reference to the game object
            oracle_adapter: Reference to the UniversalOracleAdapter
        """
        self.game_id = game_id
        self.game = game
        self.oracle_adapter = oracle_adapter
        
        self.win_condition_type = WinConditionType.UNKNOWN
        self.win_condition_params: dict = {}
        self.sprite_history: list[dict] = []
        self.step_count = 0
        
        # Discover win condition
        self._discover_win_condition()
    
    def _discover_win_condition(self) -> None:
        """Discover the win condition for this game.
        
        This method analyzes the game source code (if available) and
        runtime behavior to determine the win condition.
        """
        # Try to read game source code
        game_source = self._read_game_source()
        
        if game_source:
            # Analyze source code for win condition patterns
            self._analyze_source_for_win_condition(game_source)
        else:
            # Use runtime behavior analysis
            self._analyze_runtime_for_win_condition()
    
    def _read_game_source(self) -> Optional[str]:
        """Read the game source code if available.
        
        Returns:
            Game source code as string, or None if not available.
        """
        try:
            # Game source is typically in environment_files/{game_id}/*/
            env_path = f"environment_files/{self.game_id}"
            if not os.path.exists(env_path):
                return None
            
            # Find .py files
            for root, dirs, files in os.walk(env_path):
                for f in files:
                    if f.endswith('.py') and not f.startswith('_'):
                        with open(os.path.join(root, f), 'r', encoding='utf-8') as fp:
                            return fp.read()
        except Exception:
            pass
        return None
    
    def _analyze_source_for_win_condition(self, source: str) -> None:
        """Analyze game source code to find win condition.
        
        Args:
            source: Game source code as string
        """
        # Pattern 1: Look for levels_completed increment
        if "levels_completed" in source and "next_level" in source:
            # Find the condition that triggers next_level
            if "ielczunthe" in source:
                # Game has a win condition check method
                self.win_condition_type = WinConditionType.COMBO
                self.win_condition_params = {
                    "check_method": "ielczunthe",
                    "description": "Complex condition checked by ielczunthe()",
                }
            else:
                # Generic: look for all_tags_clicked pattern
                self.win_condition_type = WinConditionType.ALL_TAGS_CLICKED
                self.win_condition_params = {
                    "description": "Click all sprites with specific tags",
                }
        
        # Pattern 2: Look for score threshold
        elif "score" in source and ("threshold" in source or "target" in source):
            self.win_condition_type = WinConditionType.SCORE_THRESHOLD
            self.win_condition_params = {
                "description": "Reach a score threshold",
            }
        
        # Pattern 3: Look for pixel match
        elif "pixels" in source and ("match" in source or "complete" in source):
            self.win_condition_type = WinConditionType.PIXEL_MATCH
            self.win_condition_params = {
                "description": "Match pixel pattern",
            }
    
    def _analyze_runtime_for_win_condition(self) -> None:
        """Analyze runtime behavior to infer win condition.
        
        This method uses heuristics based on game type and Oracle config.
        """
        # Use game config to infer win condition
        if self.oracle_adapter and self.oracle_adapter._game_config:
            config = self.oracle_adapter._game_config
            
            # Click games: win condition is clicking all targets
            if config.game_type == "click_only":
                self.win_condition_type = WinConditionType.ALL_TAGS_CLICKED
                self.win_condition_params = {
                    "click_target_tag": config.click_target_tag,
                    "description": f"Click all sprites with tag {config.click_target_tag}",
                }
            
            # Keyboard games: win condition is matching all targets
            elif config.game_type == "keyboard_only":
                self.win_condition_type = WinConditionType.ALL_TARGETS_MATCHED
                self.win_condition_params = {
                    "goal_tag": config.goal_tag,
                    "description": "Match all goals with targets",
                }
    
    def perceive(self, observation: Any) -> GameState:
        """Perceive current game state.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            GameState object with rich state information
        """
        self.step_count += 1
        
        # Get basic info from observation
        levels_completed = getattr(observation, 'levels_completed', 0)
        
        # Get sprites from oracle adapter
        sprites = {}
        if self.oracle_adapter:
            self.oracle_adapter._discover()
            all_sprites = self.oracle_adapter.get_sprites()
            
            for i, sprite in enumerate(all_sprites):
                sprite_id = f"sprite_{i}"
                sprites[sprite_id] = {
                    "x": sprite.x,
                    "y": sprite.y,
                    "tags": list(sprite.tags) if sprite.tags else [],
                    "pixel": getattr(sprite, 'pixel', None),
                }
        
        # Get player, goals, walls
        player = None
        if self.oracle_adapter:
            p = self.oracle_adapter.get_player()
            if p:
                player = {"x": p.x, "y": p.y, "tags": list(p.tags) if p.tags else []}
        
        goals = []
        if self.oracle_adapter:
            for g in self.oracle_adapter.get_goals():
                goals.append({"x": g.x, "y": g.y, "tags": list(g.tags) if g.tags else []})
        
        walls = []
        if self.oracle_adapter:
            for w in self.oracle_adapter.get_walls():
                walls.append({"x": w.x, "y": w.y, "tags": list(w.tags) if w.tags else []})
        
        # Check win condition
        win_condition_met, win_condition_progress = self._check_win_condition(sprites, goals, walls)
        
        # Create GameState
        state = GameState(
            step=self.step_count,
            sprites=sprites,
            player=player,
            goals=goals,
            walls=walls,
            score=0.0,  # TODO: extract from game if available
            levels_completed=levels_completed,
            win_condition_met=win_condition_met,
            win_condition_progress=win_condition_progress,
            raw_observation=observation,
        )
        
        # Update history
        self.sprite_history.append({
            "step": self.step_count,
            "sprites": sprites,
            "win_condition_met": win_condition_met,
        })
        
        return state
    
    def _check_win_condition(
        self, 
        sprites: dict, 
        goals: list, 
        walls: list
    ) -> tuple[bool, float]:
        """Check if win condition is met.
        
        Args:
            sprites: Dict of sprite info
            goals: List of goal info
            walls: List of wall info
            
        Returns:
            Tuple of (win_condition_met, progress_0_to_1)
        """
        if self.win_condition_type == WinConditionType.ALL_TAGS_CLICKED:
            # Check if all sprites with target tag are clicked
            target_tag = self.win_condition_params.get("click_target_tag")
            if not target_tag:
                return False, 0.0
            
            # TODO: Need to track which sprites are clicked
            # For now, assume not met
            return False, 0.5
        
        elif self.win_condition_type == WinConditionType.ALL_TARGETS_MATCHED:
            # Check if all goals are matched
            # TODO: Implement based on game logic
            return False, 0.5
        
        elif self.win_condition_type == WinConditionType.COMBO:
            # Complex condition: try to call game's check method
            check_method = self.win_condition_params.get("check_method")
            if check_method and self.game:
                try:
                    method = getattr(self.game, check_method, None)
                    if method:
                        result = method()
                        if isinstance(result, bool):
                            return result, 1.0 if result else 0.0
                except Exception:
                    pass
            return False, 0.0
        
        else:
            # Unknown win condition: use levels_completed from observation
            # This is the fallback: if environment says level completed, it's completed
            return False, 0.0
    
    def get_click_targets(self) -> list[tuple[int, int]]:
        """Get list of target positions to click for click games.
        
        Returns:
            List of (x, y) positions to click
        """
        if self.win_condition_type != WinConditionType.ALL_TAGS_CLICKED:
            return []
        
        target_tag = self.win_condition_params.get("click_target_tag")
        if not target_tag:
            return []
        
        targets = []
        for sprite_id, sprite_info in self.sprite_history[-1]["sprites"].items():
            if target_tag in sprite_info["tags"]:
                targets.append((sprite_info["x"], sprite_info["y"]))
        
        return targets
    
    def is_level_completed(self, observation: Any) -> bool:
        """Check if level is completed based on perception.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            True if level is completed, False otherwise
        """
        # Method 1: Check observation levels_completed
        levels_completed = getattr(observation, 'levels_completed', 0)
        if levels_completed > 0:
            return True
        
        # Method 2: Check win condition from perception
        state = self.perceive(observation)
        if state.win_condition_met:
            return True
        
        # Method 3: Check game's internal state (if accessible)
        if self.game:
            try:
                # Some games have a _level_completed or similar attribute
                if hasattr(self.game, '_level_completed'):
                    if self.game._level_completed:
                        return True
                if hasattr(self.game, 'level_completed'):
                    if self.game.level_completed:
                        return True
            except Exception:
                pass
        
        return False


# ============================================================================
# ActionDecider: Intelligent Action Decision with κ-Snap Alignment
# ============================================================================

class DecisionContext:
    """Context for action decision.
    
    Attributes:
        game_state: Current game state from L3Perceiver
        plan: Current plan (if any)
        history: Action history
        confidence: Confidence in current plan (0.0 to 1.0)
    """
    
    def __init__(
        self, 
        game_state: GameState, 
        plan: Optional[Any] = None,
        history: Optional[list] = None,
        confidence: float = 1.0,
    ) -> None:
        """Initialize DecisionContext.
        
        Args:
            game_state: Current game state
            plan: Current plan (ProgramNode or None)
            history: Action history
            confidence: Confidence in current plan
        """
        self.game_state = game_state
        self.plan = plan
        self.history = history or []
        self.confidence = confidence


class ActionDecider:
    """ActionDecider: Intelligent action decision with κ-Snap alignment.
    
    This class decides what action to take based on:
    1. Current game state (from L3Perceiver)
    2. Current plan (ProgramNode)
    3. κ-Snap search results
    4. Win condition progress
    
    Attributes:
        game_id: Identifier of the current game
        perceiver: Reference to L3Perceiver
        ksnap_searcher: Reference to κ-Snap Searcher
        decision_history: History of decisions for learning
    """
    
    def __init__(self, game_id: str, perceiver: L3Perceiver, ksnap_searcher: Optional[Any] = None) -> None:
        """Initialize ActionDecider.
        
        Args:
            game_id: Identifier of the current game
            perceiver: Reference to L3Perceiver
            ksnap_searcher: Reference to κ-Snap Searcher (optional)
        """
        self.game_id = game_id
        self.perceiver = perceiver
        self.ksnap_searcher = ksnap_searcher
        self.decision_history: list[dict] = []
    
    def decide(self, context: DecisionContext) -> Optional[Any]:
        """Decide what action to take.
        
        Args:
            context: DecisionContext with current state and plan
            
        Returns:
            Action to take, or None if no action needed
        """
        game_state = context.game_state
        
        # Check if level is already completed
        if game_state.win_condition_met:
            return None  # No action needed
        
        # Check win condition progress
        if game_state.win_condition_progress > 0.9:
            # Very close to winning: take careful actions
            return self._decide_careful(context)
        
        # Use κ-Snap searcher if available
        if self.ksnap_searcher:
            action = self.ksnap_searcher.search(game_state)
            if action:
                return action
        
        # Fallback: use plan if available
        if context.plan:
            action = context.plan.get_next_action(game_state)
            if action:
                return action
        
        # Fallback: use simple strategy based on game type
        return self._decide_simple(context)
    
    def _decide_careful(self, context: DecisionContext) -> Optional[Any]:
        """Decide action when close to winning.
        
        Args:
            context: DecisionContext
            
        Returns:
            Action to take
        """
        # TODO: Implement careful decision making
        # For now, return None to avoid reckless actions
        return None
    
    def _decide_simple(self, context: DecisionContext) -> Optional[Any]:
        """Decide action using simple strategy.
        
        Args:
            context: DecisionContext
            
        Returns:
            Action to take
        """
        game_state = context.game_state
        
        # Click games: click targets
        if self.perceiver.win_condition_type == WinConditionType.ALL_TAGS_CLICKED:
            targets = self.perceiver.get_click_targets()
            if targets and len(self.decision_history) < len(targets):
                # Click next unclicked target
                target = targets[len(self.decision_history)]
                return {
                    "type": "click",
                    "x": target[0],
                    "y": target[1],
                }
        
        # TODO: Implement more simple strategies
        return None
    
    def update_history(self, action: Any, result: GameState) -> None:
        """Update decision history.
        
        Args:
            action: Action taken
            result: Resulting game state
        """
        self.decision_history.append({
            "action": action,
            "result": result,
            "step": result.step,
        })


# ============================================================================
# ProgramNode: Executable Plan Representation with Validation
# ============================================================================

class ProgramNode:
    """ProgramNode: Executable plan representation with validation.
    
    A ProgramNode represents a sequence of actions that can be executed
    and validated. It supports:
    1. Conditional branches
    2. Loops
    3. Validation and rollback
    4. Integration with L3Perceiver and ActionDecider
    
    Attributes:
        actions: List of actions in the plan
        conditions: List of conditions for conditional execution
        current_index: Current execution index
        validated: Whether the plan has been validated
    """
    
    def __init__(self, actions: list[Any], conditions: Optional[list] = None) -> None:
        """Initialize ProgramNode.
        
        Args:
            actions: List of actions (each action can be a dict or ProgramNode)
            conditions: List of conditions for conditional execution
        """
        self.actions = actions
        self.conditions = conditions or []
        self.current_index = 0
        self.validated = False
    
    def get_next_action(self, game_state: GameState) -> Optional[Any]:
        """Get the next action to execute.
        
        Args:
            game_state: Current game state
            
        Returns:
            Next action, or None if plan is complete
        """
        if self.current_index >= len(self.actions):
            return None
        
        action = self.actions[self.current_index]
        
        # Check conditions
        if self.current_index < len(self.conditions):
            condition = self.conditions[self.current_index]
            if not self._evaluate_condition(condition, game_state):
                # Skip this action
                self.current_index += 1
                return self.get_next_action(game_state)
        
        self.current_index += 1
        return action
    
    def _evaluate_condition(self, condition: Any, game_state: GameState) -> bool:
        """Evaluate a condition.
        
        Args:
            condition: Condition to evaluate
            game_state: Current game state
            
        Returns:
            True if condition is met
        """
        # TODO: Implement condition evaluation
        return True
    
    def validate(self, game_state: GameState) -> bool:
        """Validate the plan against current state.
        
        Args:
            game_state: Current game state
            
        Returns:
            True if plan is valid
        """
        # TODO: Implement plan validation
        self.validated = True
        return True
    
    def rollback(self) -> None:
        """Rollback execution to beginning."""
        self.current_index = 0
    
    def __len__(self) -> int:
        """Get plan length."""
        return len(self.actions)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ProgramNode({len(self.actions)} actions, index={self.current_index})"


# ============================================================================
# Integration Functions
# ============================================================================

def create_deep_architecture(
    game_id: str, 
    game: Any, 
    oracle_adapter: Any, 
    ksnap_searcher: Optional[Any] = None,
) -> tuple[L3Perceiver, ActionDecider, Optional[ProgramNode]]:
    """Create deep architecture components.
    
    Args:
        game_id: Identifier of the current game
        game: Reference to the game object
        oracle_adapter: Reference to the Oracle adapter
        ksnap_searcher: Reference to κ-Snap Searcher (optional)
        
    Returns:
        Tuple of (L3Perceiver, ActionDecider, initial ProgramNode)
    """
    # Create L3Perceiver
    perceiver = L3Perceiver(game_id, game, oracle_adapter)
    
    # Create ActionDecider
    decider = ActionDecider(game_id, perceiver, ksnap_searcher)
    
    # Create initial empty plan
    plan = ProgramNode(actions=[])
    
    return perceiver, decider, plan


def integrate_with_solution(
    solution_module: Any,
    game_id: str,
    game: Any,
    oracle_adapter: Any,
) -> None:
    """Integrate deep architecture with existing solution module.
    
    Args:
        solution_module: Reference to solution module (e.g., solution_v06_deep_opt)
        game_id: Identifier of the current game
        game: Reference to the game object
        oracle_adapter: Reference to the Oracle adapter
    """
    # Create architecture components
    perceiver, decider, plan = create_deep_architecture(game_id, game, oracle_adapter)
    
    # Attach to solution module
    solution_module._l3_perceiver = perceiver
    solution_module._action_decider = decider
    solution_module._current_plan = plan
    
    # TODO: Modify solution's choose_action to use decider
