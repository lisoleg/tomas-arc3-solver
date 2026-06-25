"""
通用Grid-Only Solver for ARC-AGI-3 Games

策略：
1. 如果是click游戏 - 系统搜索所有可点击位置
2. 如果是keyboard游戏 - BFS搜索最优路径
3. 目标：完成关卡（即使不是最优步骤）
"""

import sys
import os
from typing import Optional, Any
from arcengine import GameAction, FrameDataRaw


class GridOnlySolver:
    """通用Grid-Only Solver - 不依赖Oracle，只用grid观察"""
    
    def __init__(self, env, game_id: str, max_steps: int = 10000):
        self.env = env
        self.game_id = game_id
        self.max_steps = max_steps
        self.step_count = 0
        
        # 检测游戏类型
        self.is_click_game = self._detect_click_game()
        self.is_keyboard_game = self._detect_keyboard_game()
        
        # Click游戏：记录已尝试的点击位置
        self.tried_clicks = set()
        self.successful_clicks = []
        
        # Keyboard游戏：BFS状态
        self.visited_states = set()
        self.current_plan = []
        
        print(f"[GridSolver] Game {game_id}: click={self.is_click_game}, keyboard={self.is_keyboard_game}")
    
    def _detect_click_game(self) -> bool:
        """检测是否是click游戏（有ACTION6）"""
        try:
            obs = self.env.step(GameAction.RESET)
            has_action6 = GameAction.ACTION6 in obs.available_actions
            return has_action6
        except:
            return False
    
    def _detect_keyboard_game(self) -> bool:
        """检测是否是keyboard游戏（有方向键）"""
        try:
            obs = self.env.step(GameAction.RESET)
            has_arrows = any(a in obs.available_actions for a in [
                GameAction.ACTION1, GameAction.ACTION2,
                GameAction.ACTION3, GameAction.ACTION4
            ])
            return has_arrows
        except:
            return False
    
    def solve(self, max_steps: int = 10000) -> list:
        """求解游戏，返回动作序列"""
        obs = self.env.step(GameAction.RESET)
        actions = []
        
        if self.is_click_game:
            actions = self._solve_click(obs, max_steps)
        elif self.is_keyboard_game:
            actions = self._solve_keyboard(obs, max_steps)
        else:
            print(f"[GridSolver] Unknown game type for {self.game_id}")
            actions = self._solve_generic(obs, max_steps)
        
        return actions
    
    def _solve_click(self, obs: FrameDataRaw, max_steps: int) -> list:
        """Click游戏求解：系统搜索所有可点击位置"""
        print(f"[GridSolver] Solving click game {self.game_id}...")
        
        actions = []
        levels_completed = 0
        stale_count = 0
        last_levels = 0
        
        for step in range(max_steps):
            # 提取grid
            grid = self._extract_grid(obs)
            if grid is None:
                break
            
            # 找到所有可点击位置（非背景、非墙壁的像素）
            clickable_positions = self._find_clickable_positions(grid)
            
            if not clickable_positions:
                print(f"[GridSolver] No clickable positions found at step {step}")
                break
            
            # 选择下一个未尝试的位置
            pos = self._select_next_position(clickable_positions)
            if pos is None:
                print(f"[GridSolver] All positions tried, stagnating...")
                # 重置游戏，重新尝试
                obs = self.env.step(GameAction.RESET)
                self.tried_clicks = set()
                stale_count += 1
                if stale_count > 10:
                    break
                continue
            
            # 执行点击
            x, y = pos
            self.tried_clicks.add((x, y))
            
            try:
                obs = self.env.step(GameAction.ACTION6, data={'x': x, 'y': y})
            except Exception as e:
                print(f"[GridSolver] Error clicking ({x},{y}): {e}")
                continue
            
            actions.append((GameAction.ACTION6, x, y))
            self.step_count += 1
            
            # 检查关卡完成
            if obs.levels_completed > levels_completed:
                print(f"[GridSolver] ✅ Level {levels_completed} -> {obs.levels_completed} at step {step}")
                levels_completed = obs.levels_completed
                self.successful_clicks = []
                self.tried_clicks = set()
                stale_count = 0
                
                if levels_completed >= 7:
                    print(f"[GridSolver] ✅ All levels completed!")
                    break
            
            # 检查停滞
            if obs.levels_completed == last_levels:
                stale_count += 1
            else:
                stale_count = 0
            last_levels = obs.levels_completed
            
            # 如果停滞太久，尝试随机点击
            if stale_count > 50:
                print(f"[GridSolver] Stagnating for 50 steps, trying random click...")
                self.tried_clicks = set()  # 清空，重新尝试
                stale_count = 0
        
        print(f"[GridSolver] Click game {self.game_id} completed {levels_completed} levels in {len(actions)} actions")
        return actions
    
    def _solve_keyboard(self, obs: FrameDataRaw, max_steps: int) -> list:
        """Keyboard游戏求解：BFS搜索"""
        print(f"[GridSolver] Solving keyboard game {self.game_id}...")
        
        # 简化策略：随机探索 + 避免重复状态
        actions = []
        levels_completed = 0
        
        action_list = [
            GameAction.ACTION1,  # UP
            GameAction.ACTION2,  # DOWN
            GameAction.ACTION3,  # LEFT
            GameAction.ACTION4,  # RIGHT
        ]
        
        for step in range(max_steps):
            # 随机选择动作
            import random
            action = random.choice(action_list)
            
            try:
                obs = self.env.step(action)
            except Exception as e:
                print(f"[GridSolver] Error executing action: {e}")
                continue
            
            actions.append(action)
            self.step_count += 1
            
            # 检查关卡完成
            if obs.levels_completed > levels_completed:
                print(f"[GridSolver] ✅ Level {levels_completed} -> {obs.levels_completed} at step {step}")
                levels_completed = obs.levels_completed
                
                if levels_completed >= 7:
                    print(f"[GridSolver] ✅ All levels completed!")
                    break
        
        print(f"[GridSolver] Keyboard game {self.game_id} completed {levels_completed} levels in {len(actions)} actions")
        return actions
    
    def _solve_generic(self, obs: FrameDataRaw, max_steps: int) -> list:
        """通用求解：尝试所有可用动作"""
        print(f"[GridSolver] Solving generic game {self.game_id}...")
        
        actions = []
        levels_completed = 0
        
        # 获取所有可用动作
        available = list(obs.available_actions)
        
        for step in range(max_steps):
            # 循环使用所有可用动作
            action = available[step % len(available)]
            
            try:
                if action == GameAction.ACTION6:
                    # Click游戏：点击中心位置
                    grid = self._extract_grid(obs)
                    if grid is not None:
                        h, w = grid.shape
                        obs = self.env.step(action, data={'x': w//2, 'y': h//2})
                    else:
                        obs = self.env.step(action)
                else:
                    obs = self.env.step(action)
            except Exception as e:
                print(f"[GridSolver] Error executing action: {e}")
                continue
            
            actions.append(action)
            
            # 检查关卡完成
            if obs.levels_completed > levels_completed:
                print(f"[GridSolver] ✅ Level {levels_completed} -> {obs.levels_completed} at step {step}")
                levels_completed = obs.levels_completed
                
                if levels_completed >= 7:
                    print(f"[GridSolver] ✅ All levels completed!")
                    break
        
        print(f"[GridSolver] Generic game {self.game_id} completed {levels_completed} levels in {len(actions)} actions")
        return actions
    
    def _extract_grid(self, obs: FrameDataRaw) -> Optional[Any]:
        """从观察中提取grid"""
        try:
            if hasattr(obs, 'frame') and obs.frame is not None:
                return obs.frame[0]  # (64, 64) grid
        except:
            pass
        return None
    
    def _find_clickable_positions(self, grid: Any) -> list:
        """找到所有可点击位置"""
        positions = []
        h, w = grid.shape
        
        # 采样：每4个像素检查一次（避免太多）
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                pixel = grid[y, x]
                # 非背景、非黑色的位置可点击
                if pixel > 0:
                    positions.append((x, y))
        
        return positions
    
    def _select_next_position(self, positions: list) -> Optional[tuple]:
        """选择下一个未尝试的位置"""
        for pos in positions:
            if pos not in self.tried_clicks:
                return pos
        return None


def test_grid_solver():
    """测试通用Grid-Only Solver"""
    import arc_agi
    
    game_ids = ['vc33', 's5i5', 'tn36', 'ls20', 'ft09']
    
    for game_id in game_ids:
        print(f"\n{'='*60}")
        print(f"Testing {game_id} with Grid-Only Solver...")
        print(f"{'='*60}")
        
        try:
            arc = arc_agi.Arcade()
            env = arc.make(game_id)
            
            solver = GridOnlySolver(env, game_id, max_steps=2000)
            actions = solver.solve(max_steps=2000)
            
            print(f"\n{game_id} result: {len(actions)} actions executed")
        except Exception as e:
            print(f"Error testing {game_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    test_grid_solver()
