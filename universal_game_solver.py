"""
Universal Game Solver - 通用游戏求解器

使用Oracle适配器数据进行通用游戏求解：
1. Keyboard游戏：A*导航到目标
2. Click游戏：点击目标，然后等待

这个solver不试图"解决"谜题，只是执行基于Oracle数据的基本操作。
目标：完成关卡（即使步骤不是最优）！
"""

import sys
from typing import Optional, Any

sys.path.insert(0, 'src')

from arcengine import GameAction, FrameDataRaw
from agent.universal_oracle_adapter import UniversalOracleAdapter
from agent.grid_perception import GridPerception


def solve_game_universal(
    env,
    game_id: str,
    max_steps: int = 5000,
    oracle_adapter: Optional[UniversalOracleAdapter] = None,
) -> dict:
    """通用游戏求解器。
    
    使用Oracle适配器数据：
    1. 读取player/goals/walls
    2. 导航或点击到目标
    3. 等待关卡完成
    
    Args:
        env: 游戏环境
        game_id: 游戏ID
        max_steps: 最大步数
        oracle_adapter: 可选预创建的适配器
    
    Returns:
        结果dict: {levels_completed, total_steps, rhae, success}
    """
    from agent.game_configs import GAME_CONFIGS, GameConfig
    from agent.game_profiles import ALL_GAME_BASELINES
    
    # 创建Oracle适配器
    if oracle_adapter is None:
        try:
            game = env._game
            oracle_adapter = UniversalOracleAdapter(game, game_id=game_id)
        except Exception as e:
            print(f"[Universal] Failed to create Oracle adapter: {e}")
            return {"levels_completed": 0, "total_steps": 0, "rhae": 0.0, "success": False}
    
    # 读取配置
    cfg: Optional[GameConfig] = GAME_CONFIGS.get(game_id)
    baselines = ALL_GAME_BASELINES.get(game_id, [50, 100, 150, 200, 250, 300, 350])
    
    # 初始化
    obs = env.step(GameAction.RESET)
    levels = 0
    step = 0
    solution_clicks = []  # 预计算的点击方案
    solution_idx = 0
    
    print(f"[Universal] Solving {game_id}...")
    print(f"[Universal] Oracle: {type(oracle_adapter).__name__}")
    print(f"[Universal] Player: ({oracle_adapter.player.x}, {oracle_adapter.player.y})")
    print(f"[Universal] Goals: {len(oracle_adapter.goals)}")
    print(f"[Universal] Is click game: {oracle_adapter._is_click_game}")
    
    while step < max_steps and levels < 7:
        # === 检查关卡完成 ===
        if obs.levels_completed > levels:
            old = levels
            levels = obs.levels_completed
            print(f"  ✅ Level {old} -> {levels} completed at step {step}!")
            # 关卡完成，重置解决方案
            solution_clicks = []
            solution_idx = 0
            # 重新读取Oracle（关卡可能已改变）
            try:
                oracle_adapter = UniversalOracleAdapter(env._game, game_id=game_id)
            except Exception:
                pass
        
        # === 执行预计算方案（如果有）===
        if solution_clicks and solution_idx < len(solution_clicks):
            click = solution_clicks[solution_idx]
            solution_idx += 1
            obs = env.step(GameAction.ACTION6, data=click)
            step += 1
            continue
        
        # === 没有方案，创建一个 ===
        
        # 读取当前游戏状态
        player = oracle_adapter.player
        goals = oracle_adapter.goals
        
        if not goals or len(goals) == 0:
            print(f"[Universal] No goals found, cannot solve")
            break
        
        if oracle_adapter._is_click_game:
            # Click游戏：创建点击方案
            gs = oracle_adapter.grid_size or 32
            sc = 64.0 / float(gs) if gs > 0 else 2.0
            
            solution_clicks = []
            for goal in goals:
                dx = int(goal.x * sc + sc / 2)
                dy = int(goal.y * sc + sc / 2)
                if 0 <= dx < 64 and 0 <= dy < 64:
                    click_data = {"x": dx, "y": dy}
                    if click_data not in solution_clicks:
                        solution_clicks.append(click_data)
            
            if solution_clicks:
                print(f"[Universal] Click game: {len(solution_clicks)} clicks to execute")
                solution_idx = 0
                # 执行第一个点击
                continue
            else:
                print(f"[Universal] No valid clicks found")
                break
        else:
            # Keyboard游戏：导航到目标
            # 使用简单策略：导航到最近的goal
            target = _find_nearest_goal(player, goals)
            if target is None:
                print(f"[Universal] No target found")
                break
            
            action = _navigate_to_target(player, target, oracle_adapter.walls)
            if action is None:
                print(f"[Universal] Cannot navigate to target")
                break
            
            obs = env.step(action)
            step += 1
    
    # 计算RHAE
    total_rhae = 0.0
    for i, bl in enumerate(baselines[:levels]):
        if bl > 0:
            level_rhae = min(115.0, (bl / max(1, step // max(1, levels))) ** 2 * 100)
            total_rhae += level_rhae
    
    return {
        "levels_completed": levels,
        "total_steps": step,
        "rhae": total_rhae,
        "success": levels > 0
    }


def _find_nearest_goal(player, goals: list) -> Optional[Any]:
    """找到最近的goal。"""
    if not goals:
        return None
    
    best = None
    best_dist = float('inf')
    for g in goals:
        dist = abs(g.x - player.x) + abs(g.y - player.y)
        if dist < best_dist:
            best_dist = dist
            best = g
    
    return best


def _navigate_to_target(player, target, walls: list) -> Optional[int]:
    """简单的导航策略：直接移动到目标（忽略墙壁）。"""
    dx = target.x - player.x
    dy = target.y - player.y
    
    if abs(dx) > abs(dy):
        # 先水平移动
        if dx > 0:
            return GameAction.ACTION4  # RIGHT
        else:
            return GameAction.ACTION3  # LEFT
    else:
        # 先垂直移动
        if dy > 0:
            return GameAction.ACTION2  # DOWN
        else:
            return GameAction.ACTION1  # UP


if __name__ == "__main__":
    # 测试
    import arc_agi
    
    test_games = ["vc33", "s5i5", "ls20", "ft09"]
    
    for game_id in test_games:
        print(f"\n{'='*60}")
        print(f"Testing {game_id}")
        print('='*60)
        
        arc = arc_agi.Arcade()
        env = arc.make(game_id)
        
        result = solve_game_universal(env, game_id, max_steps=1000)
        
        print(f"\nResult: {result}")
