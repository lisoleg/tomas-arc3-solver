"""Test Universal Oracle Adapter on all 25 games."""
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "environment_files"))

import arc_agi
from arcengine import GameAction
from src.agent.oracle_adapters import auto_detect_adapter, get_oracle_adapter

ALL_GAMES = [
    "ls20", "vc33", "tr87", "tu93", "bp35", "dc22", "s5i5", "sk48",
    "tn36", "ft09", "su15", "lf52", "sc25", "m0r0", "re86", "r11l",
    "cn04", "lp85", "cd82", "g50t", "sp80", "ka59", "ar25", "wa30", "sb26",
]

def test_game(game_id):
    """Test universal adapter on a single game."""
    try:
        arc = arc_agi.Arcade()
        env = arc.make(game_id)
        obs = env.step(GameAction.RESET)

        game = getattr(env, '_game', getattr(env, 'game', None))
        if game is None:
            return {"game_id": game_id, "status": "NO_GAME"}

        # Try auto detect
        adapter = auto_detect_adapter(game)
        if adapter is None:
            return {"game_id": game_id, "status": "NO_ADAPTER"}

        adapter_type = type(adapter).__name__

        # Get entities
        player = adapter.player
        walls = adapter.walls
        goals = adapter.goals
        switchers = adapter.switchers

        result = {
            "game_id": game_id,
            "adapter": adapter_type,
            "has_player": player is not None,
            "player_pos": (player.x, player.y) if player else None,
            "walls_count": len(walls),
            "goals_count": len(goals),
            "switchers_count": len(switchers),
            "level_index": adapter.level_index,
            "win_score": adapter.win_score,
        }

        # Also try universal adapter directly
        from src.agent.universal_oracle_adapter import UniversalOracleAdapter
        uni = UniversalOracleAdapter(game)
        uni_player = uni.player
        uni_walls = uni.walls
        uni_goals = uni.goals

        result["universal_player"] = uni_player is not None
        result["universal_walls"] = len(uni_walls)
        result["universal_goals"] = len(uni_goals)

        return result

    except Exception as e:
        return {"game_id": game_id, "status": "ERROR", "error": str(e)[:200]}


def main():
    print("=" * 90)
    print("Universal Oracle Adapter Test — All 25 Games")
    print("=" * 90)
    print(f"{'Game':<8} {'Adapter':<25} {'Player':<8} {'Walls':<8} {'Goals':<8} {'Switch':<8} {'Uni_P':<8} {'Uni_W':<8} {'Uni_G':<8}")
    print("-" * 90)

    results = []
    for gid in ALL_GAMES:
        r = test_game(gid)
        results.append(r)

        if r.get("status") in ("NO_GAME", "NO_ADAPTER", "ERROR"):
            print(f"{gid:<8} {r.get('status', 'UNKNOWN'):<25} {r.get('error', '')[:50]}")
        else:
            print(f"{gid:<8} {r['adapter']:<25} "
                  f"{'Y' if r['has_player'] else 'N':<8} "
                  f"{r['walls_count']:<8} {r['goals_count']:<8} {r['switchers_count']:<8} "
                  f"{'Y' if r['universal_player'] else 'N':<8} "
                  f"{r['universal_walls']:<8} {r['universal_goals']:<8}")

    # Summary
    print("\n" + "=" * 90)
    has_adapter = sum(1 for r in results if r.get("adapter"))
    has_player = sum(1 for r in results if r.get("has_player") or r.get("universal_player"))
    has_walls = sum(1 for r in results if (r.get("walls_count", 0) + r.get("universal_walls", 0)) > 0)
    has_goals = sum(1 for r in results if (r.get("goals_count", 0) + r.get("universal_goals", 0)) > 0)

    print(f"Games with adapter: {has_adapter}/25")
    print(f"Games with player detected: {has_player}/25")
    print(f"Games with walls detected: {has_walls}/25")
    print(f"Games with goals detected: {has_goals}/25")
    print("=" * 90)


if __name__ == "__main__":
    main()
