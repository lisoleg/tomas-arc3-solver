"""
Mock Oracle: Simulates ARC-AGI-3 Oracle Mode for RHAE Estimation
"""

import numpy as np
import json
import argparse
import time
from typing import Dict, List, Any, Callable, Tuple

class MockOracle:
    def __init__(self, failure_rate: float = 0.0, seed: int = 42):
        self.failure_rate = failure_rate
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.grid = None
        self.max_steps = 200
        self.done = False

    def reset(self, grid: np.ndarray) -> Dict:
        self.step_count = 0
        self.grid = grid.copy()
        self.done = False
        return {'grid': self.grid.copy(), 'step': 0, 'done': False}

    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            return {'grid': self.grid.copy(), 'step': self.step_count, 'done': True}, 0.0, True, {}
        self.step_count += 1
        a_type = action.get('type', 'NOOP')
        if self.failure_rate > 0 and self.rng.random() < self.failure_rate:
            self.done = self.step_count >= self.max_steps
            return {'grid': self.grid.copy(), 'step': self.step_count, 'done': self.done}, -0.1, self.done, {'injected': True}
        reward = self._apply(a_type, action)
        self.done = self.step_count >= self.max_steps
        return {'grid': self.grid.copy(), 'step': self.step_count, 'done': self.done}, reward, self.done, {}

    def _apply(self, a_type: str, action: Dict) -> float:
        h, w = self.grid.shape
        if a_type == 'LEFT':
            self._shift(dx=-1)
            return 0.1
        elif a_type == 'RIGHT':
            self._shift(dx=1)
            return 0.1
        elif a_type == 'UP':
            self._shift(dy=-1)
            return 0.1
        elif a_type == 'DOWN':
            self._shift(dx=0, dy=1)
            return 0.1
        return -0.05

    def _shift(self, dx: int = 0, dy: int = 0):
        h, w = self.grid.shape
        nonzero = np.argwhere(self.grid > 0)
        if len(nonzero) == 0:
            return
        new = self.grid.copy()
        ys, xs = nonzero[:, 0], nonzero[:, 1]
        new[ys, xs] = 0
        new[np.clip(ys + dy, 0, h-1), np.clip(xs + dx, 0, w-1)] = self.grid[ys, xs]
        self.grid = new

    def estimate_rhae(self, actual_steps: int, optimal_steps: int) -> float:
        return actual_steps / max(optimal_steps, 1)

def run_episode(oracle: MockOracle, task: Dict, solver_actions: List[Dict]) -> Dict:
    inp = np.array(task['demos'][0]['input'], dtype=np.int32)
    obs = oracle.reset(inp)
    total_reward = 0.0
    for i, action in enumerate(solver_actions):
        obs, reward, done, info = oracle.step(action)
        total_reward += reward
        if done:
            break
    rhae = oracle.estimate_rhae(oracle.step_count, task.get('optimal_steps', 5))
    return {
        'task_id': task.get('game_id', 'unknown'),
        'steps': oracle.step_count,
        'total_reward': total_reward,
        'rhae_estimate': rhae,
        'solved': oracle.step_count <= task.get('optimal_steps', 5) * 2,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--solver', type=str, default=None)
    args = parser.parse_args()
    with open(args.task) as f:
        task = json.load(f)
    oracle = MockOracle()
    inp = np.array(task['demos'][0]['input'], dtype=np.int32)
    obs = oracle.reset(inp)
    actions = [{'type': 'RIGHT'}] * 3
    result = run_episode(oracle, task, actions)
    print(f"\nEpisode complete:")
    print(f"   Steps: {result['steps']}")
    print(f"   RHAE estimate: {result['rhae_estimate']:.2f}")
    print(f"   Total reward: {result['total_reward']:.2f}")
    print(f"   Solved: {result['solved']}")
