"""
ARC-AGI-3 Offline Diagnostics Suite
=====================================
Integrates: Mock Oracle + GaussEx Pre-check + Topo-Map Sensitivity + LOTO Generalization
Author: TOMAS Framework
Version: v3.3.0-diagnostics
"""

import numpy as np
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# =============================================================================
#  Section 1: Mock Oracle (Simulates Kaggle Backend Behavior)
# =============================================================================

class MockOracle:
    """
    Simulates ARC-AGI-3 Oracle Mode interaction.
    Replicates Kaggle backend behavior:
    - Accepts action dict
    - Returns (obs, reward, done, info)
    - Tracks RHAE (Relative Human Action Efficiency)
    - Supports failure injection (random / systematic)
    """

    def __init__(
        self,
        true_program: Optional[List[str]] = None,
        grid_shape: Tuple[int, int] = (30, 30),
        failure_rate: float = 0.0,
        seed: int = 42,
    ):
        self.true_program = true_program or []
        self.grid_shape = grid_shape
        self.failure_rate = failure_rate
        self.rng = np.random.RandomState(seed)
        self.step_count = 0
        self.total_reward = 0.0
        self.max_steps = 200
        self.grid = np.zeros(grid_shape, dtype=np.int32)
        self.done = False

    def reset(self, initial_grid: Optional[np.ndarray] = None) -> Dict:
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        if initial_grid is not None:
            self.grid = initial_grid.copy()
        else:
            self.grid = np.zeros(self.grid_shape, dtype=np.int32)
        return {'grid': self.grid.copy(), 'step': 0, 'done': False}

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            return {'grid': self.grid.copy(), 'step': self.step_count, 'done': True}, 0.0, True, {'msg': 'Already done'}

        self.step_count += 1
        a_type = action.get('type', 'NOOP')

        # --- Failure injection ---
        if self.failure_rate > 0 and self.rng.random() < self.failure_rate:
            reward = -0.1
            self.total_reward += reward
            self.done = self.step_count >= self.max_steps
            return (
                {'grid': self.grid.copy(), 'step': self.step_count, 'done': self.done},
                reward, self.done, {'msg': 'Failure injected', 'injected': True}
            )

        # --- Execute action (simplified grid world) ---
        old_grid = self.grid.copy()
        try:
            self._apply_action(a_type, action)
            reward = self._compute_reward(old_grid, self.grid)
        except Exception as e:
            reward = -0.5
            logger.warning(f"  [MockOracle] Invalid action {a_type}: {e}")

        self.total_reward += reward
        self.done = self.step_count >= self.max_steps

        return (
            {'grid': self.grid.copy(), 'step': self.step_count, 'done': self.done},
            reward, self.done, {'msg': 'OK', 'injected': False}
        )

    def _apply_action(self, a_type: str, action: Dict):
        h, w = self.grid.shape
        if a_type == 'LEFT':
            self._move_sprite(dx=-1, dy=0)
        elif a_type == 'RIGHT':
            self._move_sprite(dx=1, dy=0)
        elif a_type == 'UP':
            self._move_sprite(dx=0, dy=-1)
        elif a_type == 'DOWN':
            self._move_sprite(dx=0, dy=1)
        elif a_type == 'SPACE':
            pass
        elif a_type == 'CLICK':
            x = action.get('x', 0)
            y = action.get('y', 0)
            if 0 <= x < w and 0 <= y < h:
                self.grid[y, x] = 0

    def _move_sprite(self, dx: int, dy: int):
        h, w = self.grid.shape
        nonzero = np.argwhere(self.grid > 0)
        if len(nonzero) == 0:
            return
        from scipy.ndimage import label
        labeled, num = label(self.grid > 0)
        if num == 0:
            return
        sizes = [(labeled == i).sum() for i in range(1, num + 1)]
        main_comp = np.argmax(sizes) + 1
        mask = labeled == main_comp
        new_grid = self.grid.copy()
        new_grid[mask] = 0
        ys, xs = np.where(mask)
        new_ys = np.clip(ys + dy, 0, h - 1)
        new_xs = np.clip(xs + dx, 0, w - 1)
        new_grid[new_ys, new_xs] = self.grid[ys, xs]
        self.grid = new_grid

    def _compute_reward(self, old_grid: np.ndarray, new_grid: np.ndarray) -> float:
        if np.array_equal(old_grid, new_grid):
            return -0.05
        return 0.1

    def estimate_rhae(self, solver_steps: int, optimal_steps: int) -> float:
        if optimal_steps <= 0:
            return float(solver_steps)
        return solver_steps / optimal_steps

# =============================================================================
#  Section 2: GaussEx Pre-Check (Demo Consistency Verification)
# =============================================================================

class GaussExPreChecker:
    def __init__(self, grid_executor: Optional[Callable] = None):
        self.executor = grid_executor or self._default_executor
        self.results: List[Dict] = []

    def _default_executor(self, dsl_seq: List[str], input_grid: np.ndarray) -> np.ndarray:
        output = input_grid.copy()
        for cmd in dsl_seq:
            if cmd.startswith("repeat("):
                inner = cmd.split('(')[1].split(')')[0]
                a_type = inner.split(',')[0].strip().strip("'")
                count = int(inner.split(',')[1].strip())
                for _ in range(count):
                    output = self._apply_primitive(a_type, output)
            elif cmd.startswith("action('"):
                a_type = cmd.split("'")[1]
                output = self._apply_primitive(a_type, output)
        return output

    def _apply_primitive(self, a_type: str, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        if a_type in ('LEFT', 'RIGHT', 'UP', 'DOWN'):
            dx, dy = {'LEFT': (-1,0), 'RIGHT': (1,0), 'UP': (0,-1), 'DOWN': (0,1)}[a_type]
            nonzero = np.argwhere(grid > 0)
            if len(nonzero) == 0:
                return grid
            new_grid = grid.copy()
            ys, xs = nonzero[:, 0], nonzero[:, 1]
            new_ys = np.clip(ys + dy, 0, h - 1)
            new_xs = np.clip(xs + dx, 0, w - 1)
            new_grid[ys, xs] = 0
            new_grid[new_ys, new_xs] = grid[ys, xs]
            return new_grid
        return grid

    def check_program(
        self,
        dsl_sequence: List[str],
        demos: List[Tuple[np.ndarray, np.ndarray]],
        program_name: str = "unknown",
    ) -> Dict:
        result = {
            'program': program_name,
            'passed': True,
            'failed_demos': [],
            'details': [],
        }
        for idx, (inp, expected_out) in enumerate(demos):
            actual_out = self.executor(dsl_sequence, inp)
            match = np.array_equal(actual_out, expected_out)
            detail = {
                'demo_idx': idx,
                'passed': match,
                'input_shape': inp.shape,
                'output_shape': actual_out.shape,
            }
            if not match:
                result['passed'] = False
                result['failed_demos'].append(idx)
                detail['mismatches'] = int((actual_out != expected_out).sum())
            result['details'].append(detail)

        self.results.append(result)
        return result

    def check_batch(
        self,
        candidates: List[Tuple[List[str], str]],
        demos: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[Dict]:
        results = []
        for dsl_seq, name in candidates:
            r = self.check_program(dsl_seq, demos, name)
            results.append(r)
            status = "PASS" if r['passed'] else f"FAIL (demos: {r['failed_demos']})"
            logger.info(f"  [GaussEx] {name}: {status}")
        return results

    def save_report(self, path: str = "gaussex_report.json"):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"  [GaussEx] Report saved to {path}")

# =============================================================================
#  Section 3: Topo-Map Sensitivity Analyzer
# =============================================================================

class TopoMapSensitivityAnalyzer:
    def __init__(self, nar_conv_executor: Optional[Callable] = None):
        self.executor = nar_conv_executor or self._dummy_topo_extractor
        self.results: List[Dict] = []

    def _dummy_topo_extractor(self, grid: np.ndarray) -> Dict[str, Any]:
        h, w = grid.shape
        unique = len(np.unique(grid))
        try:
            from scipy.ndimage import label
            _, num = label(grid > 0)
        except:
            num = 1
        return {
            'euler_char': int(num),
            'unique_colors': unique,
            'period_x': self._detect_period(grid, axis=1),
            'period_y': self._detect_period(grid, axis=0),
            'sym_h': float(np.array_equal(grid, np.fliplr(grid))),
            'sym_v': float(np.array_equal(grid, np.flipud(grid))),
        }

    def _detect_period(self, arr: np.ndarray, axis: int) -> int:
        if axis == 0:
            for p in range(1, arr.shape[0] // 2 + 1):
                if np.array_equal(arr[:-p], arr[p:]):
                    return p
        else:
            for p in range(1, arr.shape[1] // 2 + 1):
                if np.array_equal(arr[:, :-p], arr[:, p:]):
                    return p
        return 0

    def test_color_permutation(self, grid: np.ndarray, n_shuffles: int = 10) -> Dict:
        base_topo = self.executor(grid)
        non_zero = [c for c in np.unique(grid) if c != 0]
        reports = []
        for i in range(n_shuffles):
            rng = np.random.RandomState(i)
            shuffled = non_zero.copy()
            rng.shuffle(shuffled)
            mapping = dict(zip(non_zero, shuffled))
            mapping[0] = 0
            perturbed = np.vectorize(mapping.get)(grid)
            pert_topo = self.executor(perturbed)
            diffs = {}
            for k in base_topo:
                if isinstance(base_topo[k], (int, float)):
                    diffs[k] = abs(base_topo[k] - pert_topo.get(k, 0))
            max_diff = max(diffs.values()) if diffs else 0
            passed = max_diff == 0
            reports.append({'shuffle_seed': i, 'passed': passed, 'max_diff': max_diff, 'diffs': diffs})

        result = {
            'test': 'color_permutation',
            'passed': all(r['passed'] for r in reports),
            'n_tested': n_shuffles,
            'failures': [r for r in reports if not r['passed']],
        }
        self.results.append(result)
        status = "PASS" if result['passed'] else f"WARN ({len(result['failures'])} failures)"
        logger.info(f"  [Topo-Sens] Color permutation: {status}")
        return result

    def test_noise_injection(self, grid: np.ndarray, n_trials: int = 10, noise_rate: float = 0.05) -> Dict:
        base_topo = self.executor(grid)
        reports = []
        for i in range(n_trials):
            rng = np.random.RandomState(i)
            perturbed = grid.copy()
            h, w = grid.shape
            n_noise = max(1, int(h * w * noise_rate))
            for _ in range(n_noise):
                y, x = rng.randint(0, h), rng.randint(0, w)
                perturbed[y, x] = rng.randint(1, 10)
            pert_topo = self.executor(perturbed)
            diffs = {}
            for k in base_topo:
                if isinstance(base_topo[k], (int, float)):
                    diffs[k] = abs(base_topo[k] - pert_topo.get(k, 0))
            max_diff = max(diffs.values()) if diffs else 0
            reports.append({'trial': i, 'max_diff': max_diff, 'diffs': diffs})

        max_diff_overall = max(r['max_diff'] for r in reports)
        avg_diff = np.mean([r['max_diff'] for r in reports])
        result = {
            'test': 'noise_injection',
            'passed': max_diff_overall <= 1,
            'n_tested': n_trials,
            'max_diff': max_diff_overall,
            'avg_diff': float(avg_diff),
        }
        self.results.append(result)
        status = "PASS" if result['passed'] else f"WARN (max_diff={max_diff_overall})"
        logger.info(f"  [Topo-Sens] Noise injection: {status}")
        return result

    def test_periodicity_robustness(self, grid: np.ndarray) -> Dict:
        base_topo = self.executor(grid)
        base_period_x = base_topo.get('period_x', 0)
        base_period_y = base_topo.get('period_y', 0)
        if base_period_x == 0 and base_period_y == 0:
            result = {'test': 'periodicity', 'passed': True, 'note': 'No periodicity to test'}
            self.results.append(result)
            logger.info("  [Topo-Sens] Periodicity: SKIP (no period)")
            return result
        perturbed = grid.copy()
        h, w = grid.shape
        mid_y, mid_x = h // 2, w // 2
        perturbed[mid_y, mid_x] = (perturbed[mid_y, mid_x] + 1) % 10
        pert_topo = self.executor(perturbed)
        degraded_x = pert_topo.get('period_x', 0) < base_period_x
        degraded_y = pert_topo.get('period_y', 0) < base_period_y
        result = {
            'test': 'periodicity',
            'passed': degraded_x or degraded_y,
            'base_period': (base_period_x, base_period_y),
            'perturbed_period': (pert_topo.get('period_x', 0), pert_topo.get('period_y', 0)),
            'note': 'Periodicity should degrade gracefully on perturbation',
        }
        self.results.append(result)
        status = "PASS" if result['passed'] else "WARN"
        logger.info(f"  [Topo-Sens] Periodicity robustness: {status}")
        return result

    def run_all_tests(self, grid: np.ndarray) -> List[Dict]:
        logger.info(f"[Topo-Sens] Running all tests on grid {grid.shape}...")
        self.test_color_permutation(grid)
        self.test_noise_injection(grid)
        self.test_periodicity_robustness(grid)
        return self.results

    def save_report(self, path: str = "topo_sensitivity_report.json"):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"  [Topo-Sens] Report saved to {path}")

# =============================================================================
#  Section 4: LOTO (Leave-One-Topology-Out) Generalization Tester
# =============================================================================

class LOTOGeneralizationTester:
    def __init__(self, tomas_learner=None, topo_extractor: Optional[Callable] = None):
        self.learner = tomas_learner
        self.topo_extractor = topo_extractor or self._default_topo_extractor
        self.results: List[Dict] = []

    def _default_topo_extractor(self, grid: np.ndarray) -> Dict[str, Any]:
        h, w = grid.shape
        unique = len(np.unique(grid))
        try:
            from scipy.ndimage import label
            _, num = label(grid > 0)
        except:
            num = 1
        return {'euler_char': int(num), 'period_rank': 0, 'symmetry': []}

    def _classify_topology(self, grid: np.ndarray) -> str:
        topo = self.topo_extractor(grid)
        sig_parts = [
            f"ec{topo.get('euler_char', '?')}",
            f"pr{topo.get('period_rank', '?')}",
        ]
        sym = topo.get('symmetry', [])
        if sym:
            sig_parts.append(f"sym{'-'.join(sorted(sym))}")
        return "_".join(sig_parts)

    def partition_by_topology(self, tasks: List[Dict]) -> Dict[str, List[Dict]]:
        groups = {}
        for task in tasks:
            inp = task.get('demos', [{}])[0].get('input', np.zeros((3,3)))
            topo_class = self._classify_topology(inp)
            groups.setdefault(topo_class, []).append(task)
        return groups

    def run_loto(self, tasks: List[Dict], n_folds: Optional[int] = None) -> List[Dict]:
        groups = self.partition_by_topology(tasks)
        topo_classes = list(groups.keys())
        if n_folds is not None:
            topo_classes = topo_classes[:n_folds]
        logger.info(f"[LOTO] Topology groups: {list(groups.keys())}")
        fold_results = []
        for held_out in topo_classes:
            train_tasks = [t for tc, tasks_t in groups.items() if tc != held_out for t in tasks_t]
            test_tasks = groups[held_out]
            logger.info(f"  [LOTO] Fold: hold out '{held_out}' | train={len(train_tasks)}, test={len(test_tasks)}")

            fast_path_hits = 0
            fast_path_misses = 0
            for task in test_tasks:
                if self.learner:
                    inp = task.get('demos', [{}])[0].get('input', np.zeros((3,3)))
                    topo_feat = self.topo_extractor(inp)
                    game_tags = task.get('tags', [])
                    dsl_seq, fp = self.learner.try_fast_path(
                        game_state={'grid': inp, 'sprites': []},
                        game_tags=game_tags,
                        topo_features=topo_feat,
                    )
                    if dsl_seq is not None:
                        fast_path_hits += 1
                    else:
                        fast_path_misses += 1
                else:
                    fast_path_misses += 1

            hit_rate = fast_path_hits / max(len(test_tasks), 1)
            fold_result = {
                'held_out_topology': held_out,
                'train_size': len(train_tasks),
                'test_size': len(test_tasks),
                'fast_path_hits': fast_path_hits,
                'fast_path_misses': fast_path_misses,
                'hit_rate': hit_rate,
            }
            fold_results.append(fold_result)
            logger.info(f"  [LOTO] Hit rate: {hit_rate:.1%}")

        self.results = fold_results
        return fold_results

    def save_report(self, path: str = "loto_report.json"):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"  [LOTO] Report saved to {path}")

# =============================================================================
#  Section 5: Beam Profile Analyzer
# =============================================================================

class BeamProfileAnalyzer:
    def __init__(self):
        self.profiles: List[Dict] = []

    def analyze_beam(
        self,
        task_id: str,
        candidates: List[Tuple[List[str], float]],
        found_solution: bool,
        solution_depth: int,
        optimal_depth: int,
    ) -> Dict:
        if not candidates:
            return {'task_id': task_id, 'error': 'no_candidates'}
        scores = [s for _, s in candidates]
        lengths = [len(dsl) for dsl, _ in candidates]
        profile = {
            'task_id': task_id,
            'n_candidates': len(candidates),
            'found_solution': found_solution,
            'solution_depth': solution_depth,
            'optimal_depth': optimal_depth,
            'depth_efficiency': optimal_depth / max(solution_depth, 1),
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
            'score_min': float(min(scores)),
            'score_max': float(max(scores)),
            'top5_score_range': float(scores[0] - scores[min(4, len(scores)-1)]),
            'avg_candidate_length': float(np.mean(lengths)),
            'ambiguity': float(np.std(scores)),
        }
        warnings = []
        if found_solution and solution_depth > optimal_depth * 2:
            warnings.append('DEPTH_INEFFICIENT')
        if profile['top5_score_range'] < 0.01:
            warnings.append('SCORE_DEGENERATE')
        if profile['ambiguity'] > profile['score_mean'] * 0.5:
            warnings.append('HIGH_AMBIGUITY')
        if not found_solution:
            warnings.append('NO_SOLUTION')
        profile['warnings'] = warnings
        self.profiles.append(profile)
        if warnings:
            logger.warning(f"  [Beam] {task_id}: warnings={warnings}")
        else:
            logger.info(f"  [Beam] {task_id}: OK")
        return profile

    def save_report(self, path: str = "beam_profile.json"):
        with open(path, 'w') as f:
            json.dump(self.profiles, f, indent=2)
        logger.info(f"  [Beam] Profile saved to {path}")

# =============================================================================
#  Section 6: RHAE Estimator
# =============================================================================

class RHAEEstimator:
    def __init__(self, mock_oracle: MockOracle):
        self.oracle = mock_oracle
        self.results: List[Dict] = []

    def run_episode(
        self,
        task_id: str,
        solver_fn: Callable[[np.ndarray], List[Dict]],
        initial_grid: np.ndarray,
        optimal_steps: int,
    ) -> Dict:
        obs = self.oracle.reset(initial_grid)
        actions = solver_fn(initial_grid)
        total_reward = 0.0
        done = False
        step_details = []
        for action in actions:
            obs, reward, done, info = self.oracle.step(action)
            total_reward += reward
            step_details.append({
                'action': action,
                'reward': reward,
                'step': obs['step'],
                'done': done,
                'injected': info.get('injected', False),
            })
            if done:
                break
        actual_steps = len(step_details)
        rhae = self.oracle.estimate_rhae(actual_steps, optimal_steps)
        result = {
            'task_id': task_id,
            'actual_steps': actual_steps,
            'optimal_steps': optimal_steps,
            'rhae_estimate': rhae,
            'total_reward': total_reward,
            'solved': actual_steps <= optimal_steps * 2,
            'n_injections': sum(1 for s in step_details if s['injected']),
        }
        self.results.append(result)
        status = "OK" if result['solved'] else "WARN"
        logger.info(f"  [RHAE] {task_id}: {status} RHAE={rhae:.2f} (steps={actual_steps}/{optimal_steps})")
        return result

    def save_report(self, path: str = "rhae_report.json"):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"  [RHAE] Report saved to {path}")

# =============================================================================
#  Section 7: Master Orchestrator
# =============================================================================

class Arc3DiagnosticsOrchestrator:
    def __init__(self, tomas_learner=None, output_dir: str = "diagnostics_output"):
        self.learner = tomas_learner
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.gaussex = GaussExPreChecker()
        self.topo_analyzer = TopoMapSensitivityAnalyzer()
        self.loto_tester = LOTOGeneralizationTester(tomas_learner=tomas_learner)
        self.beam_analyzer = BeamProfileAnalyzer()
        self.mock_oracle = MockOracle()
        self.rhae_estimator = RHAEEstimator(self.mock_oracle)

    def run_full_suite(
        self,
        tasks: List[Dict],
        demos_dict: Optional[Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = None,
    ) -> Dict:
        logger.info("=" * 70)
        logger.info("[DIAGNOSTICS] Starting full offline diagnostics suite")
        logger.info("=" * 70)
        report = {
            'timestamp': time.time(),
            'n_tasks': len(tasks),
            'gaussex': [],
            'topo_sensitivity': [],
            'loto': [],
            'beam': [],
            'rhae': [],
        }

        # Phase 1: GaussEx
        logger.info("\n--- Phase 1: GaussEx Pre-Check ---")
        if demos_dict:
            for task_id, demos in demos_dict.items():
                candidates = [([f"action('DEMO_PASS')"], "demo_pass")]
                results = self.gaussex.check_batch(candidates, demos)
                report['gaussex'].extend(results)
        self.gaussex.save_report(str(self.output_dir / "gaussex_report.json"))

        # Phase 2: Topo-Map Sensitivity
        logger.info("\n--- Phase 2: Topo-Map Sensitivity ---")
        for task in tasks[:5]:
            inp = task.get('demos', [{}])[0].get('input', None)
            if inp is not None:
                results = self.topo_analyzer.run_all_tests(inp)
                report['topo_sensitivity'].extend(results)
        self.topo_analyzer.save_report(str(self.output_dir / "topo_sensitivity_report.json"))

        # Phase 3: LOTO
        logger.info("\n--- Phase 3: LOTO ---")
        loto_results = self.loto_tester.run_loto(tasks)
        report['loto'] = loto_results
        self.loto_tester.save_report(str(self.output_dir / "loto_report.json"))

        # Phase 4: Beam
        logger.info("\n--- Phase 4: Beam ---")
        for i, task in enumerate(tasks[:3]):
            candidates = [([f"action('STEP_{j}')"], 1.0 - j * 0.1) for j in range(5)]
            profile = self.beam_analyzer.analyze_beam(
                task.get('game_id', f'task_{i}'),
                candidates,
                found_solution=True,
                solution_depth=5 + i,
                optimal_depth=3,
            )
            report['beam'].append(profile)
        self.beam_analyzer.save_report(str(self.output_dir / "beam_profile.json"))

        # Phase 5: RHAE
        logger.info("\n--- Phase 5: RHAE ---")
        for i, task in enumerate(tasks[:3]):
            inp = task.get('demos', [{}])[0].get('input', np.zeros((5,5)))
            def solver(grid):
                return [{'type': 'RIGHT'}] * (3 + i)
            result = self.rhae_estimator.run_episode(
                task.get('game_id', f'task_{i}'), solver, inp, optimal_steps=3
            )
            report['rhae'].append(result)
        self.rhae_estimator.save_report(str(self.output_dir / "rhae_report.json"))

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("[DIAGNOSTICS] Suite complete.")
        logger.info("=" * 70)
        summary = self._generate_summary(report)
        with open(self.output_dir / "diagnostics_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  [SUMMARY] Saved to {self.output_dir / 'diagnostics_summary.json'}")
        return report

    def _generate_summary(self, report: Dict) -> Dict:
        summary = {
            'timestamp': report['timestamp'],
            'n_tasks': report['n_tasks'],
            'gaussex_pass_rate': 'N/A',
            'topo_pass_rate': 'N/A',
            'loto_avg_hit_rate': 'N/A',
            'beam_warning_rate': 'N/A',
            'rhae_avg': 'N/A',
        }
        if report['gaussex']:
            n_pass = sum(1 for r in report['gaussex'] if r['passed'])
            summary['gaussex_pass_rate'] = f"{n_pass}/{len(report['gaussex'])}"
        if report['topo_sensitivity']:
            n_pass = sum(1 for r in report['topo_sensitivity'] if r['passed'])
            summary['topo_pass_rate'] = f"{n_pass}/{len(report['topo_sensitivity'])}"
        if report['loto']:
            avg_hit = np.mean([r['hit_rate'] for r in report['loto']])
            summary['loto_avg_hit_rate'] = f"{avg_hit:.1%}"
        if report['beam']:
            n_warn = sum(1 for r in report['beam'] if r.get('warnings'))
            summary['beam_warning_rate'] = f"{n_warn}/{len(report['beam'])}"
        if report['rhae']:
            avg_rhae = np.mean([r['rhae_estimate'] for r in report['rhae']])
            summary['rhae_avg'] = f"{avg_rhae:.2f}"
        return summary

# =============================================================================
#  CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ARC-AGI-3 Offline Diagnostics Suite")
    parser.add_argument('--tasks', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='diagnostics_output')
    parser.add_argument('--max-tasks', type=int, default=10)
    args = parser.parse_args()

    if args.tasks and Path(args.tasks).exists():
        with open(args.tasks) as f:
            tasks = json.load(f)
    else:
        tasks = []
        for i in range(3):
            h, w = 5 + i, 5 + i
            inp = np.random.RandomState(i).randint(0, 3, (h, w))
            tasks.append({
                'game_id': f'synthetic_task_{i}',
                'level_id': '0',
                'demos': [{'input': inp, 'output': inp + 1}],
                'tags': ['synthetic'],
            })
        logger.info("[Main] Using synthetic tasks")

    tasks = tasks[:args.max_tasks]
    orch = Arc3DiagnosticsOrchestrator(output_dir=args.output_dir)
    report = orch.run_full_suite(tasks)
    logger.info("\nAll diagnostics complete.")
