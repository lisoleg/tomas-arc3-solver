#!/usr/bin/env python3
"""depth3_mdl_search.py - Depth-1/2/3 search ranked by MDL."""
import sys, json, os
import numpy as np

sys.path.insert(0, '.')

from src.solver.param_inference import ParamInference
from src.core.dsl_primitives import DSLElement, ProgramNode


def verify(prog, pairs):
    for pair in pairs:
        for i, inp in enumerate(pair['input']):
            if i < len(pair['output']):
                try:
                    if not np.array_equal(prog.apply(inp), pair['output'][i]):
                        return False
                except Exception:
                    return False
    return True


def chain(a, b):
    if a.element is None and not a.children:
        return b
    if b.element is None and not b.children:
        return a
    n = ProgramNode(a.element)
    n.children = list(a.children) + [b]
    n.combo_type = "chain"
    n.total_mdl = n.compute_mdl()
    return n


def solve(task_data, max_per_depth=5000):
    train = task_data.get('train', [])
    pairs = []
    for p in train:
        inp = [np.array(f, np.int8) for f in p.get('input', [])]
        out = [np.array(f, np.int8) for f in p.get('output', [])]
        pairs.append({'input': inp, 'output': out})
    if not pairs:
        return None

    pi = ParamInference()
    cands = pi.infer_candidates(pairs)
    print(f'  Candidates: {len(cands)}')

    # Depth-1
    for p in cands:
        if verify(p, pairs):
            return p, 1

    # Depth-2: sort by MDL, try top-N combos
    cands_sorted = sorted(cands, key=lambda c: c.total_mdl if c.total_mdl > 0 else 1e9)
    top = cands_sorted[:min(80, len(cands_sorted))]
    print(f'  Trying depth-2 ({len(top)}x{len(top)} combos)...')
    cnt = 0
    for i, c1 in enumerate(top):
        for c2 in top:
            cnt += 1
            if cnt > max_per_depth:
                break
            prog = chain(c1, c2)
            if verify(prog, pairs):
                return prog, 2
        if cnt > max_per_depth:
            break

    # Depth-3
    print(f'  Trying depth-3 (top-30 x top-30 x top-30)...')
    top3 = cands_sorted[:min(30, len(cands_sorted))]
    cnt = 0
    for i, c1 in enumerate(top3):
        for j, c2 in enumerate(top3):
            for c3 in top3:
                cnt += 1
                if cnt > max_per_depth:
                    break
                prog = chain(chain(c1, c2), c3)
                if verify(prog, pairs):
                    return prog, 3
            if cnt > max_per_depth:
                break
        if cnt > max_per_depth:
            break

    return None


if __name__ == '__main__':
    d = 'data/real_arc_converted'
    tasks = sorted(f for f in os.listdir(d) if f.endswith('.json'))
    correct = 0
    total = 0
    correct_tasks = []
    by_depth = {}
    for tf in tasks:
        tid = tf[:-5]
        with open(os.path.join(d, tf)) as f:
            task = json.load(f)
        task['task_id'] = tid
        test_items = task.get('test', [])
        if not test_items or not test_items[0].get('output'):
            continue
        test_output = np.array(test_items[0]['output'][0], np.int8)
        total += 1
        print(f'{tid}...')
        result = solve(task)
        if result and result[0] is not None:
            prog, depth = result
            test_inp = np.array(task['test'][0]['input'][0], np.int8)
            pred = prog.apply(test_inp)
            if np.array_equal(pred, test_output):
                correct += 1
                correct_tasks.append(tid)
                by_depth[depth] = by_depth.get(depth, 0) + 1
                print(f'  ✓ SOLVED (depth={depth}): {prog}')
    print(f'\n=== Depth-3 MDL Results ===')
    print(f'Correct: {correct}/{total} ({correct/total*100:.1f}%)')
    print(f'By depth: {by_depth}')
    print(f'Correct tasks: {correct_tasks}')
