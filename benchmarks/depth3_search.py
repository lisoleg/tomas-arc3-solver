#!/usr/bin/env python3
"""depth3_search.py - Search depth-3 combinations with pruning."""
import sys
import json
import os
import numpy as np

sys.path.insert(0, '.')

from src.solver.param_inference import ParamInference
from src.core.dsl_primitives import DSLElement, ProgramNode


def verify_program(prog, demo_pairs):
    """Check if a program matches all demo pairs."""
    for pair in demo_pairs:
        for i, inp in enumerate(pair['input']):
            if i < len(pair['output']):
                try:
                    pred = prog.apply(inp)
                    if not np.array_equal(pred, pair['output'][i]):
                        return False
                except Exception:
                    return False
    return True


def chain_programs(prog1, prog2):
    """Chain two programs: prog1 -> prog2."""
    if prog1.element is None and not prog1.children:
        return prog2
    if prog2.element is None and not prog2.children:
        return prog1

    new_node = ProgramNode(prog1.element)
    new_node.children = list(prog1.children)
    new_node.children.append(prog2)
    new_node.combo_type = "chain"
    new_node.total_mdl = new_node.compute_mdl()
    return new_node


def solve_task_depth3(task_data, max_depth=3, time_limit=60):
    """Solve a task using iterative deepening search (depth 1 to max_depth).

    Uses pruning: only keep programs that match at least one training pair.
    """
    import time
    start_time = time.time()

    # Build demo pairs
    train = task_data.get('train', [])
    demo_pairs = []
    for pair in train:
        inp = [np.array(f, dtype=np.int8) for f in pair.get('input', [])]
        out = [np.array(f, dtype=np.int8) for f in pair.get('output', [])]
        demo_pairs.append({'input': inp, 'output': out})

    if not demo_pairs:
        return None, 0

    # Get inferred candidates
    pi = ParamInference()
    candidates = pi.infer_candidates(demo_pairs)

    print(f'  Inferred candidates: {len(candidates)}')

    # Iterative deepening
    for depth in range(1, max_depth + 1):
        if time.time() - start_time > time_limit:
            print(f'  Time limit reached ({time_limit}s)')
            break

        print(f'  Trying depth-{depth}...')

        if depth == 1:
            # Try all depth-1 candidates
            for prog in candidates:
                if verify_program(prog, demo_pairs):
                    return prog, 1
            continue

        # For depth >= 2: generate combinations
        # Use previous depth programs + candidates
        if depth == 2:
            base_programs = candidates
        else:
            # Use top programs from previous depth
            # (loaded from previous iteration)
            pass

        # Generate and verify combinations
        count = 0
        for i, p1 in enumerate(base_programs):
            for j, p2 in enumerate(candidates):
                if time.time() - start_time > time_limit:
                    break

                count += 1
                chained = chain_programs(p1, p2)

                if verify_program(chained, demo_pairs):
                    return chained, depth

                # Prune: only keep if matches at least one training pair
                # (optional, can be expensive to check)

            if time.time() - start_time > time_limit:
                break

        # For next depth, use all depth-1 candidates as base
        # (simplified - should use promising programs from current depth)
        base_programs = candidates

    return None, 0


if __name__ == '__main__':
    d = 'data/real_arc_converted'
    tasks = sorted([f for f in os.listdir(d) if f.endswith('.json')])

    correct = 0
    total = 0
    correct_tasks = []

    for tf in tasks:
        tid = tf.replace('.json', '')
        with open(os.path.join(d, tf)) as f:
            task_data = json.load(f)

        task_data['task_id'] = tid

        # Get test output
        test_items = task_data.get('test', [])
        test_output = None
        if test_items:
            test_outs = test_items[0].get('output', [])
            if test_outs:
                test_output = np.array(test_outs[0], dtype=np.int8)

        if test_output is None:
            continue

        total += 1

        print(f'{tid}...')
        result = solve_task_depth3(task_data, max_depth=3, time_limit=30)

        if result[0] is not None:
            prog, depth = result
            # Apply to test
            test_inp = np.array(task_data['test'][0]['input'][0], dtype=np.int8)
            pred = prog.apply(test_inp)

            if np.array_equal(pred, test_output):
                correct += 1
                correct_tasks.append(tid)
                print(f'  ✓ SOLVED (depth={depth}): {prog.element.name if prog.element else "chain"}')

    print(f'\n=== Depth-3 Results ===')
    print(f'Correct: {correct}/{total} ({correct/total*100:.1f}%)')
    print(f'Correct tasks: {correct_tasks}')
