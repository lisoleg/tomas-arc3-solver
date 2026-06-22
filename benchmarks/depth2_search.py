#!/usr/bin/env python3
"""depth2_search.py - Search depth-2 combinations of inferred candidates."""

import sys
import json
import os
import numpy as np
import itertools

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
    
    # Create a new node with prog1 as base and prog2 as child
    new_node = ProgramNode(prog1.element)
    new_node.children = list(prog1.children)
    new_node.children.append(prog2)
    new_node.combo_type = "chain"
    new_node.total_mdl = new_node.compute_mdl()
    return new_node


def solve_task_depth2(task_data):
    """Solve a task using depth-2 search."""
    # Build demo pairs
    train = task_data.get('train', [])
    demo_pairs = []
    for pair in train:
        inp = [np.array(f, dtype=np.int8) for f in pair.get('input', [])]
        out = [np.array(f, dtype=np.int8) for f in pair.get('output', [])]
        demo_pairs.append({'input': inp, 'output': out})
    
    if not demo_pairs:
        return None
    
    # Get inferred candidates
    pi = ParamInference()
    candidates = pi.infer_candidates(demo_pairs)
    
    print(f'  Inferred candidates: {len(candidates)}')
    
    # Try depth-1 first
    for prog in candidates:
        if verify_program(prog, demo_pairs):
            return prog, 1
    
    # Try depth-2: chain candidates
    print(f'  Trying depth-2 combinations...')
    count = 0
    for c1 in candidates:
        for c2 in candidates:
            count += 1
            chained = chain_programs(c1, c2)
            if verify_program(chained, demo_pairs):
                return chained, 2
            if count > 50000:  # Limit to prevent too long search
                break
        if count > 50000:
            break
    
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
        result = solve_task_depth2(task_data)
        
        if result[0] is not None:
            prog, depth = result
            # Apply to test
            test_inp = np.array(task_data['test'][0]['input'][0], dtype=np.int8)
            pred = prog.apply(test_inp)
            
            if np.array_equal(pred, test_output):
                correct += 1
                correct_tasks.append(tid)
                print(f'  ✓ SOLVED (depth={depth}): {prog.element.name if prog.element else "chain"}')
    
    print(f'\n=== Depth-2 Results ===')
    print(f'Correct: {correct}/{total} ({correct/total*100:.1f}%)')
    print(f'Correct tasks: {correct_tasks}')
