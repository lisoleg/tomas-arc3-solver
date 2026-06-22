#!/usr/bin/env python3
"""smart_depth2_search.py - Beam Search depth-2 with MDL ranking."""
import sys, json, os
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


def solve_task_smart(task_data, max_candidates=50, max_combos=50000):
    """Solve a task using smart depth-2 search (MDL-ranked beam search)."""
    train = task_data.get('train', [])
    demo_pairs = []
    for pair in train:
        inp = [np.array(f, dtype=np.int8) for f in pair.get('input', [])]
        out = [np.array(f, dtype=np.int8) for f in pair.get('output', [])]
        demo_pairs.append({'input': inp, 'output': out})
    
    if not demo_pairs:
        return None, 0, []
    
    # Get inferred candidates
    pi = ParamInference()
    candidates = pi.infer_candidates(demo_pairs)
    
    print(f'  Inferred candidates: {len(candidates)}')
    
    # Sort candidates by MDL (lower = better)
    candidates.sort(key=lambda c: c.total_mdl if c.total_mdl > 0 else 999999)
    
    # Limit to top-N candidates
    top_candidates = candidates[:max_candidates]
    print(f'  Top {len(top_candidates)} candidates (MDL-ranked)')
    
    # Try depth-1 first
    for prog in top_candidates:
        if verify_program(prog, demo_pairs):
            return prog, 1, [prog]
    
    # Try depth-2: chain candidates (prioritize low MDL combinations)
    print(f'  Trying depth-2 combinations (beam search)...')
    count = 0
    results = []
    
    for i, c1 in enumerate(top_candidates):
        for j, c2 in enumerate(top_candidates):
            count += 1
            chained = chain_programs(c1, c2)
            if verify_program(chained, demo_pairs):
                return chained, 2, [c1, c2]
            if count > max_combos:
                break
        if count > max_combos:
            break
    
    return None, 0, []


if __name__ == '__main__':
    d = 'data/real_arc_converted'
    tasks = sorted([f for f in os.listdir(d) if f.endswith('.json')])
    
    correct = 0
    total = 0
    correct_tasks = []
    solved_by_depth = {1: 0, 2: 0}
    
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
        prog, depth, chain = solve_task_smart(task_data)
        
        if prog is not None:
            test_inp = np.array(task_data['test'][0]['input'][0], dtype=np.int8)
            pred = prog.apply(test_inp)
            
            if np.array_equal(pred, test_output):
                correct += 1
                correct_tasks.append(tid)
                if depth in solved_by_depth:
                    solved_by_depth[depth] += 1
                chain_names = [p.element.name if p.element else 'chain' for p in chain]
                print(f'  ✓ SOLVED (depth={depth}): {" -> ".join(chain_names)}')
    
    print(f'\n=== Smart Depth-2 Results ===')
    print(f'Correct: {correct}/{total} ({correct/total*100:.1f}%)')
    print(f'Solved by depth: {solved_by_depth}')
    print(f'Correct tasks: {correct_tasks}')
