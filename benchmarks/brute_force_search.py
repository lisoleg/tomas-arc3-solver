#!/usr/bin/env python3
"""brute_force_search.py - Try ALL depth-1 programs to find ones that match training examples."""

import sys
import json
import os
import numpy as np

sys.path.insert(0, '.')

from src.core.dsl_primitives import get_all_primitives, DSLElement, ProgramNode

def gen_all_depth1():
    """Generate all depth-1 programs with common param values."""
    programs = []
    all_prims = get_all_primitives()
    
    for prim in all_prims:
        name = prim.name
        
        if name == 'color-swap':
            for a in range(10):
                for b in range(a + 1, 10):
                    programs.append(ProgramNode(DSLElement('color-swap', {'color_a': a, 'color_b': b})))
        
        elif name == 'map-color':
            # Try single mappings
            for old in range(10):
                for new in range(10):
                    if old != new:
                        programs.append(ProgramNode(DSLElement('map-color', {'mapping': {old: new}})))
        
        elif name == 'multi-swap':
            # Try heuristic pairs
            programs.append(ProgramNode(DSLElement('multi-swap', {'swap_pairs': [[1,5],[2,6],[3,7],[4,8]]})))
            programs.append(ProgramNode(DSLElement('multi-swap', {'swap_pairs': [[0,9],[1,5],[2,6],[3,7],[4,8]]})))
        
        elif name == 'mirror':
            for axis in ['horizontal', 'vertical', 'both']:
                programs.append(ProgramNode(DSLElement('mirror', {'axis': axis})))
        
        elif name == 'rotate':
            for angle in [90, 180, 270]:
                programs.append(ProgramNode(DSLElement('rotate', {'angle': angle})))
        
        elif name == 'scale':
            for factor in [2, 3, 4]:
                programs.append(ProgramNode(DSLElement('scale', {'factor': factor})))
        
        elif name == 'tile':
            for fh in [2, 3]:
                for fw in [2, 3]:
                    programs.append(ProgramNode(DSLElement('tile', {'factor_h': fh, 'factor_w': fw})))
        
        elif name == 'move':
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    programs.append(ProgramNode(DSLElement('move', {'dx': dx, 'dy': dy})))
        
        elif name == 'crop':
            for h in [3, 5, 7, 9]:
                for w in [3, 5, 7, 9]:
                    programs.append(ProgramNode(DSLElement('crop', {'height': h, 'width': w})))
        
        else:
            # Default params
            programs.append(ProgramNode(DSLElement(name)))
    
    return programs


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


def solve_task(task_data, programs):
    """Try all programs and return the first one that matches."""
    # Build demo pairs
    train = task_data.get('train', [])
    demo_pairs = []
    for pair in train:
        inp = [np.array(f, dtype=np.int8) for f in pair.get('input', [])]
        out = [np.array(f, dtype=np.int8) for f in pair.get('output', [])]
        demo_pairs.append({'input': inp, 'output': out})
    
    # Try all programs
    for i, prog in enumerate(programs):
        if verify_program(prog, demo_pairs):
            return prog, i
    
    return None, -1


if __name__ == '__main__':
    # Generate all programs
    print('Generating all depth-1 programs...')
    programs = gen_all_depth1()
    print(f'Total programs: {len(programs)}')
    
    # Test on all real ARC tasks
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
        
        # Solve
        prog, idx = solve_task(task_data, programs)
        
        if prog is not None:
            # Apply to test
            test_inp = np.array(task_data['test'][0]['input'][0], dtype=np.int8)
            pred = prog.apply(test_inp)
            
            if np.array_equal(pred, test_output):
                correct += 1
                correct_tasks.append(tid)
                print(f'  ✓ {tid}: {prog.element.name if prog.element else "unknown"} (idx={idx})')
    
    print(f'\n=== Brute-force depth-1 results ===')
    print(f'Correct: {correct}/{total} ({correct/total*100:.1f}%)')
    print(f'Correct tasks: {correct_tasks}')
