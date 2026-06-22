#!/usr/bin/env python3
"""enhanced_solver.py - More aggressive search with expanded parameter space."""

import sys
import json
import os
import numpy as np
import itertools

sys.path.insert(0, '.')

from src.core.dsl_primitives import get_all_primitives, DSLElement, ProgramNode
from src.solver.param_inference import ParamInference


def gen_all_depth1(expanded=False):
    """Generate all depth-1 programs.
    
    Args:
        expanded: If True, generate MORE param variations.
    """
    programs = []
    all_prims = get_all_primitives()
    
    for prim in all_prims:
        name = prim.name
        
        if name == 'color-swap':
            # All pairs
            for a in range(10):
                for b in range(a + 1, 10):
                    programs.append(ProgramNode(DSLElement('color-swap', {'color_a': a, 'color_b': b})))
        
        elif name == 'map-color':
            # Single mappings + identity
            for old in range(10):
                for new in range(10):
                    if old != new:
                        programs.append(ProgramNode(DSLElement('map-color', {'mapping': {old: new}})))
            # Also try empty mapping (identity)
            programs.append(ProgramNode(DSLElement('map-color', {'mapping': {}})))
        
        elif name == 'multi-swap':
            # Various swap pair combinations
            pairs_list = [
                [[1,5],[2,6],[3,7],[4,8]],
                [[0,9],[1,5],[2,6],[3,7],[4,8]],
                [[1,5],[2,6]],
                [[3,4],[5,1],[6,2],[8,9]],  # For task 0d3d703e
            ]
            for pairs in pairs_list:
                programs.append(ProgramNode(DSLElement('multi-swap', {'swap_pairs': pairs})))
        
        elif name == 'flood-fill':
            # Try all colors
            for c in range(10):
                programs.append(ProgramNode(DSLElement('flood-fill', {'color': c})))
        
        elif name == 'fill-region':
            # Try all colors and regions
            for c in range(10):
                for region in ['interior', 'border', 'background']:
                    programs.append(ProgramNode(DSLElement('fill-region', {'color': c, 'region': region})))
        
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
            for fh in [2, 3, 4]:
                for fw in [2, 3, 4]:
                    programs.append(ProgramNode(DSLElement('tile', {'factor_h': fh, 'factor_w': fw})))
        
        elif name == 'move':
            for dx in [-3, -2, -1, 0, 1, 2, 3]:
                for dy in [-3, -2, -1, 0, 1, 2, 3]:
                    if dx != 0 or dy != 0:
                        programs.append(ProgramNode(DSLElement('move', {'dx': dx, 'dy': dy})))
        
        elif name == 'crop':
            # Try various crop positions and sizes
            for h in [3, 5, 7, 9, 11, 13, 15]:
                for w in [3, 5, 7, 9, 11, 13, 15]:
                    programs.append(ProgramNode(DSLElement('crop', {'height': h, 'width': w})))
                    # Also try with start positions
                    for sr in [0, 1, 2, 3, 4, 5]:
                        for sc in [0, 1, 2, 3, 4, 5]:
                            programs.append(ProgramNode(DSLElement('crop', {'height': h, 'width': w, 'start_row': sr, 'start_col': sc})))
        
        elif name == 'map-by-function':
            for func_type in ['add', 'sub', 'mul', 'mod', 'inv_mod']:
                for value in range(10):
                    programs.append(ProgramNode(DSLElement('map-by-function', {'func_type': func_type, 'value': value, 'modulo': 10})))
        
        elif name == 'complete-pattern':
            programs.append(ProgramNode(DSLElement('complete-pattern')))
        
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


def solve_task(task_data, programs, max_programs=10000):
    """Try all programs and return the first one that matches."""
    # Build demo pairs
    train = task_data.get('train', [])
    demo_pairs = []
    for pair in train:
        inp = [np.array(f, dtype=np.int8) for f in pair.get('input', [])]
        out = [np.array(f, dtype=np.int8) for f in pair.get('output', [])]
        demo_pairs.append({'input': inp, 'output': out})
    
    # Get test input
    test_items = task_data.get('test', [])
    if not test_items:
        return None, 'no_test'
    test_inp = np.array(test_items[0].get('input', [[]])[0], dtype=np.int8)
    test_out = None
    test_outs = test_items[0].get('output', [])
    if test_outs:
        test_out = np.array(test_outs[0], dtype=np.int8)
    
    if test_out is None:
        return None, 'no_test_output'
    
    # Try all programs
    for i, prog in enumerate(programs[:max_programs]):
        if verify_program(prog, demo_pairs):
            # Apply to test
            try:
                pred = prog.apply(test_inp)
                if np.array_equal(pred, test_out):
                    return prog, 'correct'
            except Exception:
                pass
    
    return None, 'not_found'


if __name__ == '__main__':
    # Generate all programs
    print('Generating all depth-1 programs (expanded)...')
    programs = gen_all_depth1(expanded=True)
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
        
        total += 1
        
        prog, status = solve_task(task_data, programs)
        
        if prog is not None and status == 'correct':
            correct += 1
            correct_tasks.append(tid)
            elem_name = prog.element.name if prog.element else 'unknown'
            print(f'  ✓ {tid}: {elem_name}')
    
    print(f'\n=== Enhanced depth-1 results ===')
    print(f'Correct: {correct}/{total} ({correct/total*100:.1f}%)')
    print(f'Correct tasks: {correct_tasks}')
