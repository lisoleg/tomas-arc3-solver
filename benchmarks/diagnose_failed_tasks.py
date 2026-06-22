"""
diagnose_failed_tasks.py — 诊断失败任务，找出缺失的原语类型。
对每个任务：
1. 列出所有推断候选
2. 尝试 depth-1 匹配
3. 如果失败，输出输入→输出的变换描述
4. 保存结果到 JSON
"""
import json, sys, time, os
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, 'src')
from core.dsl_primitives import DSLElement, ProgramNode, chain_programs
from solver.param_inference import ParamInference

TASKS_FILE = 'data/real_arc_benchmark_v2.5.json'
CONVERTED_DIR = 'data/real_arc_converted'


def grid_to_text(g: np.ndarray) -> str:
    """把网格转成简短文本描述。"""
    if g.ndim == 3:
        g = g[0]
    colors = sorted(np.unique(g))
    shape = g.shape
    non_zero = np.sum(g != 0)
    return f"shape={shape}, colors={colors}, non_zero={non_zero}"


def verify_program(prog: ProgramNode, demo_pairs: list) -> bool:
    """验证程序是否匹配所有训练对。"""
    try:
        for pair in demo_pairs:
            inputs = pair['input']
            expected_outputs = pair['output']
            current_inputs = inputs
            # 应用程序链
            if hasattr(prog, 'left') and hasattr(prog, 'right'):
                # 是链式 ProgramNode
                from solver.param_inference import _apply_program
                output = _apply_program(prog, inputs[0])
            else:
                output = prog.apply(inputs[0])
            expected = expected_outputs[0]
            if output.shape != expected.shape:
                return False
            if not np.array_equal(output, expected):
                return False
        return True
    except Exception:
        return False


def diagnose_task(task_id: str) -> dict:
    """诊断单个任务。"""
    result = {
        'task_id': task_id,
        'solved_depth1': False,
        'solved_depth2': False,
        'best_program': None,
        'input_desc': '',
        'output_desc': '',
        'transform_hint': '',
        'num_candidates': 0,
    }

    try:
        with open(f'{CONVERTED_DIR}/{task_id}.json') as f:
            task = json.load(f)
    except Exception as e:
        result['error'] = str(e)
        return result

    train = task.get('train', [])
    if not train:
        result['error'] = 'no training pairs'
        return result

    demo_pairs = []
    for pair in train:
        inp = [np.array(f, dtype=np.int8) for f in pair.get('input', [])]
        out = [np.array(f, dtype=np.int8) for f in pair.get('output', [])]
        demo_pairs.append({'input': inp, 'output': out})

    # 描述输入输出
    inp0 = demo_pairs[0]['input'][0]
    out0 = demo_pairs[0]['output'][0]
    result['input_desc'] = grid_to_text(inp0)
    result['output_desc'] = grid_to_text(out0)

    # 推断候选
    pi = ParamInference()
    candidates = pi.infer_candidates(demo_pairs)
    result['num_candidates'] = len(candidates)

    # 尝试 depth-1
    for prog in candidates:
        if verify_program(prog, demo_pairs):
            result['solved_depth1'] = True
            result['best_program'] = str(prog)
            result['solved_depth'] = 1
            return result

    # 尝试 depth-2（最多 2000 个组合，快速诊断）
    count = 0
    for i, c1 in enumerate(candidates[:30]):  # 只取前30个候选
        for j, c2 in enumerate(candidates[:30]):
            count += 1
            if count > 2000:
                break
            chained = chain_programs(c1, c2)
            if verify_program(chained, demo_pairs):
                result['solved_depth2'] = True
                result['best_program'] = str(chained)
                result['solved_depth'] = 2
                return result
        if count > 2000:
            break

    # 都没解决 → 生成变换提示
    result['transform_hint'] = analyze_transform(inp0, out0)
    return result


def analyze_transform(inp: np.ndarray, out: np.ndarray) -> str:
    """分析输入→输出的变换，给出提示。"""
    hints = []

    # 形状变化
    if inp.ndim == 3:
        inp = inp[0]
    if out.ndim == 3:
        out = out[0]

    if inp.shape != out.shape:
        hints.append(f'shape_change: {inp.shape} -> {out.shape}')

    # 颜色变化
    inp_colors = set(np.unique(inp))
    out_colors = set(np.unique(out))
    if inp_colors != out_colors:
        hints.append(f'color_change: {sorted(inp_colors)} -> {sorted(out_colors)}')

    # 对象数量变化（粗略）
    inp_obj = np.sum(inp != 0)
    out_obj = np.sum(out != 0)
    if abs(inp_obj - out_obj) > 3:
        hints.append(f'pixel_count: {inp_obj} -> {out_obj}')

    return '; '.join(hints) if hints else 'unknown_transform'


def main():
    with open(TASKS_FILE) as f:
        data = json.load(f)

    results = []
    for i, t in enumerate(data):
        tid = t['task']
        print(f'[{i+1}/50] Diagnosing {tid}...')
        r = diagnose_task(tid)
        results.append(r)
        status = '✅' if r.get('solved_depth1') or r.get('solved_depth2') else '❌'
        print(f'  {status} depth1={r.get("solved_depth1")}, depth2={r.get("solved_depth2")}, candidates={r.get("num_candidates")}')
        if not r.get('solved_depth1') and not r.get('solved_depth2'):
            print(f'    Hint: {r.get("transform_hint", "")}')

    # 保存结果
    with open('data/diagnose_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总
    solved = sum(1 for r in results if r.get('solved_depth1') or r.get('solved_depth2'))
    depth1 = sum(1 for r in results if r.get('solved_depth1'))
    depth2 = sum(1 for r in results if r.get('solved_depth2') and not r.get('solved_depth1'))
    print(f'\n=== Summary ===')
    print(f'Solved: {solved}/50 ({solved*2}%)')
    print(f'  Depth-1: {depth1}')
    print(f'  Depth-2: {depth2}')
    print(f'Results saved to data/diagnose_results.json')


if __name__ == '__main__':
    main()
