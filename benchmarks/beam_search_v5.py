"""
beam_search_v5.py — 改进版Beam Search (depth-3+输出相似度引导)

核心改进：
1. 结合MDL分数和输出相似度对候选排序
2. 更大的Beam Width（100）- 提高覆盖率
3. 输出相似度引导的搜索剪枝
4. 支持depth-1/2/3/4搜索
5. 早停：找到解立即返回

预期：准确率从18% → 50%+（更好的搜索引导）
"""
import json
import os
import sys
import time
from typing import Any

import numpy as np

# 正确设置Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from src.core.dsl_primitives import DSLElement, ProgramNode
from src.solver.param_inference import ParamInference


TASKS_FILE = 'data/real_arc_benchmark_v2.5.json'
CONVERTED_DIR = 'data/real_arc_converted'
BEAM_WIDTH = 100   # 增加Beam Width以提高覆盖率
MAX_DEPTH = 4       # 支持更深搜索
SIMILARITY_WEIGHT = 0.3  # 输出相似度的权重


def grid_similarity(g1: np.ndarray, g2: np.ndarray) -> float:
    """计算两个网格的相似度（0-1，1=完全相同）。"""
    if g1.shape != g2.shape:
        # 形状不同，计算重叠相似度
        h = min(g1.shape[0], g2.shape[0])
        w = min(g1.shape[1], g2.shape[1])
        if h == 0 or w == 0:
            return 0.0
        g1_crop = g1[:h, :w]
        g2_crop = g2[:h, :w]
        total = g1_crop.size
        if total == 0:
            return 0.0
        match = np.sum(g1_crop == g2_crop)
        return match / total
    
    total = g1.size
    if total == 0:
        return 1.0
    match = np.sum(g1 == g2)
    return match / total


def verify_program(prog: ProgramNode, demo_pairs: list) -> tuple:
    """验证程序是否匹配所有训练对，返回(是否匹配, 平均输出相似度)。"""
    try:
        total_sim = 0.0
        count = 0
        for pair in demo_pairs:
            for i, inp in enumerate(pair['input']):
                if i < len(pair['output']):
                    expected = pair['output'][i]
                    pred = prog.apply(inp)
                    total_sim += grid_similarity(pred, expected)
                    count += 1
                    # 如果形状不匹配，提前退出
                    if pred.shape != expected.shape:
                        return False, 0.0
        
        avg_sim = total_sim / max(count, 1)
        
        # 二次验证：是否完全匹配
        for pair in demo_pairs:
            for i, inp in enumerate(pair['input']):
                if i < len(pair['output']):
                    expected = pair['output'][i]
                    pred = prog.apply(inp)
                    if not np.array_equal(pred, expected):
                        return False, avg_sim
        
        return True, avg_sim
    except Exception:
        return False, 0.0


def chain_programs(prog1: ProgramNode, prog2: ProgramNode) -> ProgramNode:
    """正确地链式组合两个程序。"""
    if prog1.element is None and not prog1.children:
        return prog2
    if prog2.element is None and not prog2.children:
        return prog1
    
    # 创建新的ProgramNode，prog1作为基础，prog2作为子节点
    new_node = ProgramNode(prog1.element)
    new_node.children = list(prog1.children)
    new_node.children.append(prog2)
    new_node.combo_type = "chain"
    new_node.total_mdl = new_node.compute_mdl()
    return new_node


def compute_candidate_score(prog: ProgramNode, demo_pairs: list) -> float:
    """计算候选的排序分数（越低越好）。"""
    # MDL分数（越低越好）
    mdl = prog.total_mdl if hasattr(prog, 'total_mdl') else prog.mdl_cost
    
    # 计算输出相似度
    _, avg_sim = verify_program(prog, demo_pairs)
    
    # 综合分数 = MDL + (1 - similarity) * weight
    score = mdl + (1.0 - avg_sim) * SIMILARITY_WEIGHT * 100
    
    return score


def solve_task_beam_v5(task_data: dict) -> tuple:
    """用改进的Beam Search解决任务。"""
    # 构建demo pairs
    train = task_data.get('train', [])
    demo_pairs = []
    for pair in train:
        inp = [np.array(f, dtype=np.int8) for f in pair.get('input', [])]
        out = [np.array(f, dtype=np.int8) for f in pair.get('output', [])]
        demo_pairs.append({'input': inp, 'output': out})
    
    if not demo_pairs:
        return None, 0
    
    # 获取推断的候选
    pi = ParamInference()
    candidates = pi.infer_candidates(demo_pairs)
    
    if not candidates:
        return None, 0
    
    print(f'  Inferred candidates: {len(candidates)}')
    
    # ========= Depth-1 =========
    print(f'  Trying depth-1...')
    best_prog = None
    best_score = float('inf')
    
    for prog in candidates:
        ok, sim = verify_program(prog, demo_pairs)
        if ok:
            score = compute_candidate_score(prog, demo_pairs)
            if score < best_score:
                best_prog = prog
                best_score = score
    
    if best_prog is not None:
        return best_prog, 1
    
    # ========= Beam Search Depth-2 =========
    print(f'  Trying depth-2 (beam width={BEAM_WIDTH})...')
    
    # 按综合分数排序候选
    candidates_with_score = [(c, compute_candidate_score(c, demo_pairs)) for c in candidates]
    candidates_with_score.sort(key=lambda x: x[1])
    beam = [c for c, _ in candidates_with_score[:BEAM_WIDTH]]
    
    best_prog = None
    best_score = float('inf')
    
    # 尝试beam内的所有组合
    tried = 0
    for i, c1 in enumerate(beam):
        for j, c2 in enumerate(beam):
            tried += 1
            chained = chain_programs(c1, c2)
            ok, sim = verify_program(chained, demo_pairs)
            if ok:
                score = compute_candidate_score(chained, demo_pairs)
                if score < best_score:
                    best_prog = chained
                    best_score = score
    
    if best_prog is not None:
        return best_prog, 2
    
    # ========= Beam Search Depth-3 =========
    if MAX_DEPTH >= 3:
        print(f'  Trying depth-3 (beam width={BEAM_WIDTH})...')
        
        for i, c1 in enumerate(beam):
            for j, c2 in enumerate(beam):
                for k, c3 in enumerate(beam):
                    tried += 1
                    # 创建3层链式：c1 -> c2 -> c3
                    chained_12 = chain_programs(c1, c2)
                    chained_123 = chain_programs(chained_12, c3)
                    
                    ok, sim = verify_program(chained_123, demo_pairs)
                    if ok:
                        score = compute_candidate_score(chained_123, demo_pairs)
                        if score < best_score:
                            best_prog = chained_123
                            best_score = score
        
        if best_prog is not None:
            return best_prog, 3
    
    # ========= Beam Search Depth-4 =========
    if MAX_DEPTH >= 4:
        print(f'  Trying depth-4 (beam width={BEAM_WIDTH})...')
        
        for i, c1 in enumerate(beam[:20]):  # 只用Top-20做depth-4（避免组合爆炸）
            for j, c2 in enumerate(beam[:20]):
                for k, c3 in enumerate(beam[:20]):
                    for l, c4 in enumerate(beam[:20]):
                        tried += 1
                        chained_12 = chain_programs(c1, c2)
                        chained_123 = chain_programs(chained_12, c3)
                        chained_1234 = chain_programs(chained_123, c4)
                        
                        ok, sim = verify_program(chained_1234, demo_pairs)
                        if ok:
                            score = compute_candidate_score(chained_1234, demo_pairs)
                            if score < best_score:
                                best_prog = chained_1234
                                best_score = score
        
        if best_prog is not None:
            return best_prog, 4
    
    print(f'  Tried {tried} combinations, no solution found.')
    return None, 0


def main():
    """主函数：运行基准测试。"""
    # 读取任务列表
    tasks_dir = CONVERTED_DIR
    tasks = sorted([f for f in os.listdir(tasks_dir) if f.endswith('.json')])
    
    correct = 0
    total = 0
    correct_tasks = []
    details = []
    
    start_time = time.time()
    
    for i, tf in enumerate(tasks):
        tid = tf.replace('.json', '')
        
        # 读取任务数据
        try:
            with open(os.path.join(tasks_dir, tf)) as f:
                task_data = json.load(f)
        except Exception as e:
            print(f'[{i+1}/50] {tid}... Error loading: {e}')
            continue
        
        # 获取测试输出
        test_items = task_data.get('test', [])
        test_output = None
        if test_items:
            test_outs = test_items[0].get('output', [])
            if test_outs:
                test_output = np.array(test_outs[0], dtype=np.int8)
        
        if test_output is None:
            continue
        
        total += 1
        print(f'[{i+1}/50] {tid}...')
        
        # 解决任务
        result = solve_task_beam_v5(task_data)
        
        if result[0] is not None:
            prog, depth = result
            # 应用到测试输入
            test_inp = np.array(task_data['test'][0]['input'][0], dtype=np.int8)
            pred = prog.apply(test_inp)
            
            if np.array_equal(pred, test_output):
                correct += 1
                correct_tasks.append(tid)
                print(f'  ✓ SOLVED (depth={depth}): {prog.element.name if prog.element else "chain"}')
            else:
                print(f'  ✗ Incorrect')
        else:
            print(f'  ✗ FAILED')
    
    elapsed = time.time() - start_time
    print(f'\n=== Beam Search V5 Results ===')
    print(f'Correct: {correct}/{total} ({correct/total*100:.1f}%)')
    print(f'Solved tasks: {correct_tasks}')
    print(f'Time elapsed: {elapsed:.1f}s')
    
    # 保存详细结果
    with open('data/beam_search_v5_results.json', 'w') as f:
        json.dump(details, f, indent=2)


if __name__ == '__main__':
    main()
