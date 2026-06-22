"""
TOMAS Solver Runner - 求解任务执行引擎

在后台线程中运行 TOMAS 求解器，通过 SSE 推送实时进度。
支持多种推理模式（video, bayesian, fusion, auto）。

v2.4.1: 真实 solver 调用 + 运行时数据采集。
"""

import sys
import os
import threading
import logging
import time
import json
from typing import Dict, Optional, Callable
from datetime import datetime

from services.sse_manager import create_sse_queue, push_event, close_sse_queue

logger = logging.getLogger(__name__)


def generate_task_id() -> str:
    """生成唯一任务 ID。"""
    import uuid
    return str(uuid.uuid4())[:8]


# 任务状态存储
_active_tasks: Dict[str, dict] = {}


def get_task_status(task_id: str) -> Optional[dict]:
    """获取任务状态。"""
    return _active_tasks.get(task_id)


def run_solver(task_id: str, params: dict):
    """
    在后台线程中运行 TOMAS 求解器，通过 SSE 推送实时进度。
    捕获真实运行时数据供可视化使用。

    Args:
        task_id: 任务 ID
        params: 求解参数
            - input_path: 输入文件路径或目录
            - output_path: 输出文件路径
            - mode: 推理模式 (video/bayesian/fusion/auto)
            - config_overrides: 配置覆盖 (可选)
    """
    import sys
    import os
    import json as _json

    # 确保项目根目录在 sys.path 中
    project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    project_root = os.path.abspath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        # 推送开始事件
        push_event(task_id, {
            'type': 'started',
            'task_id': task_id,
            'params': params,
            'timestamp': datetime.now().isoformat(),
        })

        # 导入 TOMAS 求解器
        from src.solver.tomas_solver import TOMASSolver
        from src.utils.config import load_config

        # 加载配置
        config_path = os.path.join(project_root, 'config', 'default.yaml')
        config = load_config(config_path)

        # 应用配置覆盖
        if 'config_overrides' in params:
            for key, value in params['config_overrides'].items():
                config[key] = value

        # 创建求解器
        solver = TOMASSolver(config)

        # 推送初始化事件
        push_event(task_id, {
            'type': 'init',
            'message': 'TOMAS Solver initialized',
            'timestamp': datetime.now().isoformat(),
        })

        # 加载任务数据
        input_path = params.get('input_path', '')
        output_path = params.get('output_path', 'output/result.json')
        mode = params.get('mode', 'auto')

        # 读取任务 JSON
        task_data = None
        if input_path and os.path.exists(input_path):
            with open(input_path, 'r') as f:
                task_data = _json.load(f)
        elif input_path:
            # Try relative to project root
            full_path = os.path.join(project_root, input_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    task_data = _json.load(f)

        if task_data is None:
            raise FileNotFoundError(f"Task file not found: {input_path}")

        # 推送编码阶段事件
        push_event(task_id, {
            'type': 'progress',
            'task_id': task_id,
            'step': 1,
            'total': 5,
            'progress': 20.0,
            'message': 'Encoding input grids to octonion hyperedges',
            'timestamp': datetime.now().isoformat(),
        })

        # 解析输入
        video_task = solver.parse_input(task_data)
        demo_pairs = video_task.demo_pairs

        # 推送搜索阶段事件
        push_event(task_id, {
            'type': 'progress',
            'task_id': task_id,
            'step': 2,
            'total': 5,
            'progress': 40.0,
            'message': f'Running kappa-Snap search ({mode} mode)',
            'timestamp': datetime.now().isoformat(),
        })

        # 运行求解
        start_time = time.time()
        result = solver.solve(task_data, mode=mode)
        duration_sec = time.time() - start_time

        # 推送验证阶段事件
        push_event(task_id, {
            'type': 'progress',
            'task_id': task_id,
            'step': 3,
            'total': 5,
            'progress': 60.0,
            'message': 'GaussEx fiber verification',
            'timestamp': datetime.now().isoformat(),
        })

        # 推送融合阶段事件
        push_event(task_id, {
            'type': 'progress',
            'task_id': task_id,
            'step': 4,
            'total': 5,
            'progress': 80.0,
            'message': 'Bayesian confidence ranking',
            'timestamp': datetime.now().isoformat(),
        })

        # 捕获运行时数据供可视化使用
        try:
            from services.runtime_collector import get_collector
            collector = get_collector()
            collector.capture_result(
                task_id=task_id,
                solver=solver,
                result=result,
                demo_pairs=demo_pairs,
                mode=mode,
                duration_sec=duration_sec,
            )
        except Exception as capture_err:
            logger.warning(f"Runtime data capture failed: {capture_err}")

        # 保存输出
        output_full = os.path.join(project_root, output_path)
        os.makedirs(os.path.dirname(output_full) or '.', exist_ok=True)
        with open(output_full, 'w') as f:
            _json.dump({
                'task_id': task_id,
                'mode': mode,
                'result': {
                    'predictions': result.get('predictions', []),
                    'best_program_mdl': result.get('best_program_mdl', 0),
                    'best_posterior': result.get('best_posterior', result.get('best_fusion_score', 0.0)),
                    'num_valid_programs': result.get('num_valid_programs', 0),
                },
                'duration_sec': round(duration_sec, 4),
            }, f, indent=2, ensure_ascii=False, default=str)

        # 推送完成事件
        result_summary = {
            'task_id': task_id,
            'status': 'success',
            'output_path': output_path,
            'mode': mode,
            'duration_sec': round(duration_sec, 4),
            'predictions_count': len(result.get('predictions', [])),
            'best_program_mdl': result.get('best_program_mdl', 0),
            'num_valid_programs': result.get('num_valid_programs', 0),
            'message': 'Solving completed successfully',
        }

        # 保存历史
        _save_history(task_id, params, result_summary)

        push_event(task_id, {
            'type': 'done',
            'task_id': task_id,
            'result': result_summary,
            'timestamp': datetime.now().isoformat(),
        })

        # 更新任务状态
        _active_tasks[task_id]['status'] = 'completed'
        _active_tasks[task_id]['result'] = result_summary

        logger.info(f"Task {task_id} completed successfully in {duration_sec:.2f}s")

    except Exception as e:
        logger.error(f"Task {task_id} failed: {e}", exc_info=True)

        error_result = {
            'task_id': task_id,
            'status': 'error',
            'error': str(e),
        }

        push_event(task_id, {
            'type': 'done',
            'task_id': task_id,
            'result': error_result,
            'timestamp': datetime.now().isoformat(),
        })

        _active_tasks[task_id]['status'] = 'failed'
        _active_tasks[task_id]['error'] = str(e)

    finally:
        close_sse_queue(task_id)


def start_solver(params: dict) -> dict:
    """
    启动求解任务。

    Args:
        params: 求解参数

    Returns:
        dict: {task_id, status, message}
    """
    task_id = generate_task_id()

    # 注册任务
    _active_tasks[task_id] = {
        'task_id': task_id,
        'params': params,
        'status': 'running',
        'created_at': datetime.now().isoformat(),
    }

    # 创建 SSE 队列
    create_sse_queue(task_id)

    # 在后台线程中运行
    thread = threading.Thread(
        target=run_solver,
        args=(task_id, params),
        daemon=True,
    )
    thread.start()

    return {
        'task_id': task_id,
        'status': 'started',
        'message': 'Solver task started',
    }


def _save_history(task_id: str, params: dict, result: dict):
    """保存任务历史到 JSON 文件。"""
    import json
    from config import HISTORY_FILE

    history_entry = {
        'task_id': task_id,
        'params': params,
        'result': result,
        'timestamp': datetime.now().isoformat(),
    }

    try:
        # 读取现有历史
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        else:
            history = []

        # 添加新条目
        history.append(history_entry)

        # 写回文件
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        logger.info(f"History saved for task {task_id}")

    except Exception as e:
        logger.error(f"Failed to save history: {e}")
