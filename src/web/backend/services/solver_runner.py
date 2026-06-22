"""
TOMAS Solver Runner - 求解任务执行引擎

在后台线程中运行 TOMAS 求解器，通过 SSE 推送实时进度。
支持多种推理模式（video, bayesian, fusion, auto）。
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
    在后台线程中运行 TOMAS 求解器。

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
        
        # 运行求解
        input_path = params.get('input_path', '')
        output_path = params.get('output_path', 'output/result.json')
        mode = params.get('mode', 'auto')
        
        # TODO: 实际的求解逻辑
        # 这里需要实现进度回调，定期推送进度事件
        
        # 模拟求解过程 (临时实现)
        total_steps = 10
        for step in range(total_steps):
            time.sleep(0.5)  # 模拟计算
            
            push_event(task_id, {
                'type': 'progress',
                'task_id': task_id,
                'step': step + 1,
                'total': total_steps,
                'progress': (step + 1) / total_steps * 100,
                'message': f'Step {step + 1}/{total_steps}',
                'timestamp': datetime.now().isoformat(),
            })
        
        # 推送完成事件
        result = {
            'task_id': task_id,
            'status': 'success',
            'output_path': output_path,
            'message': 'Solving completed successfully',
        }
        
        # 保存历史
        _save_history(task_id, params, result)
        
        push_event(task_id, {
            'type': 'done',
            'task_id': task_id,
            'result': result,
            'timestamp': datetime.now().isoformat(),
        })
        
        # 更新任务状态
        _active_tasks[task_id]['status'] = 'completed'
        _active_tasks[task_id]['result'] = result
        
        logger.info(f"Task {task_id} completed successfully")
        
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
