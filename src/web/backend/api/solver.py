"""
TOMAS Solver API

路由:
- GET  /api/solver/modes           获取所有可用推理模式
- POST /api/solver/run             运行求解任务
- GET  /api/solver/status/<id>     查询任务状态
- GET  /api/solver/progress/<id>   SSE 进度流
- GET  /api/solver/history         获取历史列表
- GET  /api/solver/history/<id>    获取历史详情
- DELETE /api/solver/history/<id>  删除历史
"""

import json
from flask import Blueprint, jsonify, request, Response
from services.solver_runner import (
    get_task_status, start_solver,
)
from services.sse_manager import generate_sse_stream

solver_bp = Blueprint('solver', __name__)


@solver_bp.route('/modes', methods=['GET'])
def get_modes():
    """获取所有可用推理模式。"""
    try:
        modes = [
            {'id': 'video', 'name': 'Video Mode', 'description': 'Pure symbolic reasoning, fast (<40s)'},
            {'id': 'bayesian', 'name': 'Bayesian Mode', 'description': 'Bayesian posterior ranking, medium (40-80s)'},
            {'id': 'fusion', 'name': 'Fusion Mode', 'description': 'Multi-modal fusion, slow (>80s), highest accuracy'},
            {'id': 'auto', 'name': 'Auto Mode', 'description': 'Time-budget adaptive selection'},
        ]
        return jsonify({'modes': modes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@solver_bp.route('/run', methods=['POST'])
def run_solver_endpoint():
    """
    运行求解任务。

    请求体: {input_path: str, output_path: str, mode: str, config_overrides: dict (可选)}
    响应: {task_id: str, status: str, message: str}
    """
    try:
        data = request.get_json(force=True)
        if not data or 'input_path' not in data:
            return jsonify({'error': 'Missing input_path'}), 400

        input_path = data['input_path']
        output_path = data.get('output_path', 'output/result.json')
        mode = data.get('mode', 'auto')
        config_overrides = data.get('config_overrides', {})

        result = start_solver({
            'input_path': input_path,
            'output_path': output_path,
            'mode': mode,
            'config_overrides': config_overrides,
        })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@solver_bp.route('/status/<task_id>', methods=['GET'])
def solver_status(task_id: str):
    """
    查询求解任务状态。

    响应: {task_id, status, params, result, error}
    """
    try:
        from services.solver_runner import _active_tasks
        status = _active_tasks.get(task_id)
        if status is None:
            return jsonify({'error': 'Task not found'}), 404
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@solver_bp.route('/progress/<task_id>', methods=['GET'])
def solver_progress(task_id: str):
    """
    SSE 进度流端点。

    返回 text/event-stream，实时推送求解进度。
    """
    from flask import Response
    return Response(
        generate_sse_stream(task_id),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',
        }
    )


@solver_bp.route('/history', methods=['GET'])
def get_history():
    """获取求解历史列表。"""
    try:
        import os
        from config import HISTORY_FILE
        
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            return jsonify({'history': history})
        else:
            return jsonify({'history': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@solver_bp.route('/history/<task_id>', methods=['DELETE'])
def delete_history(task_id: str):
    """删除指定历史条目。"""
    try:
        import os
        from config import HISTORY_FILE
        
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
            
            history = [h for h in history if h['task_id'] != task_id]
            
            with open(HISTORY_FILE, 'w') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            
            return jsonify({'status': 'deleted', 'task_id': task_id})
        else:
            return jsonify({'error': 'History file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
