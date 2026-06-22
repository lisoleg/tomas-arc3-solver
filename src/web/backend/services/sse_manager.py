"""
TOMAS Web Dashboard - SSE (Server-Sent Events) 连接管理器

管理前端与后端的 SSE 连接，用于实时推送求解进度。
每个活动任务维护一个队列，Flask 端点从队列读取事件并流式发送。
"""

import json
import logging
from queue import Queue, Empty
from typing import Dict

logger = logging.getLogger(__name__)

# 全局 SSE 连接注册表: task_id -> Queue
_active_connections: Dict[str, Queue] = {}


def create_sse_queue(task_id: str) -> Queue:
    """
    为指定 task_id 创建消息队列。

    Args:
        task_id: 任务 ID

    Returns:
        Queue: 线程安全的消息队列
    """
    q = Queue()
    _active_connections[task_id] = q
    logger.info(f"SSE queue created for task_id={task_id}")
    return q


def push_event(task_id: str, data: dict) -> bool:
    """
    向指定任务的 SSE 队列推送事件。

    Args:
        task_id: 任务 ID
        data: 事件数据字典

    Returns:
        bool: 是否推送成功
    """
    q = _active_connections.get(task_id)
    if q is None:
        logger.warning(f"No active SSE connection for task_id={task_id}")
        return False
    try:
        q.put_nowait(data)
        return True
    except Exception as e:
        logger.error(f"Failed to push SSE event: {e}")
        return False


def close_sse_queue(task_id: str):
    """
    关闭指定任务的 SSE 队列。

    Args:
        task_id: 任务 ID
    """
    q = _active_connections.pop(task_id, None)
    if q is not None:
        logger.info(f"SSE queue closed for task_id={task_id}")


def generate_sse_stream(task_id: str):
    """
    SSE 流生成器，供 Flask Response 使用。

    从队列中轮询读取事件，格式化为 SSE 数据流。
    当收到 {"type": "done"} 或客户端断开时结束。

    Args:
        task_id: 任务 ID

    Yields:
        str: SSE 格式的字符串
    """
    import time
    q = _active_connections.get(task_id)
    if q is None:
        yield f"data: {json.dumps({'error': 'invalid task_id'})}\n\n"
        return

    try:
        while True:
            try:
                data = q.get(timeout=30)  # 30秒超时，防止挂死
            except Empty:
                # 发送心跳保持连接
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                continue

            if data.get('type') == 'done':
                yield f"data: {json.dumps(data)}\n\n"
                break

            yield f"data: {json.dumps(data)}\n\n"

    except GeneratorExit:
        logger.info(f"SSE connection closed by client for task_id={task_id}")
    finally:
        close_sse_queue(task_id)
