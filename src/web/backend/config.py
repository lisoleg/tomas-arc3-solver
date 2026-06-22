"""
TOMAS Web Dashboard - Flask 后端配置

控制 Flask 应用的 CORS、端口、目录等全局配置。
"""

import os

# Flask 服务端口
PORT = 5050

# 调试模式 (生产环境设为 False)
DEBUG = True

# CORS 允许的来源 (前端开发服务器地址)
CORS_ORIGINS = [
    'http://localhost:5173',
    'http://127.0.0.1:5173',
    'http://localhost:3000',
    'http://127.0.0.1:3000',
]

# 基础目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 求解历史存储文件 (JSON 文件)
HISTORY_FILE = os.path.join(BASE_DIR, '..', '..', '..', 'logs', 'solver_history.json')

# 前端构建输出目录 (生产环境静态文件)
FRONTEND_DIST = os.path.join(BASE_DIR, '..', 'frontend', 'dist')

# 日志级别
LOG_LEVEL = 'INFO'

# 确保必要目录存在
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
