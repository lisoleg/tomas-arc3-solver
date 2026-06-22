"""
TOMAS Web Dashboard - Flask 后端入口

启动 Flask REST API 服务器，注册所有 Blueprint 路由，
配置 CORS 和静态文件服务。
"""

import sys
import os
import logging

# 确保项目根目录在 sys.path 中
_base = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if _base not in sys.path:
    sys.path.insert(0, _base)

from flask import Flask, jsonify
from flask_cors import CORS

from config import PORT, DEBUG, CORS_ORIGINS, FRONTEND_DIST

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """
    创建并配置 Flask 应用。

    Returns:
        Flask: 配置好的 Flask 应用实例
    """
    app = Flask(__name__, static_folder=None)

    # CORS 配置 (允许前端开发服务器跨域访问)
    CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

    # 注册 API Blueprints
    from api.solver import solver_bp

    app.register_blueprint(solver_bp, url_prefix='/api/solver')

    # 健康检查端点
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """健康检查端点。"""
        return jsonify({
            'status': 'ok',
            'service': 'TOMAS Web Dashboard API',
            'version': '2.4.0',
        })

    # 生产环境：服务前端静态文件
    if os.path.exists(FRONTEND_DIST):
        from flask import send_from_directory

        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_frontend(path: str):
            """服务 Vite 前端构建产物。"""
            if path and os.path.exists(os.path.join(FRONTEND_DIST, path)):
                return send_from_directory(FRONTEND_DIST, path)
            return send_from_directory(FRONTEND_DIST, 'index.html')

        logger.info(f"Serving frontend from: {FRONTEND_DIST}")

    # 错误处理
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500

    return app


if __name__ == '__main__':
    app = create_app()
    logger.info(f"Starting TOMAS Web Dashboard on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG)
