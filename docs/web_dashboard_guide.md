# TOMAS Web Dashboard 使用指南

## 概述

TOMAS Web Dashboard 是一个基于 Flask + React 的可视化调试界面，用于实时监控和分析 ARC-AGI-3 求解过程。

## 架构

```
src/web/
├── backend/                  # Flask REST API
│   ├── app.py               # 入口 (端口 5050)
│   ├── config.py            # CORS/端口/目录配置
│   ├── api/
│   │   ├── solver.py        # 求解器 API (run/status/progress/history)
│   │   └── visualizations.py # 可视化 API (search-tree/fiber/pruning/benchmark)
│   └── services/
│       ├── solver_runner.py  # 后台求解执行引擎
│       ├── sse_manager.py    # SSE 实时进度推送
│       └── viz_data.py       # 可视化数据生成
└── frontend/                 # Vite + React + MUI
    └── src/
        ├── App.tsx           # 路由 (Solver / History)
        ├── api/client.ts     # API 客户端 (axios + SSE)
        ├── pages/
        │   ├── SolverPage.tsx  # 求解页面 (配置 + 运行 + 可视化)
        │   └── HistoryPage.tsx # 历史记录页面 (列表 + 详情)
        └── components/
            ├── KappaSnapTree.tsx  # κ-Snap 搜索树 (D3.js)
            ├── GaussExFiber.tsx   # GaussEx 纤维验证 (表格+进度条)
            └── PruningChart.tsx   # 8策略剪枝率 (Recharts)
```

## 快速启动

### 1. 启动后端

```bash
cd tomas-arc3-solver
pip install -r src/web/backend/requirements.txt
cd src/web/backend
python app.py
# → http://localhost:5050
```

### 2. 启动前端

```bash
cd src/web/frontend
npm install
npm run dev
# → http://localhost:5173
```

### 3. 生产部署

```bash
# 构建前端
cd src/web/frontend
npm run build
# 后端自动服务 dist/ 静态文件
# 访问 http://localhost:5050
```

## 功能说明

### 求解页面 (Solver)

1. **任务配置**：
   - 输入路径（ARC JSON 文件或目录）
   - 输出路径
   - 推理模式（video / bayesian / fusion / auto）

2. **高级功能开关**：
   - ψ-Gate 语义门控
   - AEGIS 演进引擎
   - 因果 DSL 先验

3. **实时进度**：
   - SSE 推送求解进度
   - 实时日志显示
   - 进度条

4. **求解后可视化**：
   - κ-Snap 搜索树（D3.js 层级树图）
   - GaussEx 纤维验证矩阵
   - 8策略剪枝率图表（Recharts）

### 历史页面 (History)

1. **任务列表**：
   - 任务ID、输入文件、模式、状态
   - 耗时、候选数、验证数、剪枝率
   - 置信度、MDL、ψ-Gate/AEGIS 开关状态
   - 时间戳

2. **筛选**：
   - 按推理模式筛选
   - 按状态筛选

3. **详情查看**：
   - 点击查看按钮弹出详情对话框
   - 包含完整的三种可视化

4. **删除记录**：
   - 单条删除

### 可视化组件

#### κ-Snap 搜索树 (D3.js)

- **颜色编码**：
  - 绿色：Phase A 通过
  - 红色：Phase A 失败
  - 蓝色：Phase B 验证通过
  - 橙色：Phase B 验证失败
- **节点信息**：名称、MDL、置信度
- **统计数据**：总候选数、Phase A 通过/失败数、验证数

#### GaussEx 纤维验证

- **Demo Pairs 表格**：输入/输出形状、纤维数、交叉数、CRC32 匹配
- **候选验证矩阵**：每个候选对每个 demo 的验证状态
- **验证率进度条**：总体验证通过率

#### 8策略剪枝率图表 (Recharts)

- **柱状图**：每策略的剪枝数和剩余数
- **折线图**：剪枝率 + 累计剪枝率
- **漏斗图**：候选数逐步减少趋势

## API 端点

### 求解器 API

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/solver/modes` | 获取推理模式列表 |
| POST | `/api/solver/run` | 启动求解任务 |
| GET | `/api/solver/status/<id>` | 查询任务状态 |
| GET | `/api/solver/progress/<id>` | SSE 进度流 |
| GET | `/api/solver/history` | 获取历史列表 |

### 可视化 API

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/api/viz/search-tree/<id>` | κ-Snap 搜索树数据 |
| GET | `/api/viz/fiber-verification/<id>` | GaussEx 纤维验证数据 |
| GET | `/api/viz/pruning-stats/<id>` | 8策略剪枝统计数据 |
| GET | `/api/viz/task-detail/<id>` | 完整可视化数据 |
| GET | `/api/viz/history` | 历史记录列表 |
| GET | `/api/viz/history/<id>` | 历史详情 |
| DELETE | `/api/viz/history/<id>` | 删除历史 |
| GET | `/api/viz/benchmark` | 性能基准对比数据 |

## 配置

修改 `src/web/backend/config.py`：

```python
PORT = 5050              # 后端端口
DEBUG = True             # 调试模式
CORS_ORIGINS = [...]     # CORS 允许来源
HISTORY_FILE = '...'     # 历史记录存储路径
FRONTEND_DIST = '...'    # 前端构建输出目录
```

## 技术栈

- **后端**：Flask 2.x, Flask-CORS, SSE
- **前端**：React 18, MUI 5, Vite 5, TypeScript
- **可视化**：D3.js 7 (搜索树), Recharts 2 (图表)
- **通信**：Axios (REST), EventSource (SSE)
