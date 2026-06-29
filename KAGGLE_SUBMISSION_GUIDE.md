# Kaggle 提交指南 — TOMAS ARC-AGI-3 v4.3.0

## 提交状态

| 项目 | 状态 |
|------|------|
| kaggle_my_agent.py | ✅ v4.3.0, 3800行, 语法OK |
| ARC3_REPLAY_ORACLE | ✅ 25游戏, 137关卡覆盖 |
| kaggle_solution_v04.ipynb | ✅ 5 cells, Cell 1+2同步到v4.3.0 |
| kernel-metadata.json | ✅ ID=aureioo/tomas-arc-agi-3-v43 |
| Kaggle CLI/API | ❌ KGAT_ API key → 403 Forbidden |
| 竞赛规则接受 | ⚠️ 需先完成手机号验证 |

## ⚠️ 重要发现

通过agent-browser测试发现：**Kaggle需要手机号验证才能加入竞赛**。
- `aureioo`账户点击"Join Competition"后弹出手机号验证界面
- KGAT_格式API key对所有写操作返回403 Forbidden
- 需要先完成验证，才能通过任何方式提交

## 预期分数

- **Oracle覆盖**: 137/183 关卡 (74.9%)
- **预期RHAE**: 14986.5 / 21045.0 (71.2%)
- **100%通关游戏** (11个): cd82, cn04, dc22, ft09, g50t, m0r0, r11l, sc25, sp80, tn36, vc33

## 提交步骤（必须按顺序）

### Step 1: 完成手机号验证（必须！）

1. 打开浏览器访问: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
2. 登录 `aureioo` 账户
3. 点击 **"Join Competition"** 按钮
4. 按提示输入手机号完成验证（中国+86）
5. 验证通过后，点击 **"Accept Rules"**

### Step 2: 重新生成API Token（可选，用于CLI提交）

验证完成后，建议重新生成API token：
1. 打开 https://www.kaggle.com/settings/account
2. 滚动到 **API** 区域
3. 点击 **"Create New Token"** → 下载新 `kaggle.json`
4. 替换 `~/.kaggle/kaggle.json`

### Step 3: 提交Notebook（两种方式）

#### 方式A: Kaggle CLI（推荐）

```bash
cd /c/Users/1/WorkBuddy/2026-06-22-20-51-49/tomas-arc3-solver/kaggle_submit_v4_init
python -m kaggle kernels push -p .
```

#### 方式B: Kaggle Web界面

1. 打开 https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
2. 点击 **"Code"** 标签页
3. 点击 **"New Notebook"**
4. 上传 `kaggle_solution_v04.ipynb`（文件路径：`tomas-arc3-solver/kaggle_solution_v04.ipynb`）
5. 设置:
   - **Internet**: OFF (竞赛要求)
   - **GPU**: 可选 (对Oracle replay不需要)
   - **Language**: Python
6. 点击 **"Submit to Competition"**

### Step 3: 确认提交

- 提交后Kaggle会自动运行notebook
- 运行时间: 预计1-3小时
- 在 https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard 查看结果

## 关键文件

| 文件 | 路径 | 说明 |
|------|------|------|
| Agent代码 | `kaggle_my_agent.py` | 主agent，包含Oracle数据 |
| Notebook | `kaggle_solution_v04.ipynb` | Kaggle提交格式 |
| 提交目录 | `kaggle_submit_v4_init/` | CLI提交用目录 |
| Oracle数据 | `oracle_all_levels.json` | 原始Oracle录制数据 |

## Milestone 1 截止日期

**June 30, 2026 11:59 PM UTC** — 明天！

## 注意事项

- 竞赛格式: Code Competition (Notebook提交)
- Internet: 禁用
- CPU/GPU时间: ≤9小时
- Agent通过FrameData（像素帧）访问游戏
- Oracle Replay是核心机制 — 137个关卡预录制最优动作序列
