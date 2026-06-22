# Kaggle API 凭证设置指南

## 步骤一：获取 Kaggle API Token

1. 登录 https://www.kaggle.com/account
2. 滚动到 **API** 部分
3. 点击 **Create New Token**
4. 浏览器自动下载 `kaggle.json` 文件

## 步骤二：放置凭证文件

### Linux / macOS
```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Windows (Git Bash / WSL)
```bash
mkdir -p ~/.kaggle
cp /c/Users/YOUR_USER/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Windows (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Copy-Item "C:\Users\YOUR_USER\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\kaggle.json"
```

## 步骤三：验证

```bash
python kaggle/submit.py --check
```

预期输出：
```
[OK] Kaggle credentials found (user: your_username)
[OK] kaggle API reachable
[OK] Notebook has v2.4 config
[OK] All checks passed (4/4)
```

## 步骤四：提交到 Kaggle

```bash
# 上传 notebook（首次）
python kaggle/submit.py --upload

# 提交到 ARC-AGI-3 竞赛
python kaggle/submit.py --submit competition-arc-agi-3
```

## 常见问题

**Q: `403 Forbidden` 错误**
A: Token 已过期，重新 Create New Token

**Q: `Competition not found`**
A: 确认已加入竞赛 https://www.kaggle.com/competitions/competition-arc-agi-3/rules
   （需接受竞赛规则才能提交）

**Q: Windows 路径问题**
A: 确保 `%USERPROFILE%\.kaggle\kaggle.json` 存在，内容格式：
```json
{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}
```
