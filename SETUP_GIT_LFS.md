# Git LFS 设置指南

## 问题
有4个大文件超过 GitHub 的限制：
- `ofc-ml-challenge-data-code/Features/Train/COSMOS/COSMOS_features.csv` (167MB) - 超过100MB
- `ofc-ml-challenge-data-code/Features/Train/COSMOS_features.csv` (105MB) - 超过100MB  
- `ofc-ml-challenge-data-code/Features/Train/COSMOS/features/COSMOS_features_features.csv` (104MB) - 超过100MB
- `ofc-ml-challenge-data-code/Features/Train/COSMOS_labels.csv` (53MB) - 超过50MB推荐大小

## 解决方案：使用 Git LFS

### 步骤 1: 安装 Git LFS

**macOS (使用 Homebrew):**
```bash
brew install git-lfs
```

**或者手动安装:**
访问 https://git-lfs.github.com/ 下载安装

### 步骤 2: 初始化 Git LFS
```bash
cd /Users/gaozhengqi/Documents/research/20260108ofc_challenge/code
git lfs install
```

### 步骤 3: 重新添加文件
```bash
# 添加 .gitattributes（已创建）
git add .gitattributes

# 重新添加大文件（现在会通过 Git LFS）
git add ofc-ml-challenge-data-code/Features/Train/COSMOS*.csv
git add ofc-ml-challenge-data-code/Features/Train/COSMOS/**/*.csv

# 提交
git commit -m "Add large CSV files using Git LFS"

# 推送
git push
```

## 注意事项

1. `.gitattributes` 文件已创建，配置所有 `.csv` 文件使用 Git LFS
2. 确保 Git LFS 已安装并初始化后再添加文件
3. 如果已经提交了大文件，可能需要从历史中移除（使用 `git filter-branch` 或 `git filter-repo`）
