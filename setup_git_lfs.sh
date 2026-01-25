#!/bin/bash

# Git LFS 设置脚本
# 用于处理超过 GitHub 100MB 限制的大文件

echo "=========================================="
echo "Git LFS 设置脚本"
echo "=========================================="

# 检查 Git LFS 是否已安装
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS 未安装"
    echo ""
    echo "请先安装 Git LFS:"
    echo "  macOS: brew install git-lfs"
    echo "  或访问: https://git-lfs.github.com/"
    echo ""
    exit 1
fi

echo "✅ Git LFS 已安装"

# 初始化 Git LFS
echo ""
echo "初始化 Git LFS..."
git lfs install

# 检查 .gitattributes 是否存在
if [ ! -f .gitattributes ]; then
    echo "创建 .gitattributes..."
    echo "*.csv filter=lfs diff=lfs merge=lfs -text" > .gitattributes
fi

# 添加 .gitattributes
echo ""
echo "添加 .gitattributes..."
git add .gitattributes

# 重新添加大文件
echo ""
echo "重新添加大文件（通过 Git LFS）..."
git add ofc-ml-challenge-data-code/Features/Train/COSMOS*.csv
git add ofc-ml-challenge-data-code/Features/Train/COSMOS/**/*.csv

# 显示状态
echo ""
echo "当前状态:"
git status --short | head -20

echo ""
echo "=========================================="
echo "✅ 设置完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "  1. 检查文件状态: git status"
echo "  2. 提交更改: git commit -m 'Add large CSV files using Git LFS'"
echo "  3. 推送到 GitHub: git push"
echo ""
