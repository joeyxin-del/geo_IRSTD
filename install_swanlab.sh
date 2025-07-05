#!/bin/bash

echo "=== 快速安装 SwanLab 监控工具 ==="

# 检查是否在 geo 环境中
if [[ "$CONDA_DEFAULT_ENV" != "geo" ]]; then
    echo "⚠️  警告: 当前不在 geo 环境中"
    echo "请先运行: conda activate geo"
    echo ""
    read -p "是否继续安装? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "安装已取消"
        exit 1
    fi
fi

echo "1. 安装 SwanLab..."
pip install swanlab

if [ $? -ne 0 ]; then
    echo "❌ SwanLab 安装失败"
    exit 1
fi

echo "2. 验证安装..."
python -c "import swanlab; print('✅ SwanLab 安装成功!')"

if [ $? -ne 0 ]; then
    echo "❌ SwanLab 验证失败"
    exit 1
fi

echo ""
echo "=== SwanLab 安装完成！ ==="
echo ""
echo "使用方法："
echo "1. 运行训练脚本: python train_geo_modified.py"
echo "2. SwanLab 会自动启动并显示 URL"
echo "3. 在浏览器中打开 URL 查看训练监控"
echo ""
echo "示例："
echo "python train_geo_modified.py --num_epochs 10 --batch_size 2"
echo ""
echo "如果不想使用 SwanLab，可以禁用："
echo "python train_geo_modified.py --use_swanlab False" 