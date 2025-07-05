#!/bin/bash

echo "=== 创建 geo_IRSTD 项目环境 (Python 3.8) ==="

# 1. 创建并激活 conda 环境
echo "1. 创建 conda 环境..."
conda create -n geo python=3.8 -y

if [ $? -ne 0 ]; then
    echo "错误: 创建环境失败"
    exit 1
fi

echo "2. 激活环境..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate geo

if [ $? -ne 0 ]; then
    echo "错误: 激活环境失败"
    exit 1
fi

# 3. 安装 PyTorch 1.12.0 (兼容版本)
echo "3. 安装 PyTorch 1.12.0..."
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y

if [ $? -ne 0 ]; then
    echo "错误: PyTorch 安装失败"
    exit 1
fi

# 4. 安装基础科学计算包
echo "4. 安装基础科学计算包..."
conda install numpy=1.21.6 scipy=1.7.3 matplotlib scikit-image pillow -y

if [ $? -ne 0 ]; then
    echo "错误: 基础包安装失败"
    exit 1
fi

# 5. 安装其他基础包
echo "5. 安装其他基础包..."
conda install tqdm pytest -y
conda install -c conda-forge pywavelets -y

if [ $? -ne 0 ]; then
    echo "错误: 其他基础包安装失败"
    exit 1
fi

# 6. 安装特殊包 (pip)
echo "6. 安装特殊包..."
pip install opencv-python-headless torchinfo timm tensorboard tensorboardX thop albumentations==1.3.0

if [ $? -ne 0 ]; then
    echo "错误: 特殊包安装失败"
    exit 1
fi

# 7. 安装 SwanLab 监控工具
echo "7. 安装 SwanLab 监控工具..."
pip install swanlab

if [ $? -ne 0 ]; then
    echo "错误: SwanLab 安装失败"
    exit 1
fi

# 8. 安装 mmcv 和 mmdet (兼容版本)
echo "8. 安装 MMCV 和 MMDet..."
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

if [ $? -ne 0 ]; then
    echo "错误: MMCV 安装失败"
    exit 1
fi

pip install mmdet==3.0.0

if [ $? -ne 0 ]; then
    echo "错误: MMDet 安装失败"
    exit 1
fi

# 9. 安装其他依赖
echo "9. 安装其他依赖..."
pip install fvcore
pip install -U openmim

if [ $? -ne 0 ]; then
    echo "错误: 其他依赖安装失败"
    exit 1
fi

# 10. 验证安装
echo "10. 验证安装..."
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import mmcv; import mmdet; print('✓ MMCV and MMDet OK')"
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import scipy; print(f'✓ SciPy: {scipy.__version__}')"
python -c "import swanlab; print('✓ SwanLab OK')"

echo ""
echo "=== 环境创建完成！ ==="
echo ""
echo "使用方法："
echo "conda activate geo"
echo "python train_geo_modified.py"
echo ""
echo "环境信息："
echo "- Python: 3.8"
echo "- PyTorch: 1.12.0 (CUDA 11.3)"
echo "- NumPy: 1.21.6"
echo "- SciPy: 1.7.3"
echo "- MMCV: 2.0.0"
echo "- MMDet: 3.0.0"
echo "- SwanLab: 已安装 (用于训练监控)"
echo ""
echo "SwanLab 使用说明："
echo "1. 训练时会自动启动 SwanLab 监控"
echo "2. 在浏览器中打开显示的 URL 查看训练进度"
echo "3. 可以通过 --use_swanlab False 禁用监控"
echo ""
echo "如果遇到问题，请检查："
echo "1. CUDA 版本是否兼容"
echo "2. 网络连接是否正常"
echo "3. conda 和 pip 是否正常工作" 