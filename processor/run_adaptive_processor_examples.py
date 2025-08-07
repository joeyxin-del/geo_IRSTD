#!/usr/bin/env python3
"""
自适应角度距离处理器使用示例

这个脚本展示了如何使用不同的配置来运行自适应角度距离处理器：
1. 同时使用角度和步长聚类（默认模式）
2. 仅使用角度聚类
3. 仅使用步长聚类
4. 不使用聚类（仅基础处理）
"""

import os
import subprocess
import json
from pathlib import Path

def run_processor(config_name, cmd_args):
    """运行处理器并返回结果"""
    print(f"\n{'='*60}")
    print(f"运行配置: {config_name}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = ["python", "processor/adaptive_angle_distance_processor.py"] + cmd_args
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        # 运行命令
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("执行成功!")
        print("输出:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e}")
        print("错误输出:")
        print(e.stderr)
        return False

def main():
    """主函数"""
    
    # 基础配置
    base_config = [
        "--pred_path", "results/spotgeov2-IRSTD/WTNet/predictions_8807.json",
        "--gt_path", "datasets/spotgeov2-IRSTD/test_anno.json",
        "--no_visualization",  # 禁用可视化以加快处理速度
        "--confidence_threshold", "0.35",
        "--angle_tolerance", "12.0",
        "--step_tolerance", "0.22",
        "--point_distance_threshold", "7.0"
    ]
    
    # 配置1: 同时使用角度和步长聚类（默认模式）
    config1_args = base_config + [
        "--output_path", "results/spotgeov2/WTNet/adaptive_both_clustering.json",
        "--use_angle_clustering",
        "--use_step_clustering",
        "--angle_weight", "0.65",
        "--step_weight", "0.35"
    ]
    
    # 配置2: 仅使用角度聚类
    config2_args = base_config + [
        "--output_path", "results/spotgeov2/WTNet/adaptive_angle_only.json",
        "--use_angle_clustering",
        "--no_step_clustering"
    ]
    
    # 配置3: 仅使用步长聚类
    config3_args = base_config + [
        "--output_path", "results/spotgeov2/WTNet/adaptive_step_only.json",
        "--no_angle_clustering",
        "--use_step_clustering"
    ]
    
    # 配置4: 不使用聚类（仅基础处理）
    config4_args = base_config + [
        "--output_path", "results/spotgeov2/WTNet/adaptive_no_clustering.json",
        "--no_angle_clustering",
        "--no_step_clustering"
    ]
    
    # 配置5: 调整权重（更重视角度）
    config5_args = base_config + [
        "--output_path", "results/spotgeov2/WTNet/adaptive_angle_heavy.json",
        "--use_angle_clustering",
        "--use_step_clustering",
        "--angle_weight", "0.8",
        "--step_weight", "0.2"
    ]
    
    # 配置6: 调整权重（更重视步长）
    config6_args = base_config + [
        "--output_path", "results/spotgeov2/WTNet/adaptive_step_heavy.json",
        "--use_angle_clustering",
        "--use_step_clustering",
        "--angle_weight", "0.3",
        "--step_weight", "0.7"
    ]
    
    # 运行所有配置
    configurations = [
        ("同时使用角度和步长聚类（默认）", config1_args),
        ("仅使用角度聚类", config2_args),
        ("仅使用步长聚类", config3_args),
        ("不使用聚类（基础处理）", config4_args),
        ("角度权重更高（0.8:0.2）", config5_args),
        ("步长权重更高（0.3:0.7）", config6_args)
    ]
    
    results = {}
    
    for config_name, config_args in configurations:
        success = run_processor(config_name, config_args)
        results[config_name] = success
        
        if success:
            print(f"✅ {config_name} 执行成功")
        else:
            print(f"❌ {config_name} 执行失败")
    
    # 打印总结
    print(f"\n{'='*60}")
    print("执行总结")
    print(f"{'='*60}")
    
    for config_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{config_name}: {status}")
    
    # 生成比较脚本
    generate_comparison_script()

def generate_comparison_script():
    """生成比较不同配置结果的脚本"""
    
    comparison_script = '''#!/usr/bin/env python3
"""
比较不同配置的处理结果
"""

import json
import os
from pathlib import Path

def load_results(file_path):
    """加载处理结果"""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

def count_predictions(predictions):
    """统计预测数量"""
    if not predictions:
        return 0
    
    total_objects = 0
    for img_name, pred_info in predictions.items():
        total_objects += pred_info.get('num_objects', 0)
    
    return len(predictions), total_objects

def main():
    """主函数"""
    
    # 结果文件路径
    result_files = {
        "同时使用角度和步长聚类": "results/spotgeov2/WTNet/adaptive_both_clustering.json",
        "仅使用角度聚类": "results/spotgeov2/WTNet/adaptive_angle_only.json",
        "仅使用步长聚类": "results/spotgeov2/WTNet/adaptive_step_only.json",
        "不使用聚类": "results/spotgeov2/WTNet/adaptive_no_clustering.json",
        "角度权重更高": "results/spotgeov2/WTNet/adaptive_angle_heavy.json",
        "步长权重更高": "results/spotgeov2/WTNet/adaptive_step_heavy.json"
    }
    
    print("="*80)
    print("不同配置处理结果比较")
    print("="*80)
    
    results_summary = {}
    
    for config_name, file_path in result_files.items():
        predictions = load_results(file_path)
        if predictions:
            num_images, num_objects = count_predictions(predictions)
            results_summary[config_name] = {
                "num_images": num_images,
                "num_objects": num_objects,
                "avg_objects_per_image": num_objects / num_images if num_images > 0 else 0
            }
            print(f"{config_name}:")
            print(f"  图像数量: {num_images}")
            print(f"  目标总数: {num_objects}")
            print(f"  平均每张图像目标数: {num_objects / num_images:.2f}")
        else:
            print(f"{config_name}: 文件不存在或加载失败")
        print()
    
    # 保存比较结果
    with open("results/spotgeov2/WTNet/config_comparison_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print("比较结果已保存到: results/spotgeov2/WTNet/config_comparison_summary.json")

if __name__ == '__main__':
    main()
'''
    
    # 保存比较脚本
    script_path = "processor/compare_configurations.py"
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(comparison_script)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    print(f"\n比较脚本已生成: {script_path}")
    print("运行 'python processor/compare_configurations.py' 来比较不同配置的结果")

if __name__ == '__main__':
    main() 