import os

# 获取图片文件夹中的所有文件
image_dir = 'datasets/spotgeov2/images'
all_files = os.listdir(image_dir)

# 筛选出测试图片
test_files = []
for file_name in all_files:
    if file_name.endswith('_test.png'):
        # 去掉.png后缀
        test_files.append(file_name[:-4])

# 按数字顺序排序
test_files.sort(key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))

# 保存到新的文件
output_file = 'datasets/spotgeov2/img_idx/test_SpotGEOv2.txt'
with open(output_file, 'w') as f:
    for file_name in test_files:
        f.write(file_name + '\n')

print(f"已生成新的测试文件列表：{output_file}")
print(f"总共包含 {len(test_files)} 个测试文件")
print("\n前10个测试文件示例：")
for file in test_files[:10]:
    print(file)

# 读取test_SpotGEOv2.txt文件中的列表
txt_file_path = 'datasets/spotgeov2/img_idx/test_SpotGEOv2 - 副本.txt'
image_dir = 'datasets/spotgeov2/images'

# 读取文件列表
with open(txt_file_path, 'r') as f:
    file_list = [line.strip() + '.png' for line in f.readlines()]

# 获取实际图片文件列表
actual_files = set(os.listdir(image_dir))

# 检查缺失的文件
missing_files = []
for file_name in file_list:
    if file_name not in actual_files:
        missing_files.append(file_name)

# 打印结果
print(f"文件列表中总共有 {len(file_list)} 个文件")
print(f"实际文件夹中有 {len(actual_files)} 个文件")
print(f"缺失的文件数量: {len(missing_files)}")

# 显示前10个实际存在的文件
print("\n实际文件夹中的前10个文件:")
sorted_files = sorted(list(actual_files))
for file in sorted_files[:10]:
    print(file)

print("\n缺失的文件列表:")
for file in missing_files:
    print(file)

# 检查是否有额外的文件（在文件夹中但不在列表中）
extra_files = []
for file_name in actual_files:
    if file_name.endswith('.png') and file_name not in file_list:
        extra_files.append(file_name)

print(f"\n额外文件数量: {len(extra_files)}")
print("额外文件的前10个示例:")
sorted_extra = sorted(extra_files)
for file in sorted_extra[:10]:
    print(file) 