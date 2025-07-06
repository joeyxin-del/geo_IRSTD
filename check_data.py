import json
import os
import pprint

predictions_file = os.path.join("datasets", "spotgeov2", "test_anno.json")

with open(predictions_file, 'r') as f:
    data = json.load(f)
    
print("数据类型:", type(data))
if isinstance(data, list):
    print("列表长度:", len(data))
    if len(data) > 0:
        print("\n前三个元素的详细信息:")
        pp = pprint.PrettyPrinter(indent=2)
        for i in range(min(3, len(data))):
            print(f"\n第{i+1}个元素:")
            pp.pprint(data[i])
elif isinstance(data, dict):
    print("字典键数量:", len(data))
    print("\n键的列表:")
    print(list(data.keys())[:5])  # 只显示前5个键 