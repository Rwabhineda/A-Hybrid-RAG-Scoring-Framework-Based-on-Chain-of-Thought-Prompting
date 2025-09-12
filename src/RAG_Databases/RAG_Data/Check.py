import os
import json
import pandas as pd

# 路径设定
BASE_DIR = r"../RAG_Data"
JSONL_FILE = os.path.join(BASE_DIR, "rag_vector_data.jsonl")

# 1. 读取数据
data = []
with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# 2. 检查重复条款
dup_mask = df["text"].duplicated(keep=False)  # 标记所有重复的text（包含所有出现的行）
duplicates = df[dup_mask].sort_values("text")  # 按文本排序便于观察

if duplicates.empty:
    print("没有发现重复条款！")
else:
    print(f"发现{duplicates.shape[0]}条重复（包括所有重复项）：")
    # 打印重复条款的id、所在行号和内容
    for idx, row in duplicates.iterrows():
        print(f"行号: {idx}, id: {row['id']}, 条款内容: {row['text']}\n")
    # 也可以按text分组，只输出有重复的条款内容和所有对应id
    print("\n【分组显示所有重复的条款及其id列表】：")
    grouped = duplicates.groupby("text")["id"].apply(list)
    for text, ids in grouped.items():
        if len(ids) > 1:
            print(f"重复条款内容：{text}\n所有重复id: {ids}\n---")
