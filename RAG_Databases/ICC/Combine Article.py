import os
import json
import pandas as pd
import numpy as np

# 预定义合法分数等级
LEGAL_SCORES = np.array([0, 0.25, 0.5, 0.75, 1.0])

def round_up_to_legal(score):
    # 若已是合法分数，直接返回
    if score in LEGAL_SCORES:
        return float(score)
    # 找到大于等于score的最小合法分数
    higher = LEGAL_SCORES[LEGAL_SCORES >= score]
    if len(higher) > 0:
        return float(higher[0])
    else:
        return 1.0  # 理论上不可能走到这里

# 文件路径
file1 = r"../Expert1 Scored/Scored Artlcie-Expert1.json"
file2 = r"../Expert2 Scored/Scored Artlcie-Expert2.json"
output_file = r"../RAG_Data/rag_vector_data.jsonl"

# 读取数据
with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    data1 = [json.loads(line) for line in f1]
    data2 = [json.loads(line) for line in f2]

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# 合并数据（以id为主键对齐）
fields = ['id', 'document_title', 'text', 'obligation', 'precision', 'delegation']
merged = pd.merge(
    df1[fields], 
    df2[['id', 'obligation', 'precision', 'delegation']], 
    on='id', 
    suffixes=('_1', '_2')
)

# 计算均值（向上四舍五入到最近合法分数）和置信度
for field in ['obligation', 'precision', 'delegation']:
    avg = merged[[f'{field}_1', f'{field}_2']].mean(axis=1)
    merged[field] = avg.apply(round_up_to_legal)
    merged[f'confidence_{field}'] = 1 - (merged[f'{field}_1'] - merged[f'{field}_2']).abs()

# 构造输出字段
final_cols = [
    'id', 'document_title', 'text',
    'obligation', 'precision', 'delegation',
    'confidence_obligation', 'confidence_precision', 'confidence_delegation'
]
output_df = merged[final_cols]

# 写入jsonl文件
with open(output_file, 'w', encoding='utf-8') as f:
    for record in output_df.to_dict(orient='records'):
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"已完成整合，输出文件路径为: {output_file}")
