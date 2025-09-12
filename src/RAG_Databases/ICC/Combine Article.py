from pathlib import Path
import json
import pandas as pd
import numpy as np

LEGAL_SCORES = np.array([0, 0.25, 0.5, 0.75, 1.0])

def round_up_to_legal(score):
    if score in LEGAL_SCORES:
        return float(score)
    higher = LEGAL_SCORES[LEGAL_SCORES >= score]
    if len(higher) > 0:
        return float(higher[0])
    return 1.0

# ---------------- Paths ----------------
repo_root = Path(__file__).resolve().parents[2]   
file1 = repo_root / "RAG_Databases" / "Expert1 Scored" / "Scored Article-Expert1.jsonl"
file2 = repo_root / "RAG_Databases" / "Expert2 Scored" / "Scored Article-Expert2.jsonl"
output_file = repo_root / "RAG_Databases" / "RAG_Data" / "rag_vector_data.jsonl"

# ---------------- Read data ----------------
with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
    data1 = [json.loads(line) for line in f1]
    data2 = [json.loads(line) for line in f2]

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# ---------------- Merge ----------------
fields = ["id", "document_title", "text", "obligation", "precision", "delegation"]
merged = df1[fields].merge(df2[fields], on="id", suffixes=("_exp1", "_exp2"))

output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for _, row in merged.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

print(f"[OK] Combined data saved to {output_file}")
