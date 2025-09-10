import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# 路径设定
BASE_DIR = r"../RAG_Data"
JSONL_FILE = os.path.join(BASE_DIR, "rag_vector_data.jsonl")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "asean_scoring"

# 1. 读取jsonl数据
data = []
with open(JSONL_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)

# 2. 检查条款数量、唯一性和质量
print("【统计信息】")
print("总条款数：", len(df))
print("条款文本去重后数：", df["text"].nunique())
print("空文本条款数：", df["text"].isnull().sum())
print("重复文本数：", df["text"].duplicated().sum())

# 可选：如需排查空文本或重复条款
if df["text"].isnull().sum() > 0:
    print("存在空文本条款，请检查！")
if df["text"].duplicated().sum() > 0:
    print("存在重复条款，可考虑去重！")

# 3. 向量化准备
texts = df["text"].tolist()
print("准备向量化的文本条款数量：", len(texts))

# 4. 加载向量模型并执行向量化
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(texts, show_progress_bar=True)

print("实际生成向量数量：", len(embeddings))

# 5. 写入Chroma数据库
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(COLLECTION_NAME)  # 避免重名
collection = client.create_collection(COLLECTION_NAME)

collection.add(
    embeddings=embeddings,
    ids=df["id"].astype(str).tolist(),
    documents=df["text"].tolist(),
    metadatas=df.drop(columns=["text"]).to_dict(orient="records")
)

print(f"\nChroma向量数据库已写入，路径：{CHROMA_DB_DIR} 集合名：{COLLECTION_NAME}")
