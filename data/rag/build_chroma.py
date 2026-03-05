"""
Build ChromaDB vector store using intfloat/e5-large-v2 embeddings.

Usage:
    uv run python data/rag/build_chroma.py

Source data : data/rag/rag_vector_data.jsonl
Output      : data/rag/chroma_db/  (collection: asean_scoring)
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
JSONL_FILE = SCRIPT_DIR / "rag_vector_data.jsonl"
CHROMA_DB_DIR = str(SCRIPT_DIR / "chroma_db")
COLLECTION_NAME = "asean_scoring"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
BATCH_SIZE = 256


def load_jsonl(path: Path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("text") or obj.get("text_exp1", "")
            if not text:
                continue

            obligation = obj.get("obligation") or obj.get("obligation_exp1")
            precision = obj.get("precision") or obj.get("precision_exp1")
            delegation = obj.get("delegation") or obj.get("delegation_exp1")
            doc_title = (
                obj.get("document_title") or obj.get("document_title_exp1", "")
            )

            conf_o = obj.get("confidence_obligation", 0.5)
            conf_p = obj.get("confidence_precision", 0.5)
            conf_d = obj.get("confidence_delegation", 0.5)

            obl2 = obj.get("obligation_exp2")
            pre2 = obj.get("precision_exp2")
            del2 = obj.get("delegation_exp2")
            if obl2 is not None and obligation is not None:
                conf_o = 1.0 - abs(obligation - obl2)
                obligation = (obligation + obl2) / 2
            if pre2 is not None and precision is not None:
                conf_p = 1.0 - abs(precision - pre2)
                precision = (precision + pre2) / 2
            if del2 is not None and delegation is not None:
                conf_d = 1.0 - abs(delegation - del2)
                delegation = (delegation + del2) / 2

            records.append(
                {
                    "id": str(obj["id"]),
                    "text": text,
                    "document_title": doc_title,
                    "obligation": obligation,
                    "precision": precision,
                    "delegation": delegation,
                    "confidence_obligation": round(conf_o, 4),
                    "confidence_precision": round(conf_p, 4),
                    "confidence_delegation": round(conf_d, 4),
                }
            )
    return records


def main():
    print(f"Loading data from {JSONL_FILE} ...")
    records = load_jsonl(JSONL_FILE)
    print(f"  Total clauses: {len(records)}")

    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    metas = [
        {k: v for k, v in r.items() if k not in ("id", "text")} for r in records
    ]

    print(f"Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Encoding (with 'passage: ' prefix for E5) ...")
    prefixed_texts = ["passage: " + t for t in texts]
    embeddings = model.encode(prefixed_texts, show_progress_bar=True, batch_size=BATCH_SIZE)
    print(f"  Vectors: {len(embeddings)}, dim: {embeddings.shape[1]}")

    print(f"Writing to ChromaDB: {CHROMA_DB_DIR} / {COLLECTION_NAME} ...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    total = len(ids)
    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        collection.add(
            embeddings=embeddings[start:end].tolist(),
            ids=ids[start:end],
            documents=texts[start:end],
            metadatas=metas[start:end],
        )
        print(f"  Written {end}/{total}")

    print(f"\nDone! ChromaDB at: {CHROMA_DB_DIR}, collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
