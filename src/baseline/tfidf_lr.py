# -*- coding: utf-8 -*-
"""
TF-IDF + Logistic Regression Baseline Scoring
"""

import json
from pathlib import Path

import joblib

# 项目根目录
REPO_ROOT = Path(__file__).resolve().parents[2]

# 路径配置
MODEL_DIR = REPO_ROOT / "models" / "tfidf-lr-baseline"
INPUT_JSONL = REPO_ROOT / "data" / "processed" / "test_articles.json"
OUTPUT_JSONL = REPO_ROOT / "outputs" / "tfidf-lr-baseline" / "results.jsonl"

SCORES = [0.0, 0.25, 0.5, 0.75, 1.0]
CLASS_TO_SCORE = {i: s for i, s in enumerate(SCORES)}
DIMS = ["obligation", "precision", "delegation"]


def read_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at line {ln}: {e}")
    return data


def main():
    print(f"Model dir: {MODEL_DIR}")
    print(f"Input: {INPUT_JSONL}")
    print(f"Output: {OUTPUT_JSONL}")
    
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model dir not found: {MODEL_DIR}")
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSONL}")
    
    # Load test data
    data = read_jsonl(INPUT_JSONL)
    print(f"Loaded {len(data)} samples")
    
    if "text" not in data[0]:
        raise KeyError(f"Missing key 'text' in test data. Keys: {list(data[0].keys())}")
    
    # Load models
    models = {}
    for dim in DIMS:
        p = MODEL_DIR / f"tfidf_lr_{dim}.joblib"
        if not p.exists():
            raise FileNotFoundError(f"Missing model file: {p}")
        models[dim] = joblib.load(p)
    print(f"[OK] Loaded 3 models from {MODEL_DIR}")
    
    # Predict
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for i, d in enumerate(data):
            text = str(d.get("text", ""))
            
            row = {
                "id": d.get("id", str(i)),
                "document_title": d.get("document_title", d.get("title", "")),
                "text": text
            }
            
            for dim in DIMS:
                pred_class = models[dim].predict([text])[0]
                row[dim] = float(CLASS_TO_SCORE[int(pred_class)])
            
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"[OK] Wrote predictions to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
