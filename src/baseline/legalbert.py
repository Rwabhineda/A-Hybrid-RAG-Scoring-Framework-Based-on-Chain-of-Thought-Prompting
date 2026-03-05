# -*- coding: utf-8 -*-
"""
Legal BERT Baseline Scoring
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertConfig, BertModel
from safetensors.torch import load_file as safe_load

# 项目根目录
REPO_ROOT = Path(__file__).resolve().parents[2]

# 路径配置
MODEL_DIR = REPO_ROOT / "models" / "legal-bert-baseline"
INPUT_JSONL = REPO_ROOT / "data" / "processed" / "test_articles.json"
OUTPUT_JSONL = REPO_ROOT / "outputs" / "legalbert-baseline" / "results.jsonl"

MAX_LEN = 256
SNAP_TO_5POINT = True


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def snap_to_5point(x: np.ndarray) -> np.ndarray:
    grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    idx = np.abs(x[:, None] - grid[None, :]).argmin(axis=-1)
    return grid[idx]


class LegalBert3HeadReg(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = BertConfig.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.encoder = BertModel(cfg)
        hidden = cfg.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_obl = nn.Linear(hidden, 1)
        self.head_pre = nn.Linear(hidden, 1)
        self.head_del = nn.Linear(hidden, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else None
        )
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        po = self.head_obl(cls).squeeze(-1)
        pp = self.head_pre(cls).squeeze(-1)
        pd = self.head_del(cls).squeeze(-1)
        return po, pp, pd


def load_trained_weights(model: nn.Module, model_dir: Path):
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        raise FileNotFoundError(f"Cannot find: {st_path}")
    
    state = safe_load(str(st_path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    if missing and any("encoder" in k for k in missing):
        print(f"[WARNING] Missing encoder keys: {missing[:5]}...")
    if unexpected:
        print(f"[INFO] Unexpected keys (ignored): {unexpected[:3]}...")
    
    print(f"[OK] Loaded weights from {st_path}")


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
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    
    # Load model
    model = LegalBert3HeadReg()
    load_trained_weights(model, MODEL_DIR)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    
    # Load data
    data = read_jsonl(INPUT_JSONL)
    print(f"Loaded {len(data)} samples")
    
    # Predict
    results = []
    with torch.no_grad():
        for i, d in enumerate(data):
            text = str(d.get("text", ""))
            inputs = tokenizer(
                text,
                max_length=MAX_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            po, pp, pd = model(**inputs)
            
            obl = po.cpu().numpy()[0]
            pre = pp.cpu().numpy()[0]
            deleg = pd.cpu().numpy()[0]
            
            results.append({
                "id": d.get("id", str(i)),
                "document_title": d.get("document_title", d.get("title", "")),
                "text": text,
                "obligation": float(obl),
                "precision": float(pre),
                "delegation": float(deleg)
            })
    
    # Snap to 5-point scale
    if SNAP_TO_5POINT:
        for r in results:
            r["obligation"] = float(snap_to_5point(np.array([r["obligation"]]))[0])
            r["precision"] = float(snap_to_5point(np.array([r["precision"]]))[0])
            r["delegation"] = float(snap_to_5point(np.array([r["delegation"]]))[0])
    
    # Save
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"[OK] Wrote {len(results)} predictions to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
