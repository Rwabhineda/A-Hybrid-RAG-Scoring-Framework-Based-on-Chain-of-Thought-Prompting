# -*- coding: utf-8 -*-
"""
Evaluation script for clause scoring.
- gold_path is fixed (gold standard file in gold_standard folder).
- pred_path is configurable via YAML config file or command-line argument.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import spearmanr


# 五级评分映射
SCORE_TO_CLASS = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
VALID_SCORES = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)


def snap_to_valid(arr: np.ndarray) -> np.ndarray:
    """将预测值吸附到最近的合法档位 {0, 0.25, 0.5, 0.75, 1.0}。"""
    return np.array([VALID_SCORES[np.argmin(np.abs(VALID_SCORES - x))] for x in arr])


def compute_metrics(y_true, y_pred) -> dict:
    """Compute evaluation metrics: Spearman, MAE, QWK, ACC."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    spearman, _ = spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    y_true_snapped = snap_to_valid(y_true)
    y_pred_snapped = snap_to_valid(y_pred)
    acc = float(np.mean(y_true_snapped == y_pred_snapped))

    qwk_true = np.array([SCORE_TO_CLASS[float(x)] for x in y_true_snapped], dtype=int)
    qwk_pred = np.array([SCORE_TO_CLASS[float(x)] for x in y_pred_snapped], dtype=int)
    qwk = cohen_kappa_score(qwk_true, qwk_pred, weights="quadratic")

    return {
        "QWK":      qwk,
        "MAE":      mae,
        "Spearman": float(spearman),
        "ACC":      acc,
    }


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold standard")
    parser.add_argument("--cfg", default=None, help="YAML config file (optional)")
    parser.add_argument("--pred", default=None, help="Prediction file (overrides config)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    GOLD_MAP = {
        "asean": repo_root / "data" / "gold" / "asean" / "Test_Article-gold_standard.json",
        "other": repo_root / "data" / "gold" / "other" / "Test_Article-gold_standard-other.json",
    }

    if args.pred:
        pred_path = Path(args.pred)
    elif args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        pred_path = repo_root / Path(cfg["pred_path"])
    else:
        raise ValueError("You must provide either --pred or --cfg")

    # 自动检测数据集类型，选择对应 gold 标准
    pred_str = str(pred_path).replace("\\", "/")
    if "other" in pred_str:
        dataset = "other"
    else:
        dataset = "asean"
    gold_path = GOLD_MAP[dataset]
    print(f"Dataset  : {dataset}")
    print(f"Gold     : {gold_path}")

    dims = ["obligation", "precision", "delegation"]

    df_gold = pd.read_json(gold_path)
    df_pred = pd.read_json(pred_path, lines=True)

    use_cols = ["id"] + dims
    df = (
        df_gold[use_cols]
        .merge(df_pred[use_cols], on="id", suffixes=("_true", "_pred"))
        .sort_values("id")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No overlapping IDs between gold and pred files")

    y_true_all = np.concatenate([df[f"{d}_true"].to_numpy(dtype=float) for d in dims])
    y_pred_all = np.concatenate([df[f"{d}_pred"].to_numpy(dtype=float) for d in dims])

    metrics = compute_metrics(y_true_all, y_pred_all)

    print("=== Overall Metrics ===")
    for k, v in metrics.items():
        print(f"{k:24s}: {v:.4f}")


if __name__ == "__main__":
    main()
