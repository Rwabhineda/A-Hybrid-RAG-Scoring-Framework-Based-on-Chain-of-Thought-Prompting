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
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score


# ---------------- Metrics ----------------
def icc_2_1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Two-way random, absolute agreement, single rater: ICC(2,1)."""
    X = np.vstack([y_true, y_pred]).T
    n, k = X.shape
    mean_raters = np.mean(X, axis=0)
    mean_subjects = np.mean(X, axis=1)
    grand_mean = np.mean(X)

    MS_subject = (k / (n - 1)) * np.sum((mean_subjects - grand_mean) ** 2)
    MS_rater = (n / (k - 1)) * np.sum((mean_raters - grand_mean) ** 2)
    MS_res = (1 / ((n - 1) * (k - 1))) * np.sum(
        (X - mean_subjects[:, None] - mean_raters + grand_mean) ** 2
    )
    icc = (MS_subject - MS_res) / (
        MS_subject + (k - 1) * MS_res + (k / n) * (MS_rater - MS_res)
    )
    return float(max(min(icc, 1.0), -1.0))


def compute_metrics(y_true, y_pred, thr=0.75) -> dict:
    """Compute evaluation metrics: ICC(2,1), MAE, Exact Agreement, Recall, Precision, F1."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    icc = icc_2_1(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    exact_agree = float(np.mean(y_true == y_pred))

    y_true_bin = (y_true >= thr).astype(int)
    y_pred_bin = (y_pred >= thr).astype(int)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    return {
        "ICC(2,1)": icc,
        "MAE": mae,
        "Exact Agreement Rate": exact_agree,
        f"Recall@{thr}": recall,
        f"Precision@{thr}": precision,
        f"F1-Score@{thr}": f1,
    }


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against gold standard")
    parser.add_argument("--cfg", default=None, help="YAML config file (optional)")
    parser.add_argument("--pred", default=None, help="Prediction file (overrides config)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    gold_path = repo_root / "src" / "DATA_Analyze" / "data" / "gold_standard" / "Test_Article-gold_standard.jsonl"

    if args.pred:
        pred_path = Path(args.pred)
    elif args.cfg:
        with open(args.cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        pred_path = repo_root / Path(cfg["pred_path"])
    else:
        raise ValueError("You must provide either --pred or --cfg")

    dims = ["obligation", "precision", "delegation"]
    threshold = 0.75

    df_gold = pd.read_json(gold_path, lines=True)
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

    metrics = compute_metrics(y_true_all, y_pred_all, thr=threshold)

    print("=== Overall Metrics ===")
    for k, v in metrics.items():
        print(f"{k:24s}: {v:.4f}")


if __name__ == "__main__":
    main()
