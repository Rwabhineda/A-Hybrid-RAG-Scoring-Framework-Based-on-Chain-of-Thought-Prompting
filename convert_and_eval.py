# -*- coding: utf-8 -*-
"""Convert string ID format to numeric ID format and evaluate."""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score


def icc_2_1(y_true, y_pred):
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


def compute_metrics(y_true, y_pred, thr=0.75):
    """Compute evaluation metrics."""
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


def main():
    repo_root = Path(__file__).resolve().parent
    gold_path = repo_root / "data" / "gold" / "Test_Article-gold_standard.jsonl"
    pred_path = repo_root / "The Legalization of Internation" / "new_project" / "Test_Article_scored.jsonl"
    
    dims = ["obligation", "precision", "delegation"]
    
    # Load gold data with numeric IDs
    gold_data = {}
    with open(gold_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            # Use text as key for matching
            gold_data[d['text'].strip()] = d
    
    # Load pred data with string IDs
    pred_data = {}
    with open(pred_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            pred_data[d['text'].strip()] = d
    
    # Match by text content
    matched = []
    for text, gold in gold_data.items():
        if text in pred_data:
            matched.append((gold, pred_data[text]))
    
    print(f"Gold clauses: {len(gold_data)}")
    print(f"Pred clauses: {len(pred_data)}")
    print(f"Matched: {len(matched)}")
    
    # Extract scores
    y_true_all = []
    y_pred_all = []
    for gold, pred in matched:
        for dim in dims:
            y_true_all.append(gold[dim])
            y_pred_all.append(pred[dim])
    
    y_true_all = np.array(y_true_all, dtype=float)
    y_pred_all = np.array(y_pred_all, dtype=float)
    
    metrics = compute_metrics(y_true_all, y_pred_all)
    print(f"\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k:24s}: {v:.4f}")


if __name__ == "__main__":
    main()
