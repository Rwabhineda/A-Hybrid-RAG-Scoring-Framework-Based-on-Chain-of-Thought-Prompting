# -*- coding: utf-8 -*-
"""
Batch evaluation script for clause scoring.
Evaluates all prediction files in a directory structure.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from scipy.stats import spearmanr


# 五级评分映射
SCORE_TO_CLASS = {0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
VALID_SCORES = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=float)


def snap_to_valid(arr: np.ndarray) -> np.ndarray:
    """将预测值吸附到最近的合法档位 {0, 0.25, 0.5, 0.75, 1.0}。"""
    return np.array([VALID_SCORES[np.argmin(np.abs(VALID_SCORES - x))] for x in arr])


def compute_metrics(y_true, y_pred) -> dict:
    """Compute evaluation metrics: QWK, MAE, Spearman, ACC."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 2:
        return {"QWK": 0.0, "MAE": 0.0, "Spearman": 0.0, "ACC": 0.0}

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


def evaluate_single(gold_df: pd.DataFrame, pred_path: Path, dims: List[str]) -> Optional[dict]:
    """Evaluate a single prediction file."""
    if not pred_path.exists():
        return None
    
    try:
        df_pred = pd.read_json(pred_path, lines=True)
    except Exception as e:
        print(f"  [ERROR] Failed to read {pred_path.name}: {e}")
        return None
    
    use_cols = ["id"] + dims
    if not all(c in df_pred.columns for c in use_cols):
        print(f"  [ERROR] Missing columns in {pred_path.name}")
        return None
    
    df = (
        gold_df[use_cols]
        .merge(df_pred[use_cols], on="id", suffixes=("_true", "_pred"))
        .sort_values("id")
        .reset_index(drop=True)
    )
    
    if df.empty:
        return None
    
    y_true_all = np.concatenate([df[f"{d}_true"].to_numpy(dtype=float) for d in dims])
    y_pred_all = np.concatenate([df[f"{d}_pred"].to_numpy(dtype=float) for d in dims])
    
    return compute_metrics(y_true_all, y_pred_all)


def detect_dataset(path_str: str) -> str:
    """Detect dataset type from path."""
    return "other" if "other" in path_str.replace("\\", "/") else "asean"


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate predictions")
    parser.add_argument("--dir", "-d", default=None, help="Directory to scan (default: outputs/other)")
    parser.add_argument("--output", "-o", default=None, help="Output CSV file (optional)")
    parser.add_argument("--runs", nargs="+", default=["run1", "run2"], help="Run suffixes to evaluate")
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[2]
    
    # Default directory
    target_dir = Path(args.dir) if args.dir else repo_root / "outputs" / "other"
    if not target_dir.is_absolute():
        target_dir = repo_root / target_dir
    
    # Detect dataset and load gold
    dataset = detect_dataset(str(target_dir))
    gold_map = {
        "asean": repo_root / "data" / "gold" / "asean" / "Test_Article-gold_standard.json",
        "other": repo_root / "data" / "gold" / "other" / "Test_Article-gold_standard-other.json",
    }
    gold_path = gold_map[dataset]
    
    print("=" * 60)
    print(f"Batch Evaluation")
    print(f"Directory : {target_dir}")
    print(f"Dataset   : {dataset}")
    print(f"Gold      : {gold_path}")
    print("=" * 60)
    
    dims = ["obligation", "precision", "delegation"]
    df_gold = pd.read_json(gold_path)
    
    # Collect all results
    results = []
    
    # Scan subdirectories for model/mode structure
    for model_dir in sorted(target_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        
        # Check if it's a model directory with modes (base/cot/full/rag)
        mode_dirs = [d for d in model_dir.iterdir() if d.is_dir()]
        has_modes = any(d.name in ["base", "cot", "full", "rag"] for d in mode_dirs)
        
        if has_modes:
            # Standard structure: model/mode/results-run*.jsonl
            for mode_dir in sorted(mode_dirs):
                if mode_dir.name not in ["base", "cot", "full", "rag"]:
                    continue
                mode_name = mode_dir.name
                
                # Evaluate each run
                for run in args.runs:
                    pred_file = mode_dir / f"results-{run}.jsonl"
                    metrics = evaluate_single(df_gold, pred_file, dims)
                    if metrics:
                        results.append({
                            "model": model_name,
                            "mode": mode_name,
                            "run": run,
                            **metrics
                        })
        else:
            # Baseline structure: model/results.jsonl
            pred_file = model_dir / "results.jsonl"
            metrics = evaluate_single(df_gold, pred_file, dims)
            if metrics:
                results.append({
                    "model": model_name,
                    "mode": "-",
                    "run": "-",
                    **metrics
                })
    
    if not results:
        print("No results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print results table
    print("\n" + "=" * 80)
    print("Results Summary (Mean ± Std)")
    print("=" * 80)
    
    modes = ["base", "cot", "full", "rag"]
    
    for metric in ["QWK", "MAE", "Spearman", "ACC"]:
        print(f"\n--- {metric} ---")
        print(f"{'Model':20s}  {'base':>14s}  {'cot':>14s}  {'full':>14s}  {'rag':>14s}")
        print("-" * 80)
        
        for model in sorted(df["model"].unique()):
            model_df = df[df["model"] == model]
            row = [model]
            for mode in modes:
                mode_df = model_df[model_df["mode"] == mode]
                if len(mode_df) > 0:
                    mean_val = mode_df[metric].mean()
                    std_val = mode_df[metric].std() if len(mode_df) > 1 else 0.0
                    row.append(f"{mean_val:.4f}±{std_val:.4f}")
                else:
                    row.append("-")
            print(f"{row[0]:20s}  {row[1]:>14s}  {row[2]:>14s}  {row[3]:>14s}  {row[4]:>14s}")
    
    # Print detailed table
    print("\n" + "=" * 80)
    print("Detailed Results")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # Save to CSV if requested
    if args.output:
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
