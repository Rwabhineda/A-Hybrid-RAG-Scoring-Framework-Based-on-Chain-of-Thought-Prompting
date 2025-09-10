import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score

THRESHOLD = 0.75     
TOL = 0.25           
DIMS = ['obligation','precision','delegation']  

def icc_2_1(y_true, y_pred):
    """Two-way random, absolute agreement, single rater: ICC(2,1)."""
    X = np.vstack([y_true, y_pred]).T
    n, k = X.shape  # k=2
    mean_raters = np.mean(X, axis=0)
    mean_subjects = np.mean(X, axis=1)
    grand_mean = np.mean(X)

    MS_subject = (k / (n - 1)) * np.sum((mean_subjects - grand_mean) ** 2)
    MS_rater   = (n / (k - 1)) * np.sum((mean_raters   - grand_mean) ** 2)
    MS_res     = (1 / ((n - 1) * (k - 1))) * np.sum(
        (X - mean_subjects[:, None] - mean_raters + grand_mean) ** 2
    )
    return (MS_subject - MS_res) / (MS_subject + (k - 1) * MS_res + (k / n) * (MS_rater - MS_res))

def compute_metrics(y_true, y_pred, thr=THRESHOLD):
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

gold_path = r'C:\Users\zzh69\Desktop\国际文书的法律化程度：基于检索增强生成技术（RAG）的文书条款评分方法\DATA Analyze\data\gold_standard\Test_Article-gold_standard.jsonl'
pred_path = r"C:\Users\zzh69\Desktop\国际文书的法律化程度：基于检索增强生成技术（RAG）的文书条款评分方法\RAG Databases\Scoring Code\gpt-5\Test_Article-gpt5.jsonl"

df_gold = pd.read_json(gold_path, lines=True)
df_pred = pd.read_json(pred_path, lines=True)

use_cols = ['id'] + DIMS
df = (df_gold[use_cols]
      .merge(df_pred[use_cols], on='id', suffixes=('_true','_pred'))
      .sort_values('id')
      .reset_index(drop=True))

y_true_all = np.concatenate([df[f'{d}_true'].to_numpy(dtype=float) for d in DIMS])
y_pred_all = np.concatenate([df[f'{d}_pred'].to_numpy(dtype=float) for d in DIMS])

metrics = compute_metrics(y_true_all, y_pred_all, thr=THRESHOLD)

print("=== Overall Metrics ===")
for k, v in metrics.items():
    print(f"{k:24s}: {v:.4f}")
