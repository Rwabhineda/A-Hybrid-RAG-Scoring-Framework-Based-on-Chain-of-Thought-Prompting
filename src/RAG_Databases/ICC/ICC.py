import json
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ================= Path Settings =================
REPO_ROOT = Path(__file__).resolve().parents[2]

file_A = REPO_ROOT / "RAG_Databases" / "Expert1 Scored" / "Scored Article-Expert1.jsonl"
file_B = REPO_ROOT / "RAG_Databases" / "Expert2 Scored" / "Scored Article-Expert2.jsonl"

if not file_A.exists():
    raise FileNotFoundError(f"Expert1 file not found: {file_A}")
if not file_B.exists():
    raise FileNotFoundError(f"Expert2 file not found: {file_B}")


# ================= Functions =================
def load_jsonl_file(file_path: str) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return pd.DataFrame()


def match_data(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """Match two expert datasets by id"""
    merged_df = df_a.merge(df_b, on='id', how='inner', suffixes=('_expert1', '_expert2'))
    print(f"Expert1 sample count: {len(df_a)}")
    print(f"Expert2 sample count: {len(df_b)}")
    print(f"Matched sample count: {len(merged_df)}")
    return merged_df


def calculate_icc(data: pd.DataFrame, dimension: str) -> Dict:
    """Compute ICC(2,1) and confidence interval for a given dimension"""
    try:
        expert1_scores = data[f'{dimension}_expert1'].values
        expert2_scores = data[f'{dimension}_expert2'].values

        n_items = len(expert1_scores)
        long_data = pd.DataFrame({
            'item': list(range(n_items)) * 2,
            'rater': ['expert1'] * n_items + ['expert2'] * n_items,
            'score': np.concatenate([expert1_scores, expert2_scores])
        })

        icc_result = pg.intraclass_corr(
            data=long_data, targets='item', raters='rater',
            ratings='score', nan_policy='omit'
        )
        icc_21 = icc_result.iloc[1]  # ICC(2,1)

        return {
            'ICC': icc_21['ICC'],
            'CI95_lower': icc_21['CI95%'][0],
            'CI95_upper': icc_21['CI95%'][1],
            'F': icc_21['F'],
            'pval': icc_21['pval'],
            'interpretation': interpret_icc(icc_21['ICC'])
        }
    except Exception as e:
        print(f"Error calculating ICC for {dimension}: {e}")
        return None


def interpret_icc(icc_value: float) -> str:
    """Interpret ICC value"""
    if icc_value < 0.5:
        return "Poor reliability"
    elif icc_value < 0.75:
        return "Moderate reliability"
    elif icc_value < 0.9:
        return "Good reliability"
    else:
        return "Excellent reliability"


def calculate_other_metrics(data: pd.DataFrame, dimension: str) -> Dict:
    """Compute additional agreement metrics"""
    expert1_scores = data[f'{dimension}_expert1'].values
    expert2_scores = data[f'{dimension}_expert2'].values

    mask = ~(pd.isna(expert1_scores) | pd.isna(expert2_scores))
    expert1_clean = expert1_scores[mask]
    expert2_clean = expert2_scores[mask]

    if len(expert1_clean) == 0:
        return None

    pearson_r, pearson_p = stats.pearsonr(expert1_clean, expert2_clean)
    spearman_r, spearman_p = stats.spearmanr(expert1_clean, expert2_clean)

    try:
        kappa = cohen_kappa_score(expert1_clean, expert2_clean)
    except Exception:
        kappa = np.nan

    abs_diff = np.abs(expert1_clean - expert2_clean)
    mean_abs_diff = np.mean(abs_diff)
    std_abs_diff = np.std(abs_diff)

    exact_agreement = np.mean(expert1_clean == expert2_clean)
    tolerance_agreement = np.mean(abs_diff <= 0.25)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'cohens_kappa': kappa,
        'mean_abs_diff': mean_abs_diff,
        'std_abs_diff': std_abs_diff,
        'exact_agreement_rate': exact_agreement,
        'tolerance_agreement_rate': tolerance_agreement,
        'n_valid_pairs': len(expert1_clean)
    }


def calculate_overall_icc(data: pd.DataFrame) -> Dict:
    """Compute overall ICC across all dimensions"""
    dimensions = ['obligation', 'precision', 'delegation']
    all_scores = []

    for dim in dimensions:
        expert1_scores = data[f'{dim}_expert1'].values
        expert2_scores = data[f'{dim}_expert2'].values
        n_items = len(expert1_scores)
        for i in range(n_items):
            if not (pd.isna(expert1_scores[i]) or pd.isna(expert2_scores[i])):
                all_scores.append({'item': f"{dim}_{i}", 'rater': 'expert1', 'score': expert1_scores[i]})
                all_scores.append({'item': f"{dim}_{i}", 'rater': 'expert2', 'score': expert2_scores[i]})

    if not all_scores:
        return None

    long_data = pd.DataFrame(all_scores)
    try:
        icc_result = pg.intraclass_corr(
            data=long_data, targets='item', raters='rater',
            ratings='score', nan_policy='omit'
        )
        icc_21 = icc_result.iloc[1]
        return {
            'ICC': icc_21['ICC'],
            'CI95_lower': icc_21['CI95%'][0],
            'CI95_upper': icc_21['CI95%'][1],
            'F': icc_21['F'],
            'pval': icc_21['pval'],
            'interpretation': interpret_icc(icc_21['ICC']),
            'n_observations': len(all_scores) // 2
        }
    except Exception as e:
        print(f"Error calculating overall ICC: {e}")
        return None


def create_visualizations(data: pd.DataFrame):
    """Generate visualization plots"""
    dimensions = ['obligation', 'precision', 'delegation']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i, dim in enumerate(dimensions):
        expert1 = data[f'{dim}_expert1']
        expert2 = data[f'{dim}_expert2']

        axes[0, i].scatter(expert1, expert2, alpha=0.6)
        axes[0, i].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[0, i].set_xlabel(f'Expert 1 - {dim.title()}')
        axes[0, i].set_ylabel(f'Expert 2 - {dim.title()}')
        axes[0, i].set_title(f'{dim.title()} Agreement')
        axes[0, i].grid(True, alpha=0.3)

        diff = expert2 - expert1
        axes[1, i].hist(diff, bins=15, alpha=0.7, edgecolor='black')
        axes[1, i].axvline(0, color='red', linestyle='--', alpha=0.8)
        axes[1, i].set_xlabel(f'Score Difference (Expert2 - Expert1)')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].set_title(f'{dim.title()} Difference Distribution')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('expert_agreement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("=" * 60)
    print("Expert Agreement Analysis Report")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    df_exp1 = load_jsonl_file(file_A)
    df_exp2 = load_jsonl_file(file_B)

    if df_exp1.empty or df_exp2.empty:
        print("Data loading failed. Please check file paths.")
        return

    # Match data
    print("\n2. Matching data...")
    matched_data = match_data(df_exp1, df_exp2)

    if matched_data.empty:
        print("No matched data found.")
        return

    print(f"\nFinal matched item count for ICC calculation: {len(matched_data)}")

    # Per-dimension analysis
    dimensions = ['obligation', 'precision', 'delegation']

    print("\n3. Calculating ICC per dimension...")
    print("-" * 60)

    for dim in dimensions:
        print(f"\n{dim.upper()} Dimension Analysis:")

        icc_result = calculate_icc(matched_data, dim)
        if icc_result:
            print(f"  ICC(2,1): {icc_result['ICC']:.4f}")
            print(f"  95% CI: [{icc_result['CI95_lower']:.4f}, {icc_result['CI95_upper']:.4f}]")
            print(f"  F-statistic: {icc_result['F']:.4f}")
            print(f"  p-value: {icc_result['pval']:.6f}")
            print(f"  Interpretation: {icc_result['interpretation']}")

        other_metrics = calculate_other_metrics(matched_data, dim)
        if other_metrics:
            print(f"  Pearson correlation: r={other_metrics['pearson_r']:.4f}, p={other_metrics['pearson_p']:.6f}")
            print(f"  Spearman correlation: ρ={other_metrics['spearman_r']:.4f}, p={other_metrics['spearman_p']:.6f}")
            print(f"  Cohen's Kappa: {other_metrics['cohens_kappa']:.4f}")
            print(f"  Mean absolute diff: {other_metrics['mean_abs_diff']:.4f} ± {other_metrics['std_abs_diff']:.4f}")
            print(f"  Exact agreement rate: {other_metrics['exact_agreement_rate']:.1%}")
            print(f"  Tolerance agreement rate (±0.25): {other_metrics['tolerance_agreement_rate']:.1%}")
            print(f"  Valid pairs: {other_metrics['n_valid_pairs']}")

    # Overall ICC
    print("\n4. Calculating overall ICC...")
    print("-" * 60)
    overall_icc = calculate_overall_icc(matched_data)
    if overall_icc:
        print(f"\nOverall ICC(2,1): {overall_icc['ICC']:.4f}")
        print(f"95% CI: [{overall_icc['CI95_lower']:.4f}, {overall_icc['CI95_upper']:.4f}]")
        print(f"F-statistic: {overall_icc['F']:.4f}")
        print(f"p-value: {overall_icc['pval']:.6f}")
        print(f"Interpretation: {overall_icc['interpretation']}")
        print(f"Observations: {overall_icc['n_observations']}")

    print("\n5. Descriptive statistics...")
    print("-" * 60)
    for dim in dimensions:
        expert1_mean = matched_data[f'{dim}_expert1'].mean()
        expert2_mean = matched_data[f'{dim}_expert2'].mean()
        expert1_std = matched_data[f'{dim}_expert1'].std()
        expert2_std = matched_data[f'{dim}_expert2'].std()

        print(f"\n{dim.upper()}:")
        print(f"  Expert1: mean={expert1_mean:.3f}, std={expert1_std:.3f}")
        print(f"  Expert2: mean={expert2_mean:.3f}, std={expert2_std:.3f}")

    print("\n6. Generating visualizations...")
    create_visualizations(matched_data)

    print("\n" + "=" * 60)
    print("Analysis completed! Figure saved as 'expert_agreement_analysis.png'")
    print("=" * 60)


if __name__ == "__main__":
    main()
