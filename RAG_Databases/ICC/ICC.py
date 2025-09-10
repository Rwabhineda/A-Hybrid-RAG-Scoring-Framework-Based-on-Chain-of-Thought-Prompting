import json
import pandas as pd
import numpy as np
from scipy import stats
import pingouin as pg
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def load_jsonl_file(file_path: str) -> pd.DataFrame:
    """读取JSONL文件并转换为DataFrame"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return pd.DataFrame(data)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return pd.DataFrame()

def match_data(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    """匹配两个专家的评分数据"""
    # 以id为key进行匹配
    merged_df = df_a.merge(df_b, on='id', how='inner', suffixes=('_expert1', '_expert2'))
    print(f"专家1数据条数: {len(df_a)}")
    print(f"专家2数据条数: {len(df_b)}")
    print(f"匹配成功的条数: {len(merged_df)}")
    return merged_df

def calculate_icc(data: pd.DataFrame, dimension: str) -> Dict:
    """计算ICC及其置信区间"""
    try:
        # 准备数据格式 - pingouin需要长格式数据
        expert1_scores = data[f'{dimension}_expert1'].values
        expert2_scores = data[f'{dimension}_expert2'].values
        
        # 创建长格式数据
        n_items = len(expert1_scores)
        long_data = pd.DataFrame({
            'item': list(range(n_items)) * 2,
            'rater': ['expert1'] * n_items + ['expert2'] * n_items,
            'score': np.concatenate([expert1_scores, expert2_scores])
        })
        
        # 计算ICC(2,1) - Two-way random effects, single measurement, absolute agreement
        icc_result = pg.intraclass_corr(data=long_data, targets='item', raters='rater', 
                                       ratings='score', nan_policy='omit')
        
        # 获取ICC(2,1)的结果 (通常是第2行，索引为1)
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
        print(f"计算 {dimension} 的ICC时出错: {e}")
        return None

def interpret_icc(icc_value: float) -> str:
    """解释ICC值"""
    if icc_value < 0.5:
        return "Poor reliability"
    elif icc_value < 0.75:
        return "Moderate reliability"
    elif icc_value < 0.9:
        return "Good reliability"
    else:
        return "Excellent reliability"

def calculate_other_metrics(data: pd.DataFrame, dimension: str) -> Dict:
    """计算其他一致性指标"""
    expert1_scores = data[f'{dimension}_expert1'].values
    expert2_scores = data[f'{dimension}_expert2'].values
    
    # 去除缺失值
    mask = ~(pd.isna(expert1_scores) | pd.isna(expert2_scores))
    expert1_clean = expert1_scores[mask]
    expert2_clean = expert2_scores[mask]
    
    if len(expert1_clean) == 0:
        return None
    
    # Pearson相关系数
    pearson_r, pearson_p = stats.pearsonr(expert1_clean, expert2_clean)
    
    # Spearman相关系数
    spearman_r, spearman_p = stats.spearmanr(expert1_clean, expert2_clean)
    
    # Cohen's Kappa (将连续评分视为有序分类)
    try:
        kappa = cohen_kappa_score(expert1_clean, expert2_clean)
    except:
        kappa = np.nan
    
    # 绝对差异统计
    abs_diff = np.abs(expert1_clean - expert2_clean)
    mean_abs_diff = np.mean(abs_diff)
    std_abs_diff = np.std(abs_diff)
    
    # 完全一致的比例
    exact_agreement = np.mean(expert1_clean == expert2_clean)
    
    # 在容忍范围内一致的比例（±0.25）
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
    """计算整体ICC（所有维度合并）"""
    dimensions = ['obligation', 'precision', 'delegation']
    all_scores = []
    
    for dim in dimensions:
        expert1_scores = data[f'{dim}_expert1'].values
        expert2_scores = data[f'{dim}_expert2'].values
        
        # 创建长格式数据
        n_items = len(expert1_scores)
        for i in range(n_items):
            if not (pd.isna(expert1_scores[i]) or pd.isna(expert2_scores[i])):
                all_scores.append({
                    'item': f"{dim}_{i}",
                    'rater': 'expert1',
                    'score': expert1_scores[i]
                })
                all_scores.append({
                    'item': f"{dim}_{i}",
                    'rater': 'expert2', 
                    'score': expert2_scores[i]
                })
    
    if not all_scores:
        return None
    
    long_data = pd.DataFrame(all_scores)
    
    try:
        icc_result = pg.intraclass_corr(data=long_data, targets='item', raters='rater',
                                       ratings='score', nan_policy='omit')
        icc_21 = icc_result.iloc[1]  # ICC(2,1)
        
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
        print(f"计算整体ICC时出错: {e}")
        return None

def create_visualizations(data: pd.DataFrame):
    """创建可视化图表"""
    dimensions = ['obligation', 'precision', 'delegation']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, dim in enumerate(dimensions):
        expert1 = data[f'{dim}_expert1']
        expert2 = data[f'{dim}_expert2']
        
        # 散点图
        axes[0, i].scatter(expert1, expert2, alpha=0.6)
        axes[0, i].plot([0, 1], [0, 1], 'r--', alpha=0.8)
        axes[0, i].set_xlabel(f'Expert 1 - {dim.title()}')
        axes[0, i].set_ylabel(f'Expert 2 - {dim.title()}')
        axes[0, i].set_title(f'{dim.title()} Agreement')
        axes[0, i].grid(True, alpha=0.3)
        
        # 差异分布
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
    # 文件路径
    file_A = r"../Expert1 Scored/Scored Artlcie-Expert1.json"
    file_B = r"../Expert2 Scored/Scored Artlcie-Expert2.json"
    
    print("=" * 60)
    print("专家评分一致性分析报告")
    print("=" * 60)
    
    # 读取数据
    print("\n1. 读取数据...")
    df_chatgpt = load_jsonl_file(file_A)
    df_deepseek = load_jsonl_file(file_B)
    
    if df_chatgpt.empty or df_deepseek.empty:
        print("数据读取失败，请检查文件路径")
        return
    
    # 匹配数据
    print("\n2. 匹配数据...")
    matched_data = match_data(df_chatgpt, df_deepseek)
    
    if matched_data.empty:
        print("没有匹配的数据")
        return
    
    print(f"\n最终参与ICC计算的条款ID数量: {len(matched_data)}")
    
    # 分析每个维度
    dimensions = ['obligation', 'precision', 'delegation']
    results = {}
    
    print("\n3. 计算各维度ICC...")
    print("-" * 60)
    
    for dim in dimensions:
        print(f"\n{dim.upper()} 维度分析:")
        
        # 计算ICC
        icc_result = calculate_icc(matched_data, dim)
        if icc_result:
            print(f"  ICC(2,1): {icc_result['ICC']:.4f}")
            print(f"  95% CI: [{icc_result['CI95_lower']:.4f}, {icc_result['CI95_upper']:.4f}]")
            print(f"  F-统计量: {icc_result['F']:.4f}")
            print(f"  p-值: {icc_result['pval']:.6f}")
            print(f"  解释: {icc_result['interpretation']}")
        
        # 计算其他指标
        other_metrics = calculate_other_metrics(matched_data, dim)
        if other_metrics:
            print(f"  Pearson相关: r={other_metrics['pearson_r']:.4f}, p={other_metrics['pearson_p']:.6f}")
            print(f"  Spearman相关: ρ={other_metrics['spearman_r']:.4f}, p={other_metrics['spearman_p']:.6f}")
            print(f"  Cohen's Kappa: {other_metrics['cohens_kappa']:.4f}")
            print(f"  平均绝对差异: {other_metrics['mean_abs_diff']:.4f} ± {other_metrics['std_abs_diff']:.4f}")
            print(f"  完全一致率: {other_metrics['exact_agreement_rate']:.1%}")
            print(f"  容忍一致率(±0.25): {other_metrics['tolerance_agreement_rate']:.1%}")
            print(f"  有效配对数: {other_metrics['n_valid_pairs']}")
        
        results[dim] = {'icc': icc_result, 'other': other_metrics}
    
    # 计算整体ICC
    print("\n4. 计算整体ICC...")
    print("-" * 60)
    overall_icc = calculate_overall_icc(matched_data)
    if overall_icc:
        print(f"\n整体ICC(2,1): {overall_icc['ICC']:.4f}")
        print(f"95% CI: [{overall_icc['CI95_lower']:.4f}, {overall_icc['CI95_upper']:.4f}]")
        print(f"F-统计量: {overall_icc['F']:.4f}")
        print(f"p-值: {overall_icc['pval']:.6f}")
        print(f"解释: {overall_icc['interpretation']}")
        print(f"观测数量: {overall_icc['n_observations']}")
    
    # 描述性统计
    print("\n5. 描述性统计...")
    print("-" * 60)
    for dim in dimensions:
        expert1_mean = matched_data[f'{dim}_expert1'].mean()
        expert2_mean = matched_data[f'{dim}_expert2'].mean()
        expert1_std = matched_data[f'{dim}_expert1'].std()
        expert2_std = matched_data[f'{dim}_expert2'].std()
        
        print(f"\n{dim.upper()}:")
        print(f"  专家1: 均值={expert1_mean:.3f}, 标准差={expert1_std:.3f}")
        print(f"  专家2: 均值={expert2_mean:.3f}, 标准差={expert2_std:.3f}")
    
    # 创建可视化
    print("\n6. 生成可视化图表...")
    create_visualizations(matched_data)
    
    print("\n" + "=" * 60)
    print("分析完成！图表已保存为 'expert_agreement_analysis.png'")
    print("=" * 60)

if __name__ == "__main__":
    main()