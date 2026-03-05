# -*- coding: utf-8 -*-
"""Compare two scoring results."""

import json
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    """Load jsonl file."""
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                # Use text as key for matching
                data[d['text'].strip()] = d
    return data


def main():
    repo_root = Path(__file__).resolve().parent
    
    # File paths
    file1 = repo_root / "outputs" / "gpt-5.2" / "full" / "results.jsonl"
    file2 = repo_root / "The Legalization of Internation" / "new_project" / "Test_Article_scored.jsonl"
    
    # Load data
    data1 = load_jsonl(file1)  # gpt-5.2 full
    data2 = load_jsonl(file2)  # Test_Article_scored
    
    dims = ['obligation', 'precision', 'delegation']
    
    # Find common texts
    common_texts = set(data1.keys()) & set(data2.keys())
    print(f"File 1 (gpt-5.2 full): {len(data1)} clauses")
    print(f"File 2 (Test_Article_scored): {len(data2)} clauses")
    print(f"Common clauses: {len(common_texts)}\n")
    
    # Compare scores
    diff_count = 0
    diff_by_dim = defaultdict(int)
    total_diff_by_dim = defaultdict(float)
    
    for text in common_texts:
        d1 = data1[text]
        d2 = data2[text]
        
        for dim in dims:
            s1 = d1[dim]
            s2 = d2[dim]
            
            if s1 != s2:
                diff_count += 1
                diff_by_dim[dim] += 1
                total_diff_by_dim[dim] += abs(s1 - s2)
    
    total_scores = len(common_texts) * 3
    
    print("=== Score Comparison ===")
    print(f"Total score comparisons: {total_scores}")
    print(f"Differences: {diff_count} ({diff_count/total_scores*100:.2f}%)\n")
    
    print("=== Differences by Dimension ===")
    for dim in dims:
        count = diff_by_dim[dim]
        total = len(common_texts)
        avg_diff = total_diff_by_dim[dim] / count if count > 0 else 0
        print(f"{dim:12s}: {count:3d}/{total} ({count/total*100:5.2f}%) - Avg diff: {avg_diff:.2f}")
    
    # Show some examples
    print("\n=== Example Differences (first 5) ===")
    shown = 0
    for text in common_texts:
        d1 = data1[text]
        d2 = data2[text]
        
        diffs = []
        for dim in dims:
            if d1[dim] != d2[dim]:
                diffs.append(f"{dim}: {d1[dim]} vs {d2[dim]}")
        
        if diffs and shown < 5:
            print(f"\nID: {d1['id']} vs {d2['id']}")
            print(f"Text: {text[:100]}...")
            print(f"Differences: {', '.join(diffs)}")
            shown += 1


if __name__ == "__main__":
    main()
