# A Hybrid RAG Scoring Framework Based on Chain-of-Thought Prompting

[![CI](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/Rwabhineda/A-Hybrid-RAG-Scoring-Framework-Based-on-Chain-of-Thought-Prompting/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](./The%20Legalization%20of%20International%20Instruments%20A%20Hybrid%20RAG%20Scoring%20Framework%20Based%20on%20Chain-of-Thought%20Prompting.pdf)

This repository contains code and data for a project that explores **Retrieval-Augmented Generation (RAG)** combined with **Chain-of-Thought (CoT) prompting** to evaluate the legalization degree of international legal instruments.

> **The Legalization of International Instruments: A Hybrid RAG Scoring Framework Based on Chain-of-Thought Prompting**  
> Yan Chen, Zihua Zeng, Muhamad Sayuti Hassan

---

## 🔎 Overview

This repository provides a reproducible pipeline for **clause-level scoring** of international instruments using a **Hybrid RAG** (retrieval + quality-weighted ranking + CE filtering) combined with **Chain-of-Thought (CoT) prompting**.

**Dimensions**: Obligation (O), Precision (P), Delegation (D)  
**Scale**: Five-point scores in {0.0, 0.25, 0.5, 0.75, 1.0} with stepwise decision rules  
**Data**: 2,611 expert-annotated ASEAN clauses (254-clause test set) + 255-clause African Union transfer set  
**Models**: GPT-3.5-Turbo, GPT-4o, GPT-4o-mini, GPT-5.2, Legal-BERT, TF-IDF+LR  
**Best**: GPT-5.2 Full → ICC **0.8223**, MAE **0.0886** | GPT-4o Full → F1@0.75 **0.7441**

---

## ✨ Highlights

- **Hybrid RAG**: dense retrieval (ChromaDB, E5-large-v2) + quality-weighted reranking
- **Legal-BERT Filtering**: Cross-Encoder binary relevance gate for RAG results
- **CoT Prompting**: stepwise rubric + few-shot exemplars (Top-K) + robust output parsing
- **Ablation Study**: 4 modes (base/rag/cot/full) × multiple LLMs for systematic comparison
- **Transfer Learning**: Cross-domain evaluation on African Union legal documents
- **Baselines**: Legal-BERT and TF-IDF+LR traditional models for comparison
- **Evaluation**: ICC(2,1), MAE, Exact Agreement, Recall/Precision/F1@0.75
- **Reproducible**: config-driven CLI, fixed seeds, organized output structure


---

## ⚙️ Installation

```bash
git clone https://github.com/Rwabhineda/A-Hybrid-RAG-Scoring-Framework-Based-on-Chain-of-Thought-Prompting.git
cd <YOUR_REPO>

# Python >=3.10 recommended
conda create -n ragcot python=3.10 -y
conda activate ragcot
pip install -r requirements.txt
```

---

## 🚀 Usage

### Running Experiments

```bash
# In-domain (ASEAN) - Full configuration
uv run python src/main.py --config configs/asean/gpt-5.2/gpt-5.2.yaml

# In-domain - Ablation study
uv run python src/main.py --config configs/asean/gpt-4o/gpt-4o-base.yaml
uv run python src/main.py --config configs/asean/gpt-4o/gpt-4o-rag.yaml
uv run python src/main.py --config configs/asean/gpt-4o/gpt-4o-cot.yaml

# Cross-domain (Transfer) - African Union data
uv run python src/main.py --config configs/transfer/gpt-5.2/gpt-5.2.yaml
```

### Evaluation

```bash
# Evaluate against gold standard
uv run python src/evaluation/eval.py --pred outputs/asean/gpt-5.2/full/results.jsonl
```

---

## 📁 Project Structure

```
.
├── configs/                    # Experiment configurations
│   ├── asean/                  # In-domain experiments (ASEAN)
│   │   ├── gpt-3.5-turbo/
│   │   ├── gpt-4o/
│   │   ├── gpt-4o-mini/
│   │   └── gpt-5.2/
│   ├── transfer/               # Cross-domain experiments (African Union)
│   │   ├── gpt-3.5-turbo/
│   │   ├── gpt-4o/
│   │   ├── gpt-4o-mini/
│   │   └── gpt-5.2/
│   └── README.md               # Configuration guide
├── data/
│   ├── gold/
│   │   └── asean/              # Gold standard annotations
│   ├── processed/
│   │   ├── asean/              # ASEAN test articles (254)
│   │   └── transfer/           # African Union articles (255)
│   ├── rag/                    # RAG vector database (ChromaDB)
│   └── cache/                  # API response cache
├── src/
│   ├── main.py                 # Entry point
│   ├── scoring/
│   │   └── engine.py           # Core scoring engine
│   ├── baseline/               # Traditional baselines
│   │   ├── legalbert_scorer.py
│   │   └── tfidf_lr_scorer.py
│   └── evaluation/
│       └── eval.py             # Evaluation script
├── outputs/
│   ├── asean/                  # In-domain results
│   │   ├── gpt-5.2/{base,rag,cot,full}/
│   │   ├── gpt-4o/{base,rag,cot,full}/
│   │   ├── legalbert-baseline/
│   │   └── tfidf-lr-baseline/
│   └── transfer/               # Cross-domain results
├── .env.example                # Environment template
├── pyproject.toml              # UV dependencies
└── README.md                   # This file
```

---

## 🧾 Data Format (JSONL)

Each line is one clause unit:
```
{
  "id": "c1",
  "document_title": "11th-ALMM-3-JS",
  "year": "2020",
  "text": "We, the Labour Ministers/Heads of Delegations of ASEAN Plus Three Countries ...",
  "obligation": 0.0,
  "precision": 0.25,
  "delegation": 0.0,
  "confidence_obligation": 1.0,
  "confidence_precision": 0.75,
  "confidence_delegation": 1.0
}
```

---

## 📊 Results

### In-Domain Evaluation (ASEAN Test Set)

| Model | Mode | ICC(2,1) | MAE | F1@0.75 |
|-------|------|:--------:|:---:|:-------:|
| **GPT-5.2** | **full** | **0.8223** | **0.0886** | 0.6835 |
| GPT-5.2 | cot | 0.7677 | 0.1128 | 0.6150 |
| GPT-5.2 | rag | 0.7182 | 0.1384 | 0.6778 |
| GPT-5.2 | base | 0.6670 | 0.1774 | 0.6471 |
| **GPT-4o** | **full** | 0.7980 | 0.0954 | **0.7441** |
| GPT-4o | cot | 0.7298 | 0.1246 | 0.6167 |
| GPT-4o | rag | 0.7264 | 0.1305 | 0.6960 |
| GPT-4o | base | 0.6919 | 0.1538 | 0.6431 |
| Legal-BERT | baseline | 0.7535 | 0.1290 | 0.6821 |
| TF-IDF+LR | baseline | 0.7194 | 0.1388 | 0.6534 |
| GPT-4o-mini | full | 0.6612 | 0.1594 | 0.6456 |
| GPT-3.5-turbo | rag | 0.5631 | 0.1804 | 0.6161 |

### Ablation Study Modes

| Mode | RAG | COT | Description |
|------|-----|-----|-------------|
| base | ❌ | ❌ | Zero-shot baseline |
| rag | ✅ | ❌ | RAG retrieval only |
| cot | ❌ | ✅ | Chain-of-thought only |
| full | ✅ | ✅ | Complete pipeline |

Exact definitions of metrics and experimental protocols are aligned with the paper (ICC, MAE, Exact, Recall/Precision/F1@0.75).


---
## 📜 License

This repository is licensed under the **MIT License** for the source code and  
the **CC BY-NC 4.0 License** for the dataset and embeddings.  
Please cite the related paper if you use any part of this repository.

© 2025-2026 Zihua Zeng
