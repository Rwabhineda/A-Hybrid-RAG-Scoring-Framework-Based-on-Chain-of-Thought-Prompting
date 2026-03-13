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
**Best**: GPT-5.2 Full → ASEAN QWK **0.8060**, MAE **0.0895** | Transfer QWK **0.7889**, MAE **0.0840**

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

### In-Domain Evaluation (ASEAN Test Set, n=254)

| Model | Mode | QWK (Mean±SD) | MAE (Mean±SD) | Spearman (Mean±SD) | ACC (Mean±SD) |
|-------|------|:-------------:|:-------------:|:------------------:|:-------------:|
| **GPT-5.2** | **full** | **0.8060±0.0143** | **0.0895±0.0018** | **0.7928±0.0111** | **0.7003±0.0164** |
| GPT-5.2 | cot | 0.7575±0.0091 | 0.1150±0.0038 | 0.7436±0.0081 | 0.5949±0.0148 |
| GPT-5.2 | rag | 0.7194±0.0023 | 0.1368±0.0023 | 0.7357±0.0046 | 0.5584±0.0102 |
| GPT-5.2 | base | 0.6747±0.0081 | 0.1760±0.0027 | 0.7129±0.0085 | 0.4409±0.0039 |
| **GPT-4o** | **full** | 0.7884±0.0080 | 0.0982±0.0030 | 0.7708±0.0087 | 0.6842±0.0087 |
| GPT-4o | cot | 0.7315±0.0076 | 0.1237±0.0039 | 0.6945±0.0109 | 0.5919±0.0146 |
| GPT-4o | rag | 0.7265±0.0034 | 0.1308±0.0010 | 0.7292±0.0018 | 0.5752±0.0020 |
| GPT-4o | base | 0.6862±0.0205 | 0.1562±0.0084 | 0.6890±0.0257 | 0.5052±0.0251 |
| GPT-4o-mini | full | 0.6799±0.0172 | 0.1516±0.0074 | 0.6843±0.0177 | 0.5455±0.0176 |
| GPT-4o-mini | cot | 0.6794±0.0192 | 0.1539±0.0051 | 0.6721±0.0267 | 0.5315±0.0102 |
| GPT-4o-mini | rag | 0.6161±0.0063 | 0.1593±0.0017 | 0.6457±0.0043 | 0.5297±0.0027 |
| GPT-4o-mini | base | 0.5261±0.0131 | 0.2054±0.0043 | 0.5830±0.0116 | 0.4016±0.0107 |
| GPT-3.5-turbo | full | 0.5122±0.0076 | 0.2275±0.0017 | 0.5554±0.0006 | 0.3548±0.0059 |
| GPT-3.5-turbo | cot | 0.5305±0.0148 | 0.2072±0.0064 | 0.5648±0.0100 | 0.3898±0.0125 |
| GPT-3.5-turbo | rag | 0.5645±0.0026 | 0.1785±0.0009 | 0.5812±0.0143 | 0.4913±0.0040 |
| GPT-3.5-turbo | base | 0.3989±0.0109 | 0.2470±0.0026 | 0.5095±0.0193 | 0.3368±0.0053 |

### Cross-Domain Evaluation (African Union Test Set, n=255)

| Model | Mode | QWK (Mean±SD) | MAE (Mean±SD) | Spearman (Mean±SD) | ACC (Mean±SD) |
|-------|------|:-------------:|:-------------:|:------------------:|:-------------:|
| **GPT-5.2** | **full** | **0.7889±0.0186** | **0.0840±0.0059** | **0.7823±0.0174** | **0.7094±0.0191** |
| GPT-5.2 | cot | 0.7546±0.0144 | 0.0947±0.0032 | 0.7490±0.0167 | 0.6767±0.0077 |
| GPT-5.2 | rag | 0.6952±0.0066 | 0.1453±0.0047 | 0.7315±0.0039 | 0.5081±0.0153 |
| GPT-5.2 | base | 0.6420±0.0014 | 0.1666±0.0010 | 0.6902±0.0042 | 0.4667±0.0060 |
| **GPT-4o** | **full** | 0.7629±0.0174 | 0.0963±0.0084 | 0.7427±0.0194 | 0.6710±0.0307 |
| GPT-4o | cot | 0.7040±0.0196 | 0.1199±0.0015 | 0.6752±0.0128 | 0.5974±0.0283 |
| GPT-4o | rag | 0.6773±0.0112 | 0.1373±0.0026 | 0.6838±0.0132 | 0.5455±0.0064 |
| GPT-4o | base | 0.6555±0.0090 | 0.1416±0.0025 | 0.6522±0.0106 | 0.5368±0.0074 |
| GPT-4o-mini | full | 0.6211±0.0028 | 0.1578±0.0042 | 0.6364±0.0101 | 0.5050±0.0197 |
| GPT-4o-mini | cot | 0.6161±0.0139 | 0.1625±0.0054 | 0.6187±0.0166 | 0.4893±0.0137 |
| GPT-4o-mini | rag | 0.5335±0.0098 | 0.1747±0.0055 | 0.5780±0.0167 | 0.4732±0.0189 |
| GPT-4o-mini | base | 0.4322±0.0120 | 0.2060±0.0051 | 0.4876±0.0190 | 0.4135±0.0193 |
| GPT-3.5-turbo | full | 0.4651±0.0155 | 0.2307±0.0043 | 0.5237±0.0224 | 0.3464±0.0112 |
| GPT-3.5-turbo | cot | 0.4948±0.0152 | 0.2050±0.0046 | 0.5290±0.0187 | 0.3856±0.0092 |
| GPT-3.5-turbo | rag | 0.4452±0.0070 | 0.2330±0.0026 | 0.5298±0.0080 | 0.3582±0.0047 |
| GPT-3.5-turbo | base | 0.3491±0.0085 | 0.2686±0.0040 | 0.4683±0.0124 | 0.2776±0.0075 |

### Ablation Study Modes

| Mode | RAG | COT | Description |
|------|-----|-----|-------------|
| base | ❌ | ❌ | Zero-shot baseline |
| rag | ✅ | ❌ | RAG retrieval only |
| cot | ❌ | ✅ | Chain-of-thought only |
| full | ✅ | ✅ | Complete pipeline |

**Metrics**: QWK (Quadratic Weighted Kappa), MAE (Mean Absolute Error), Spearman correlation, ACC (Exact Agreement). Results reported as Mean ± Standard Deviation over multiple runs.


---
## 📜 License

This repository is licensed under the **MIT License** for the source code and  
the **CC BY-NC 4.0 License** for the dataset and embeddings.  
Please cite the related paper if you use any part of this repository.

© 2025-2026 Zihua Zeng
