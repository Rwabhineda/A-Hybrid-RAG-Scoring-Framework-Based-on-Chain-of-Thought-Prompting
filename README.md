# A Hybrid RAG Scoring Framework Based on Chain-of-Thought Prompting

[![CI](https://github.com/Rwabhineda/A-Hybrid-RAG-Scoring-Framework-Based-on-Chain-of-Thought-Prompting)
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
**Data**: 2,611 expert-annotated ASEAN clauses (+ 254-clause independent test set)  
**Models**: GPT-3.5, GPT-4o, GPT-4o-mini, GPT-5  
**Best**: GPT-4o → ICC **0.8163**, MAE **0.0797**, Exact Agreement **77.0%**, F1@0.75 **0.7679**

---

## ✨ Highlights

- **Hybrid RAG**: dense retrieval (ChromaDB, cosine) + **quality-weighted** reranking by confidence
- **CE Filtering**: Legal-BERT Cross-Encoder binary relevance gate (plug-in)
- **CoT Prompting**: stepwise rubric + few-shot exemplars (Top-K) + robust output parsing
- **Evaluation**: ICC(2,1), MAE, Exact Agreement, Recall/Precision/F1@0.75
- **Reproducible**: config-driven CLI, fixed seeds, tiny demo for quick verification

---

## 🧭 Repository Structure



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

## 📊 Results (key findings)
| Model       | Setting    |   ICC(2,1) |        MAE |     Exact |   F1\@0.75 |
| ----------- | ---------- | ---------: | ---------: | --------: | ---------: |
| GPT-4o      | RAG+Prompt | **0.8163** | **0.0797** | **77.0%** | **0.7679** |
| GPT-4o-mini | RAG+Prompt |     0.7090 |     0.1270 |     69.2% |     0.6874 |
| GPT-3.5     | RAG+Prompt |     0.5227 |     0.2142 |     45.3% |     0.6079 |
| GPT-5       | RAG+Prompt |     0.7158 |     0.1545 |     47.1% |     0.5078 |

Exact definitions of metrics and experimental protocols are aligned with the paper (ICC, MAE, Exact, Recall/Precision/F1@0.75).

