# A Hybrid RAG Scoring Framework Based on Chain-of-Thought Prompting

This repository contains code and data for a project that explores **Retrieval-Augmented Generation (RAG)** combined with **Chain-of-Thought (CoT) prompting** to evaluate the legalization degree of international legal instruments.

---

## 📂 Project Overview
- **Expert-labeled data**: JSON files with clause-level scores (obligation, precision, delegation).  
- **RAG pipeline**: Uses ChromaDB vector retrieval + Cross-Encoder filtering.  
- **LLM scoring**: Large language models with CoT prompting generate clause scores.  
- **Evaluation**: ICC, correlation, MAE, F1 and other metrics are used to compare with expert coders.

---

## ⚙️ Quick Start
Clone the repository:

```bash
git clone https://github.com/Rwabhineda/A-Hybrid-RAG-Scoring-Framework-Based-on-Chain-of-Thought-Prompting.git
cd A-Hybrid-RAG-Scoring-Framework-Based-on-Chain-of-Thought-Prompting
