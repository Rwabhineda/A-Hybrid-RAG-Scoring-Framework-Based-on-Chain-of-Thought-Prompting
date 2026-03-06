# Configuration Guide

This directory contains experiment configurations for the Legal Clause Scoring system. Configurations are organized by experiment domain (`asean/` for in-domain, `transfer/` for cross-domain evaluation).

## Directory Structure

```
configs/
├── asean/                    # In-domain experiments (ASEAN legal documents)
│   ├── gpt-3.5-turbo/
│   ├── gpt-4o/
│   ├── gpt-4o-mini/
│   └── gpt-5.2/
│
└── transfer/                 # Cross-domain experiments (African Union documents)
    ├── gpt-3.5-turbo/
    ├── gpt-4o/
    ├── gpt-4o-mini/
    └── gpt-5.2/
```

## Experiment Types

| Directory | Domain | Dataset | Purpose |
|-----------|--------|---------|---------|
| `asean/` | In-domain | ASEAN legal clauses (254 samples) | Primary evaluation with RAG trained on ASEAN |
| `transfer/` | Cross-domain | African Union documents (255 samples) | Transfer learning evaluation |

## Ablation Study Configurations

Each model directory contains 4 configuration variants:

| Config File | Description | RAG | COT | Output Path |
|-------------|-------------|-----|-----|-------------|
| `*-base.yaml` | Base prompt only | No | No | `outputs/{domain}/{model}/base/` |
| `*-rag.yaml` | RAG + base prompt | Yes | No | `outputs/{domain}/{model}/rag/` |
| `*-cot.yaml` | COT + base prompt | No | Yes | `outputs/{domain}/{model}/cot/` |
| `*.yaml` (full) | Full (RAG + COT) | Yes | Yes | `outputs/{domain}/{model}/full/` |

## Supported Models

| Model Directory | Model Name | Provider |
|-----------------|------------|----------|
| `gpt-3.5-turbo/` | GPT-3.5 Turbo | OpenAI |
| `gpt-4o/` | GPT-4o | OpenAI |
| `gpt-4o-mini/` | GPT-4o Mini | OpenAI |
| `gpt-5.2/` | GPT-5.2 | OpenAI |

## Usage Examples

### In-Domain (ASEAN) Experiments

```bash
# Full configuration (RAG + COT)
uv run python src/main.py --config configs/asean/gpt-5.2/gpt-5.2.yaml

# Ablation: Base only
uv run python src/main.py --config configs/asean/gpt-5.2/gpt-5.2-base.yaml

# Ablation: RAG only
uv run python src/main.py --config configs/asean/gpt-5.2/gpt-5.2-rag.yaml

# Ablation: COT only
uv run python src/main.py --config configs/asean/gpt-5.2/gpt-5.2-cot.yaml
```

### Cross-Domain (Transfer) Experiments

```bash
# Full configuration on African Union data
uv run python src/main.py --config configs/transfer/gpt-5.2/gpt-5.2.yaml

# Ablation studies
uv run python src/main.py --config configs/transfer/gpt-4o/gpt-4o-base.yaml
uv run python src/main.py --config configs/transfer/gpt-4o/gpt-4o-rag.yaml
```

## Configuration Structure

```yaml
experiment:
  name: "gpt-5.2-full"
  description: "Full: RAG + COT + base prompt"

paths:
  input_file: "data/processed/asean/test_articles.json"      # or transfer/
  output_file: "outputs/asean/gpt-5.2/full/results.jsonl"
  cache_dir: "data/cache/asean/gpt-5.2/full"

vector_db:
  chroma_dir: "data/rag/chroma_db"
  collection_name: "asean_scoring"

models:
  openai:
    model: "gpt-5.2"
    api_url: "https://api.openai.com/v1/responses"
    temperature: 0
  embedding:
    model: "intfloat/e5-large-v2"
  filter:
    model: "nlpaueb/legal-bert-base-uncased"

features:
  wrd_enabled: true      # Legal-BERT relevance filtering
  use_rag: true          # RAG retrieval
  use_cot_guide: true    # Chain-of-thought scoring guide
```

## Output Format

Results are saved in JSONL format:

| Field | Description |
|-------|-------------|
| `id` | Clause identifier |
| `document_title` | Source document name |
| `text` | Clause text content |
| `obligation` | Obligation score (0.0, 0.25, 0.5, 0.75, 1.0) |
| `precision` | Precision score (0.0, 0.25, 0.5, 0.75, 1.0) |
| `delegation` | Delegation score (0.0, 0.25, 0.5, 0.75, 1.0) |

## Evaluation

```bash
# Evaluate ASEAN results
uv run python src/evaluation/eval.py --pred outputs/asean/gpt-5.2/full/results.jsonl

# Evaluate Transfer results (requires transfer gold standard)
uv run python src/evaluation/eval.py --pred outputs/transfer/gpt-5.2/full/results.jsonl
```

Metrics: ICC(2,1), MAE, Exact Agreement, Precision@0.75, Recall@0.75, F1@0.75

## Environment Setup

Create a `.env` file in project root:

```bash
OPENAI_API_KEY=your_openai_key_here
```
