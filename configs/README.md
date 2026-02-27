# Configuration Guide

This directory contains experiment configurations for the ASEAN Legal Clause Scoring system. Each model subdirectory includes 5 configuration variants for ablation studies. E5 relevance filtering is enabled by default when RAG is used.

## Ablation Study Configurations

| Config File | Description | RAG | COT | E5 Filter |
|-------------|-------------|-----|-----|-----------|
| **base-only** | 1. Base prompt only | No | No | - |
| **rag-only** | 2. RAG + base prompt | Yes | No | Yes (default) |
| **cot-only** | 3. COT + base prompt | No | Yes | - |
| **gpt-4o.yaml** (etc.) | 4. Full: RAG + COT + base prompt | Yes | Yes | Yes (default) |
| **full-no-e5** | 5. Full without E5 filter (ablation) | Yes | Yes | No |

- Configurations using RAG (2, 4) have E5 filtering enabled by default (`wrd_enabled: true`).
- **full-no-e5** is used for comparison: same as config 4 but disables E5 to evaluate its contribution.

## Supported Models

| Model Directory | Model Name | Provider |
|-----------------|------------|----------|
| `gpt-3.5-turbo/` | GPT-3.5 Turbo | OpenAI |
| `gpt-4-turbo/` | GPT-4 Turbo | OpenAI |
| `gpt-4o/` | GPT-4o | OpenAI |
| `gpt-4o-mini/` | GPT-4o Mini | OpenAI |
| `gpt-5/` | GPT-5 | OpenAI |

## Usage Examples

```bash
# 1. Base prompt only (zero-shot baseline)
uv run python src/main.py --config configs/gpt-4o/gpt-4o-base.yaml

# 2. RAG + base prompt (with E5 filtering)
uv run python src/main.py --config configs/gpt-4o/gpt-4o-rag.yaml

# 3. COT + base prompt (chain-of-thought reasoning)
uv run python src/main.py --config configs/gpt-4o/gpt-4o-cot.yaml

# 4. Full configuration (RAG + COT + E5)
uv run python src/main.py --config configs/gpt-4o/gpt-4o.yaml

# 5. Full without E5 filter (ablation study)
uv run python src/main.py --config configs/gpt-5/gpt-5-full-no-e5.yaml
```

## Configuration Structure

Each YAML file follows this structure:

```yaml
experiment:
  name: "experiment-name"          # Used for output directory naming
  description: "Brief description"

paths:
  input_file: "data/processed/test_articles.json"
  output_file: "outputs/{experiment-name}/results.jsonl"
  exception_log: "logs/{experiment-name}_exceptions.log"
  cache_dir: "data/cache/{model-name}"

models:
  openai:
    model: "gpt-4o"                 # Model identifier
    api_url: "https://api.openai.com/v1/chat/completions"
    api_key: ""                     # Loaded from .env file
    temperature: 0.0                # Sampling temperature (0.0 for deterministic)
  embedding:
    model: "sentence-transformers/all-mpnet-base-v2"
  filter:
    model: "intfloat/e5-large-v2"   # E5 model for relevance filtering

features:
  use_rag: true                     # Enable RAG retrieval
  use_cot_guide: true               # Include CoT scoring guide in prompt
  wrd_enabled: true                 # Enable E5 relevance filtering
```

## Output Files

Output paths correspond to experiment names (e.g., `outputs/gpt-4o-base/results.jsonl`). Results are saved in JSONL format with the following fields:

- `id`: Clause identifier
- `document_title`: Source document name
- `text`: Clause text content
- `obligation`: Obligation score (0.0, 0.25, 0.5, 0.75, 1.0)
- `precision`: Precision score (0.0, 0.25, 0.5, 0.75, 1.0)
- `delegation`: Delegation score (0.0, 0.25, 0.5, 0.75, 1.0)

## Environment Setup

Create a `.env` file in the project root:

```bash
API_PROVIDER=openai                    # or 'deepseek' for DeepSeek API
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here  # Required if using DeepSeek
```

## Evaluation

After scoring, evaluate results against the gold standard:

```bash
uv run python src/evaluation/eval.py --pred outputs/gpt-4o/results.jsonl
```

Metrics computed: ICC(2,1), MAE, Exact Agreement Rate, Precision@0.75, Recall@0.75, F1-Score@0.75
