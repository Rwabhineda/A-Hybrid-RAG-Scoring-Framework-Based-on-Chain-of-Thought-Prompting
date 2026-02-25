# 实验配置说明

每个模型目录下包含 5 档实验配置；使用 RAG 时默认启用 E5 相关度过滤。

## 五档实验

| 配置文件名 | 说明 | RAG | COT | E5 过滤 |
|------------|------|-----|-----|--------|
| **base-only** | 1. 仅基础提示 | 否 | 否 | - |
| **rag-only** | 2. RAG + 基础提示 | 是 | 否 | 是（默认） |
| **cot-only** | 3. COT + 基础提示 | 否 | 是 | - |
| **gpt-4o.yaml**（等） | 4. RAG + COT + 基础提示（完整） | 是 | 是 | 是（默认） |
| **full-no-e5** | 5. 完整但不做 E5 过滤（对比） | 是 | 是 | 否 |

- 凡使用 RAG 的配置（2、4）默认开启 E5 过滤（`wrd_enabled: true`）。
- **full-no-e5** 用于对比：与 4 相同但关闭 E5，便于评估 E5 的贡献。

## 运行示例

```bash
# 1. 仅基础提示
uv run python src/main.py --config configs/gpt-4o/gpt-4o-base-only.yaml

# 2. RAG + 基础提示（含 E5）
uv run python src/main.py --config configs/gpt-4o/gpt-4o-rag-only.yaml

# 3. COT + 基础提示
uv run python src/main.py --config configs/gpt-4o/gpt-4o-cot-only.yaml

# 4. 完整（含 E5）
uv run python src/main.py --config configs/gpt-4o/gpt-4o.yaml

# 5. 完整但不做 E5 过滤
uv run python src/main.py --config configs/gpt-4o/gpt-4o-full-no-e5.yaml
```

输出路径与实验名对应（如 `outputs/gpt-4o-base-only/results.jsonl`）。**-wrd** 配置与对应 **gpt-4o.yaml** 等等价，保留仅为兼容。
