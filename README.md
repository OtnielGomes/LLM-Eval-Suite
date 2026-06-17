# llm-eval

A framework for evaluating Large Language Models using **prompting strategies** (Zero-Shot, Few-Shot, Chain-of-Thought) and **RAG retrieval strategies** (Naive, HyDE, Reranking), with metrics from BLEU, ROUGE, LLM-as-Judge, and RAGAS.

***

## Project Structure

```
llm-eval/
├── notebooks/
│   ├── 01_benchmark_analysis.ipynb   # Benchmark results analysis
│   └── 02_rag_comparison.ipynb       # RAG pipeline results analysis
├── scripts/
│   ├── run_benchmark.py              # CLI: benchmark runner
│   ├── evaluate_rag.py               # CLI: RAG evaluator
│   ├── ingest.py                     # CLI: document ingestion into ChromaDB
│   ├── populate_raw_docs.py          # CLI: populate raw document corpus
│   └── build_datasets.py             # CLI: build JSONL datasets
├── src/llm_eval/
│   ├── benchmark/
│   │   ├── evaluator.py              # BLEU, ROUGE, LLM-as-Judge logic
│   │   └── prompt_strategies/        # zero_shot, few_shot, chain_of_thought
│   ├── rag/
│   │   ├── rag_pipeline.py           # RAG orchestrator
│   │   └── strategies/               # naive, hyde, reranking
│   ├── clients/
│   │   ├── ollama_cliente.py         # Ollama local client
│   │   └── ollama_cloud_cliente.py   # Ollama Cloud client
│   ├── datasets/
│   │   ├── mmlu_sample.jsonl         # MMLU benchmark dataset
│   │   ├── curated_qa.jsonl          # Curated Q&A for RAG evaluation
│   │   └── raw/                      # Source documents (26 MMLU domains)
│   └── shared/
│       ├── config.py                 # Environment config
│       ├── langsmith_tracer.py       # LangSmith tracing
│       └── types.py                  # EvalResult, RAGResult types
├── results/                          # JSON + CSV outputs per model/strategy
├── chromadb/                         # Persisted ChromaDB vector store
├── tests/                            # Unit tests
└── .env                              # API keys and config (see below)
```

***

## Architecture

The project uses a **hybrid inference architecture**:

| Component | Backend | Model | Purpose |
|---|---|---|---|
| Generation | Ollama Cloud | Configurable via `.env` | Answer generation |
| Embeddings | Ollama Local | `nomic-embed-text` | Document & query embeddings |
| Judge (benchmark) | Ollama Cloud | Configurable via `.env` | LLM-as-Judge scoring |
| Judge (RAGAS) | Ollama Cloud | `qwen3-coder:480b` (default) | RAGAS metric evaluation |
| Vector Store | Local | ChromaDB | Document retrieval |

***

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [Ollama](https://ollama.com/) running locally on `http://localhost:11434`
- Ollama Cloud API key ([get one here](https://ollama.com/settings/keys))

***

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

Create a `.env` file at the project root:

```env
# Ollama Cloud
OLLAMA_API_KEY=your-ollama-cloud-key
OLLAMA_CLOUD_BASE_URL=https://api.ollama.com
OLLAMA_CLOUD_MODEL=qwen3-coder:480b-cloud

# Ollama Local
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.1:8b

# Judge model for RAGAS (use a large model for stable parsing)
OLLAMA_JUDGE_MODEL=qwen3-coder:480b-cloud

# LangSmith (optional — for tracing)
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=llm-eval
```

### 3. Pull the embedding model

```bash
ollama pull nomic-embed-text
```

***

## Usage

### Benchmark (Notebook 01)

**Run all prompting strategies:**

```bash
uv run run-benchmark --all-strategies --limit 50
```

**Run a specific strategy:**

```bash
uv run run-benchmark --strategy zero_shot --limit 25
uv run run-benchmark --strategy few_shot --limit 25
uv run run-benchmark --strategy chain_of_thought --limit 25
```

**Run locally (Ollama local model):**

```bash
uv run run-benchmark --strategy zero_shot --local --limit 10
```

Results are saved to `results/` as `.json` and `.csv`.

***

### RAG Evaluation (Notebook 02)

**Step 1 — Ingest documents into ChromaDB:**

```bash
uv run ingest-docs
# Reset and re-ingest:
uv run ingest-docs --reset
```

**Step 2 — Run RAG evaluation:**

```bash
# All strategies, all models
uv run evaluate-rag

# Specific strategy and model
uv run evaluate-rag --strategy naive --model qwen3-coder:480b-cloud --limit 10
uv run evaluate-rag --strategy hyde --limit 5
uv run evaluate-rag --strategy reranking --limit 5
```

Results are saved to `results/ragas_<strategy>_<model>.json`.

***

## Prompting Strategies

| Strategy | Description | Best for |
|---|---|---|
| **Zero-Shot** | Direct question, no examples | Large models with strong priors |
| **Few-Shot** | 3–5 solved examples prepended to the prompt | Format calibration |
| **Chain-of-Thought** | Instructs model to reason step-by-step | Smaller models on complex tasks |

***

## RAG Retrieval Strategies

| Strategy | Description | Extra cost |
|---|---|---|
| **Naive** | Direct query embedding → vector search → LLM | None |
| **HyDE** | LLM generates a hypothetical answer → embed hypothesis → vector search → LLM | +1 LLM call per query |
| **Reranking** | Broad retrieval (top-K) → cross-encoder reranks → top-N → LLM | +K cross-encoder calls |

***

## Evaluation Metrics

### Benchmark Metrics

| Metric | What it measures | Primary? |
|---|---|---|
| `judge_correct` | Whether the correct answer letter was selected (binary) | ✅ Yes |
| `judge_score` | Fluency and completeness of reasoning (1–5, LLM-as-Judge) | ⚠️ Secondary |
| `ROUGE-L` | Longest common subsequence overlap with reference | ❌ Unreliable for MCQ+CoT |
| `BLEU` | N-gram overlap with reference | ❌ Unreliable for MCQ+CoT |
| `latency_ms` | Response time per query in milliseconds | ✅ Operational |

### RAGAS Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Are claims in the answer supported by retrieved documents? |
| **Answer Relevancy** | Does the answer address the user's question? |
| **Context Recall** | Were all relevant chunks retrieved? |
| **Context Precision** | Are retrieved chunks genuinely relevant? |
| **Composite Score** | Mean of all four RAGAS metrics |

***

## Key Results

### Benchmark (MMLU — 50 questions)

| Model | Zero-Shot | Few-Shot | CoT | Mean Accuracy |
|---|---|---|---|---|
| **Qwen3-Coder 480B** | 90% | 88% | 90% | **89.3%** |
| Gemma3 27B | 68% | 64% | 88% | 73.3% |
| GPT-OSS 20B | 0% | 4% | 8% | 4% |

### RAG Evaluation — Composite RAGAS Score

| Model | Naive | HyDE | Reranking | Mean |
|---|---|---|---|---|
| **Gemma3 27B** | 0.968 | **0.988** | 0.969 | **0.975** |
| Qwen3-Coder 480B | 0.971 | 0.973 | 0.922 | 0.955 |
| GPT-OSS 20B | 0.651 | 0.000 | 0.653 | 0.435 |

***

## Production Recommendations

| Use Case | Model | Strategy | Reason |
|---|---|---|---|
| MCQ / Reasoning tasks | Qwen3-Coder 480B | Zero-Shot | 90% accuracy at ~4.1s, no prompt engineering needed |
| RAG — max faithfulness | Gemma3 27B | HyDE | Composite 0.988, faithfulness 1.00 |
| RAG — low latency | Qwen3-Coder 480B | Naive | Composite 0.971, simplest pipeline |

***

## Known Limitations

- **ROUGE-L / BLEU are unreliable for MCQ** when using CoT — long reasoning chains deflate scores against single-letter references. Use `judge_correct` as the primary accuracy metric.
- **Judge Score measures fluency, not correctness** — a model can receive a high judge score with an incorrect answer. Always cross-reference with `judge_correct`.
- **HyDE collapses with weak models** — GPT-OSS 20B produced a composite score of 0.000 with HyDE because the model cannot generate coherent hypothetical documents. Add a fallback to Naive when the hypothesis is empty or malformed.
- **Reranking degrades Qwen3 context recall** — bimodal distribution observed in the violin plot suggests the cross-encoder discards relevant chunks for technical queries.

***

## Observability

LangSmith tracing is supported out of the box. When `LANGCHAIN_TRACING_V2=true` is set in `.env`, every LLM call in the benchmark is traced with:

- Strategy name and model
- Prompt and response
- `judge_score`, `judge_correct`, BLEU, ROUGE-L, `latency_ms`
- Run-level summary per strategy

View traces at [smith.langchain.com](https://smith.langchain.com).

***

## Tests

```bash
uv run pytest tests/ -v
```

Test coverage includes the `Evaluator`, Ollama Cloud client, and RAG pipeline.

***

## License

MIT