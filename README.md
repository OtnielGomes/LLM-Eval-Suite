# LLM Eval Suite

Pipeline completo de avaliação de Large Language Models (LLMs) com suporte a benchmarks públicos, avaliação de pipelines RAG e métricas automáticas (BLEU, ROUGE, LLM-as-judge).

***

## Visão Geral

O **LLM Eval Suite** é um framework modular para avaliar modelos de linguagem em múltiplas dimensões:

- **Benchmarks públicos** — MMLU (raciocínio e conhecimento) e HumanEval (geração de código)
- **Avaliação RAG** — três estratégias de retrieval avaliadas com o framework RAGAS
- **Métricas automáticas** — BLEU, ROUGE-L e LLM-as-judge via Gemini
- **Observabilidade** — rastreamento completo de experimentos via LangSmith

***

## Estrutura do Projeto

```
llm-eval-suite/
│
├── 01_benchmark/           # Benchmark com MMLU e HumanEval
│   ├── __init__.py
│   ├── run_benchmark.py    # Execução dos benchmarks
│   └── metrics.py          # Cálculo de BLEU, ROUGE, LLM-as-judge
│
├── 02_rag_eval/            # Avaliação de pipelines RAG
│   ├── __init__.py
│   ├── ingest_docs.py      # Chunking + embeddings → ChromaDB
│   ├── rag_pipeline.py     # 3 estratégias de retrieval
│   └── evaluate_rag.py     # Avaliação com RAGAS
│
├── 03_dataset/             # Curadoria e build dos datasets
│   ├── __init__.py
│   └── build_datasets.py   # Download e processamento dos datasets
│
├── shared/                 # Utilitários compartilhados
│   ├── __init__.py
│   ├── llm_client.py       # Cliente unificado (Gemini / Ollama)
│   └── utils.py            # Helpers gerais
│
├── scripts/                # Entry points dos scripts CLI
│   ├── __init__.py
│   ├── build_datasets.py
│   ├── run_benchmark.py
│   ├── ingest_docs.py
│   └── evaluate_rag.py
│
├── tests/                  # Testes automatizados
│   └── ...
│
├── notebooks/              # Análise exploratória e visualizações
│   └── ...
│
├── .env                    # Variáveis de ambiente (não versionar)
├── .env.example            # Template de variáveis de ambiente
├── pyrightconfig.json      # Configuração do Pylance/Pyright
├── pyproject.toml          # Dependências e configuração do projeto
└── README.md
```

***

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- Chave de API do Google Gemini
- Chave de API do LangSmith (opcional, para observabilidade)
- Ollama (opcional, para modelos locais)

***

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/seu-usuario/llm-eval-suite.git
cd llm-eval-suite

# 2. Instalar uv (se não tiver)
pip install uv

# 3. Criar ambiente virtual e instalar dependências
uv sync

# 4. Instalar dependências de desenvolvimento (pytest, ruff, notebooks)
uv sync --group dev

# 5. Configurar variáveis de ambiente
cp .env.example .env
# Edite o .env com suas chaves de API
```

***

## Configuração

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Google Gemini
GEMINI_API_KEY='sua-chave-aqui'

# LangSmith (observabilidade — opcional)
LANGSMITH_API_KEY='lsv2_pt_sua-chave-aqui'
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=llm-eval-suite

# Ollama (modelos locais — opcional)
OLLAMA_BASE_URL=http://localhost:11434
```

***

## Uso

Todos os comandos são executados via `uv run`:

```bash
# 1. Baixar e processar os datasets (MMLU + HumanEval)
uv run build-datasets

# 2. Executar o benchmark completo
uv run run-benchmark

# 3. Ingerir documentos no ChromaDB (vetorização)
uv run ingest-docs

# 4. Avaliar as 3 estratégias RAG com RAGAS
uv run evaluate-rag
```

Ou, após `pip install -e .`, os comandos ficam disponíveis diretamente no terminal:

```bash
build-datasets
run-benchmark
ingest-docs
evaluate-rag
```

***

## Benchmarks Implementados

### MMLU (Massive Multitask Language Understanding)

Avalia o modelo em 57 domínios de conhecimento (matemática, direito, medicina, história, etc.) com questões de múltipla escolha de 4 alternativas.

| Métrica | Descrição |
|---------|-----------|
| Accuracy | % de questões respondidas corretamente |
| Accuracy por domínio | Desempenho segmentado por área |

### HumanEval

Avalia a capacidade do modelo de gerar código Python funcional a partir de docstrings.

| Métrica | Descrição |
|---------|-----------|
| pass@1 | % de problemas resolvidos na primeira tentativa |
| BLEU | Similaridade léxica com solução de referência |

***

## Avaliação RAG

Três estratégias de retrieval são comparadas:

| Estratégia | Descrição |
|------------|-----------|
| **Naive RAG** | Chunking fixo + similaridade cosine |
| **HyDE** | Geração de documento hipotético antes do retrieval |
| **Reranking** | Retrieval inicial + reranking por relevância |

### Métricas RAGAS

| Métrica | O que mede |
|---------|------------|
| `faithfulness` | Respostas fundamentadas nos documentos recuperados |
| `answer_relevancy` | Relevância da resposta para a pergunta |
| `context_precision` | Precisão dos chunks recuperados |
| `context_recall` | Cobertura dos chunks relevantes |

***

## Métricas Automáticas

Além do RAGAS, o projeto implementa:

- **BLEU** — similaridade n-gram entre resposta gerada e referência
- **ROUGE-L** — sequência comum mais longa entre gerado e referência
- **LLM-as-judge** — avaliação de qualidade usando Gemini como juiz (escala 1–5)

***

## Observabilidade com LangSmith

Com `LANGSMITH_TRACING=true`, todas as chamadas LLM são rastreadas automaticamente no painel do LangSmith, incluindo:

- Prompts enviados e respostas recebidas
- Latência e contagem de tokens por chamada
- Agrupamento de traces por experimento via `LANGSMITH_PROJECT`

***

## Desenvolvimento

```bash
# Rodar os testes
uv run pytest

# Rodar com cobertura
uv run pytest --cov

# Linting e formatação
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy .
```

***

## Dependências Principais

| Pacote | Uso |
|--------|-----|
| `google-genai` | Cliente oficial da API Gemini |
| `langchain` + `langchain-google-genai` | Orquestração de pipelines LLM |
| `chromadb` + `langchain-chroma` | Vector store para RAG |
| `ragas` | Avaliação automática de pipelines RAG |
| `datasets` | Acesso aos datasets MMLU e HumanEval (HuggingFace) |
| `nltk` + `rouge-score` | Métricas BLEU e ROUGE |
| `langsmith` | Rastreamento e observabilidade |
| `typer` + `rich` | Interface CLI com output formatado |

***

## Licença

MIT License — veja o arquivo [LICENSE](LICENSE) para detalhes.
