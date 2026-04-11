# Multi-Modal RAG

A Retrieval-Augmented Generation (RAG) system supporting both text and image modalities.

## Project Structure

- `data/` - Datasets and input files
- `models/` - Pretrained and fine-tuned models
- `scripts/` - Scripts for training, inference, and utilities
- `docs/` - Documentation and design notes
- `requirements.txt` - Python dependencies

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Features

- Text and image retrieval
- Integration with vector databases (e.g., FAISS)
- Extensible for new modalities and models

## Getting Started

- Add your data to the `data/` folder.
- Place or download models in the `models/` folder.
- Use scripts in `scripts/` to run training or inference.

## License

MIT

# Multi-Modal RAG

A production-ready Retrieval-Augmented Generation (RAG) system that retrieves
across **text and image** corpora and generates grounded answers via an LLM.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Retrieval Modes](#retrieval-modes)
4. [Quick Start](#quick-start)
5. [Data Ingestion](#data-ingestion)
6. [Running the API](#running-the-api)
7. [CLI Tools](#cli-tools)
8. [Evaluation](#evaluation)
9. [Configuration](#configuration)
10. [Extending the System](#extending-the-system)

---

## Architecture

```
User query
    │
    ▼
┌───────────────────────────────────────────────────────┐
│                      Retriever                        │
│                                                       │
│  ┌─────────────────┐      ┌──────────────────────┐   │
│  │  SentenceTransf.│      │    CLIP text encoder  │   │
│  │  (384-d)        │      │    (512-d)            │   │
│  └────────┬────────┘      └───────────┬───────────┘   │
│           │                           │               │
│           ▼                           ▼               │
│  ┌─────────────────┐      ┌──────────────────────┐   │
│  │   Text FAISS    │      │   Image FAISS store  │   │
│  │   store         │      │   (CLIP 512-d)       │   │
│  └────────┬────────┘      └───────────┬───────────┘   │
│           │                           │               │
│           └────────── Fusion ─────────┘               │
│                    (RRF / weighted / max)              │
└───────────────────────────────────────────────────────┘
                           │
                           ▼
                    build_context()
                           │
                           ▼
               ┌───────────────────────┐
               │    LLM Generator      │
               │  (OpenAI / local)     │
               └───────────────────────┘
                           │
                           ▼
                         Answer
```

### Key components

| Module                           | Responsibility                                          |
| -------------------------------- | ------------------------------------------------------- |
| `core/chunking.py`               | Recursive text splitting with sliding-window overlap    |
| `core/embeddings.py`             | SentenceTransformer (text) + CLIP (image + cross-modal) |
| `core/vector_store.py`           | FAISS flat index with metadata & persistence            |
| `core/multimodal_fusion.py`      | Weighted-sum, Max-score, Reciprocal Rank Fusion         |
| `core/retriever.py`              | Orchestrates retrieval modes + context assembly         |
| `scripts/ingestion/ingest.py`    | Text / image / CSV ingestion pipeline                   |
| `scripts/generation/generate.py` | LLM wrapper (OpenAI or local HuggingFace)               |
| `scripts/evaluation/evaluate.py` | Recall@K, Precision@K, MRR evaluation                   |
| `main.py`                        | FastAPI HTTP interface                                  |

---

## Project Structure

```
multimodal_rag/
├── core/
│   ├── __init__.py
│   ├── chunking.py          # TextChunker
│   ├── embeddings.py        # EmbeddingModel (SentenceTransformer + CLIP)
│   ├── vector_store.py      # VectorStore (FAISS wrapper)
│   ├── multimodal_fusion.py # MultiModalFusion
│   ├── retriever.py         # Retriever
│   └── utils.py             # load_text_files, list_images
│
├── scripts/
│   ├── ingestion/
│   │   └── ingest.py        # DataIngestor + CLI
│   ├── generation/
│   │   └── generate.py      # Generator, build_prompt + CLI
│   ├── evaluation/
│   │   └── evaluate.py      # Metrics + CLI
│   └── query.py             # Raw retrieval CLI (no generation)
│
├── data/
│   ├── text/                # Put .txt files here
│   └── images/              # Put .png/.jpg files here
│
├── stores/
│   ├── text_store/          # Auto-created by ingest.py
│   └── image_store/         # Auto-created by ingest.py
│
├── tests/
├── main.py                  # FastAPI app
├── debug.py                 # Store diagnostics
├── requirements.txt
└── README.md
```

---

## Retrieval Modes

| Mode            | Query type  | Store searched     | Embedding                 |
| --------------- | ----------- | ------------------ | ------------------------- |
| `text`          | Text string | Text FAISS         | SentenceTransformer 384-d |
| `image`         | Image file  | Image FAISS        | CLIP image 512-d          |
| `text_to_image` | Text string | Image FAISS        | CLIP text 512-d           |
| `hybrid`        | Text string | Text + Image FAISS | Both, then fused          |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Add data

```
data/text/   ← drop .txt files here
data/images/ ← drop .png / .jpg files here
```

### 4. Ingest

```bash
python -m scripts.ingestion.ingest \
    --text_folder data/text \
    --image_folder data/images
```

### 5. Query

```bash
# CLI – text retrieval + generation
python -m scripts.generation.generate --query "What is CLIP?"

# CLI – hybrid retrieval
python -m scripts.generation.generate --query "cat on a sofa" --mode hybrid

# Raw retrieval (no LLM)
python -m scripts.query --query "neural scaling laws" --top_k 5
```

---

## Data Ingestion

### Text files

```bash
python -m scripts.ingestion.ingest --text_folder data/text
```

Each `.txt` file is:

1. Split into 500-character chunks with 100-character overlap.
2. Batch-encoded with SentenceTransformer.
3. Stored in the text FAISS store with `{"source": filename, "text": chunk}` metadata.

### Images

```bash
python -m scripts.ingestion.ingest --image_folder data/images
```

Each image is encoded with CLIP and stored in the image FAISS store.

### CSV corpus

```bash
python -m scripts.ingestion.ingest --csv_file data/documents.csv
```

The CSV must have at minimum a `text` column and optionally a `source_url` column.

### Combining all sources

```bash
python -m scripts.ingestion.ingest \
    --text_folder data/text \
    --image_folder data/images \
    --csv_file data/documents.csv
```

---

## Running the API

```bash
uvicorn main:app --reload --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

### Endpoints

#### `GET /`

Health check. Returns vector counts.

#### `POST /query`

```json
{
  "query": "What is neural scaling?",
  "mode": "hybrid",
  "top_k": 5,
  "alpha": 0.6
}
```

Response:

```json
{
  "query": "...",
  "mode": "hybrid",
  "answer": "Neural scaling refers to ...",
  "sources": [...],
  "context_preview": "..."
}
```

#### `POST /image-query`

Upload a PNG/JPEG image; returns visually similar results + generated answer.

```bash
curl -X POST http://localhost:8000/image-query \
     -F "file=@my_photo.jpg"
```

---

## CLI Tools

### Raw retrieval (no LLM)

```bash
python -m scripts.query --query "deep learning" --mode text --top_k 10
python -m scripts.query --image path/to/img.jpg --mode image
python -m scripts.query --query "red car" --mode hybrid --json  # JSON output
```

### Full RAG pipeline

```bash
python -m scripts.generation.generate \
    --query "Explain attention mechanisms" \
    --mode hybrid \
    --model openai \
    --top_k 5
```

Use `--model local` to run fully offline with GPT-2 (or any HF model).

### Store diagnostics

```bash
python debug.py
```

---

## Evaluation

### Built-in metrics

| Metric        | Description                                   |
| ------------- | --------------------------------------------- |
| `Recall@K`    | Fraction of relevant docs found in top-K      |
| `Precision@K` | Fraction of top-K results that are relevant   |
| `MRR`         | Mean Reciprocal Rank of first relevant result |

### Running evaluation

```bash
python -m scripts.evaluation.evaluate --mode text --k 5
```

### Custom evaluation data

Create a JSON file:

```json
[
  { "query": "cat", "relevant_ids": ["cats_001", "cats_002"] },
  { "query": "car engine", "relevant_ids": ["auto_003"] }
]
```

Then:

```bash
python -m scripts.evaluation.evaluate \
    --eval_json eval_data.json \
    --mode hybrid \
    --k 10
```

---

## Configuration

| Parameter           | Location                     | Default             | Description                    |
| ------------------- | ---------------------------- | ------------------- | ------------------------------ |
| `chunk_size`        | `DataIngestor.__init__`      | 500                 | Max chars per chunk            |
| `chunk_overlap`     | `DataIngestor.__init__`      | 100                 | Overlap between chunks         |
| `fusion_strategy`   | `Retriever.__init__`         | `"reciprocal_rank"` | Fusion method                  |
| `alpha`             | `retrieve_hybrid` / `/query` | 0.5                 | Text weight in weighted fusion |
| `max_chars`         | `Retriever.build_context`    | 4000                | Context budget (chars)         |
| `GENERATOR_BACKEND` | env var                      | `"openai"`          | `"openai"` or `"local"`        |
| `OPENAI_API_KEY`    | env var                      | –                   | Required for OpenAI backend    |

---

## Extending the System

### Swap the text encoder

Change `text_model_name` in `EmbeddingModel.__init__` and update the
`VectorStore` dimension accordingly (e.g. `dim=768` for `all-mpnet-base-v2`).

### Swap to approximate search (large corpora)

Replace `faiss.IndexFlatIP` in `VectorStore.__init__` with:

```python
quantiser = faiss.IndexFlatIP(dim)
self.index = faiss.IndexIVFFlat(quantiser, dim, 100)   # 100 clusters
self.index.train(training_vectors)
```

### Add BM25 sparse retrieval

Integrate `rank_bm25` and fuse its results via `MultiModalFusion` alongside
the dense retrieval results.

### Use a different LLM

Add a new branch in `Generator.__init__` / `Generator.generate` (e.g. Anthropic,
Mistral, Ollama) following the same interface.
