# Multi-Modal RAG

A production-ready **Retrieval-Augmented Generation** system that retrieves across
**text and image** corpora and generates grounded answers using **free local
HuggingFace models** — no API key or paid service required.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Data Ingestion](#data-ingestion)
7. [Free Model Backends](#free-model-backends)
8. [Retrieval Modes](#retrieval-modes)
9. [Running the API](#running-the-api)
10. [CLI Reference](#cli-reference)
11. [Evaluation](#evaluation)
12. [Configuration Reference](#configuration-reference)
13. [GPU vs CPU](#gpu-vs-cpu)
14. [Extending the System](#extending-the-system)

---

## How It Works

```
                        User Query
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │                  Retriever                │
        │                                           │
        │  SentenceTransformer       CLIP text enc  │
        │      (384-d)                  (512-d)     │
        │         │                       │         │
        │         ▼                       ▼         │
        │   Text FAISS store      Image FAISS store │
        │         │                       │         │
        │         └──── Fusion (RRF) ─────┘         │
        └───────────────────────────────────────────┘
                            │
                     build_context()
                            │
                            ▼
              ┌─────────────────────────┐
              │  Free HuggingFace LLM   │
              │  flan-t5 / phi2 / gpt2  │
              └─────────────────────────┘
                            │
                            ▼
                          Answer
```

### Pipeline steps

1. **Ingest** — text files and images are encoded into dense vectors and saved to two FAISS indexes (text store: 384-d, image store: 512-d).
2. **Retrieve** — at query time, the query is embedded and the nearest vectors are fetched. In hybrid mode, text and image results are fused with Reciprocal Rank Fusion (RRF).
3. **Generate** — retrieved chunks are assembled into a context string and passed to a free local LLM which produces the final answer.

---

## Project Structure

```
multimodal_rag/
├── core/
│   ├── chunking.py           # TextChunker — recursive splitting + overlap
│   ├── embeddings.py         # EmbeddingModel — SentenceTransformer + CLIP
│   ├── vector_store.py       # VectorStore — FAISS wrapper with persistence
│   ├── multimodal_fusion.py  # MultiModalFusion — RRF / weighted / max-score
│   ├── retriever.py          # Retriever — orchestrates all retrieval modes
│   └── utils.py              # load_text_files, list_images helpers
│
├── scripts/
│   ├── ingestion/
│   │   └── ingest.py         # DataIngestor — text / image / CSV ingestion + CLI
│   ├── generation/
│   │   └── generate.py       # Generator — free LLM wrapper + full RAG CLI
│   ├── evaluation/
│   │   └── evaluate.py       # Recall@K, Precision@K, MRR metrics + CLI
│   └── query.py              # Raw retrieval CLI (no generation)
│
├── data/
│   ├── text/                 # Drop .txt files here before ingestion
│   └── images/               # Drop .png / .jpg files here before ingestion
│
├── stores/                   # Auto-created by ingest.py
│   ├── text_store/
│   └── image_store/
│
├── tests/
├── main.py                   # FastAPI HTTP API
├── debug.py                  # Store diagnostics
├── requirements.txt
└── README.md
```

---

## Requirements

| Requirement    | Details                                                                      |
| -------------- | ---------------------------------------------------------------------------- |
| Python         | 3.9 or higher                                                                |
| RAM (CPU mode) | ~2 GB for flan-t5-base, ~6 GB for phi-2                                      |
| RAM (GPU mode) | ~1 GB VRAM for flan-t5-base (float16)                                        |
| GPU            | Optional — any CUDA-capable Nvidia GPU. Falls back to CPU automatically.     |
| Internet       | First run only — models download from HuggingFace Hub and are cached locally |

---

## Installation

```bash
# 1. Clone / download the project
cd multimodal_rag

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (GPU only) Install PyTorch with CUDA support
#    Visit https://pytorch.org/get-started/locally/ for the right command.
#    Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

No API keys needed. No `.env` file required for default usage.

---

## Quick Start

### Step 1 — Add your data

```
data/text/          ← drop .txt and .csv files here
data/images/        ← drop images here (sub-folders like bike/ are found automatically)
```

Supported image formats: `.bmp`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`

### Step 2 — Ingest

```bash
python -m scripts.ingestion.ingest \
    --text_folder data/text \
    --image_folder data/images
```

This creates `stores/text_store/` and `stores/image_store/` automatically.

### Step 3 — Query

```bash
# Ask a question (uses flan-t5-base by default — free, no key needed)
python -m scripts.generation.generate --query "What is CLIP?"

# Hybrid retrieval (text + image)
python -m scripts.generation.generate --query "cat on a sofa" --mode hybrid

# Use a stronger free model
python -m scripts.generation.generate --query "What is CLIP?" --backend phi2
```

---

## Data Ingestion

### Expected folder layout

```
data/
├── text/
│   ├── report.txt           ← plain text files
│   └── documents.csv        ← CSV files auto-detected alongside .txt files
└── images/
    ├── photo.jpg             ← top-level images work fine
    └── bike/                 ← sub-folders are walked recursively
        ├── 1.bmp             ← .bmp fully supported
        ├── 2.png
        └── racing/
            └── 3.jpg
```

### Run ingestion

```bash
python -m scripts.ingestion.ingest \
    --text_folder data/text \
    --image_folder data/images
```

That single command handles everything automatically:

- **Text folder** — scans for `.txt` and `.csv` files. CSV files sitting inside `data/text/` (like `data/text/documents.csv`) are picked up without any extra flag.
- **Image folder** — walks the full directory tree recursively, so `bike/1.bmp`, `bike/racing/3.jpg`, and any other nested images are all found.

### Supported file formats

| Type   | Formats                                                                   |
| ------ | ------------------------------------------------------------------------- |
| Text   | `.txt`                                                                    |
| CSV    | `.csv` (must have a `text` column; `source_url` used as label if present) |
| Images | `.bmp`, `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.tif`                 |

### What happens to each file type

**`.txt` files** — read as plain text → split into 500-char chunks with 100-char overlap → batch-encoded with SentenceTransformer → stored in the text FAISS store.

**`.csv` files** — every row's `text` column is chunked and embedded the same way as `.txt`. The `source_url` column is used as the source label if present; otherwise the CSV filename is used.

**Images** — encoded with CLIP (512-d) → stored in the image FAISS store. The sub-folder path is preserved in the stored ID (e.g. `bike/1.bmp`) so results are fully traceable back to their origin.

### Optional: explicit extra CSV path

If you have a CSV file outside the text folder, pass it separately:

```bash
python -m scripts.ingestion.ingest \
    --text_folder data/text \
    --image_folder data/images \
    --csv_file /other/path/extra.csv
```

---

## Free Model Backends

All models are downloaded from HuggingFace Hub on first use and cached locally.
After the first download, the system works fully offline.

| Backend                  | Model                 | Size    | Task     | Best for                                |
| ------------------------ | --------------------- | ------- | -------- | --------------------------------------- |
| `flan-t5` ✅ **default** | `google/flan-t5-base` | ~250 MB | seq2seq  | Q&A, CPU-friendly                       |
| `phi2`                   | `microsoft/phi-2`     | ~6 GB   | text-gen | Better reasoning, needs GPU or high RAM |
| `gpt2`                   | `gpt2`                | ~500 MB | text-gen | Smoke-testing, minimal hardware         |

### Selecting a backend

```bash
# Default (flan-t5-base)
python -m scripts.generation.generate --query "..."

# Phi-2 (stronger)
python -m scripts.generation.generate --query "..." --backend phi2

# GPT-2 (lightest)
python -m scripts.generation.generate --query "..." --backend gpt2

# Any custom HuggingFace model (e.g. larger flan-t5)
python -m scripts.generation.generate \
    --query "..." \
    --custom_model google/flan-t5-large

# List all built-in backends
python -m scripts.generation.generate --list_backends
```

### Why flan-t5 is the default

Flan-T5 is a **seq2seq** (encoder-decoder) model. For Q&A this has two key advantages over decoder-only models like GPT-2:

- The output is the **answer only** — no prompt text is included, so no stripping logic is needed.
- It was explicitly fine-tuned for instruction-following tasks, so it reliably answers "using only the context" rather than hallucinating.

---

## Retrieval Modes

| Mode            | Query input     | Searches                | Embedding used             |
| --------------- | --------------- | ----------------------- | -------------------------- |
| `text`          | Text string     | Text FAISS store        | SentenceTransformer 384-d  |
| `image`         | Image file path | Image FAISS store       | CLIP image encoder 512-d   |
| `text_to_image` | Text string     | Image FAISS store       | CLIP text encoder 512-d    |
| `hybrid`        | Text string     | Both stores, then fused | Both above, merged via RRF |

**Hybrid mode** runs text retrieval and cross-modal image retrieval in parallel,
then merges the two ranked lists using **Reciprocal Rank Fusion (RRF)** — a
rank-based algorithm that is robust to score scale differences between modalities.

---

## Running the API

```bash
uvicorn main:app --reload --port 8000
```

Interactive Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### `GET /` — Health check

```bash
curl http://localhost:8000/
```

```json
{ "status": "running", "text_vectors": 1240, "image_vectors": 87 }
```

### `POST /query` — Retrieve + generate

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is neural scaling?",
       "mode": "hybrid",
       "top_k": 5,
       "alpha": 0.6
     }'
```

```json
{
  "query": "What is neural scaling?",
  "mode": "hybrid",
  "answer": "Neural scaling refers to ...",
  "sources": [ { "id": "...", "score": 0.91, "metadata": { ... } } ],
  "context_preview": "..."
}
```

**Request fields:**

| Field   | Type   | Default  | Description                                                |
| ------- | ------ | -------- | ---------------------------------------------------------- |
| `query` | string | required | Natural-language question                                  |
| `mode`  | string | `"text"` | `text` / `hybrid` / `text_to_image`                        |
| `top_k` | int    | `5`      | Number of chunks to retrieve (1–20)                        |
| `alpha` | float  | `0.5`    | Text weight in weighted fusion (0=image only, 1=text only) |

### `POST /image-query` — Image upload + generate

```bash
curl -X POST http://localhost:8000/image-query \
     -F "file=@photo.jpg"
```

Accepts PNG, JPEG, WEBP. Returns visually similar documents and a generated answer.

### Selecting backend via environment variable

```bash
# Use phi-2 for the API server
GENERATOR_BACKEND=phi2 uvicorn main:app --reload

# Use any custom HuggingFace model
CUSTOM_MODEL=google/flan-t5-large uvicorn main:app --reload
```

---

## CLI Reference

### Ingestion

```bash
python -m scripts.ingestion.ingest [OPTIONS]

  --text_folder    PATH   Folder of .txt files          (default: data/text)
  --image_folder   PATH   Folder of image files         (default: data/images)
  --csv_file       PATH   CSV file with a 'text' column (optional)
```

### Generation (full RAG pipeline)

```bash
python -m scripts.generation.generate [OPTIONS]

  --query          TEXT   Question to answer             (required)
  --mode           STR    text | hybrid | text_to_image  (default: text)
  --backend        STR    flan-t5 | phi2 | gpt2          (default: flan-t5)
  --custom_model   STR    Any HuggingFace model ID       (overrides --backend)
  --top_k          INT    Chunks to retrieve             (default: 5)
  --max_new_tokens INT    Tokens to generate             (default: 256)
  --list_backends         Print available backends and exit
```

### Raw retrieval (no generation)

```bash
python -m scripts.query [OPTIONS]

  --query   TEXT   Text query                            (for text/hybrid/text_to_image)
  --image   PATH   Image file path                       (for image mode)
  --mode    STR    text | image | hybrid | text_to_image (default: text)
  --top_k   INT    Results to return                     (default: 5)
  --json           Output raw JSON instead of formatted text
```

### Evaluation

```bash
python -m scripts.evaluation.evaluate [OPTIONS]

  --mode       STR    text | hybrid | text_to_image      (default: text)
  --k          INT    Rank cutoff for Recall/Precision   (default: 5)
  --eval_json  PATH   JSON file with ground-truth data   (optional)
```

### Store diagnostics

```bash
python debug.py
```

Prints vector counts, dimensions, sample IDs, and sample metadata for both stores.

---

## Evaluation

### Metrics

| Metric          | Formula                       | Meaning                                       |
| --------------- | ----------------------------- | --------------------------------------------- |
| **Recall@K**    | `hits / total_relevant`       | Did we find all relevant docs in top-K?       |
| **Precision@K** | `hits / K`                    | How many of the top-K results are relevant?   |
| **MRR**         | `mean(1 / rank_of_first_hit)` | How high does the first relevant result rank? |

### Ground-truth format

Create a JSON file with this structure:

```json
[
  {
    "query": "What is CLIP?",
    "relevant_ids": ["clip_paper_0", "clip_paper_1"]
  },
  {
    "query": "cat sitting on a mat",
    "relevant_ids": ["cat_img_001"]
  }
]
```

The `relevant_ids` must match the IDs assigned during ingestion
(format: `<source_filename>_<chunk_index>` for text, `<filename>` for images).

### Running evaluation

```bash
# With built-in dummy data
python -m scripts.evaluation.evaluate --mode text --k 5

# With your own ground truth
python -m scripts.evaluation.evaluate \
    --eval_json my_eval.json \
    --mode hybrid \
    --k 10
```

---

## Configuration Reference

### Code-level parameters

| Parameter         | File / Location             | Default             | Description                                      |
| ----------------- | --------------------------- | ------------------- | ------------------------------------------------ |
| `chunk_size`      | `DataIngestor.__init__`     | `500`               | Max characters per text chunk                    |
| `chunk_overlap`   | `DataIngestor.__init__`     | `100`               | Characters shared between adjacent chunks        |
| `batch_size`      | `DataIngestor.__init__`     | auto                | 256 on GPU, 32 on CPU                            |
| `fusion_strategy` | `Retriever.__init__`        | `"reciprocal_rank"` | `weighted_sum` / `max_score` / `reciprocal_rank` |
| `alpha`           | `retrieve_hybrid()`         | `0.5`               | Text weight in `weighted_sum` fusion             |
| `max_chars`       | `Retriever.build_context()` | `4000`              | Context character budget passed to LLM           |
| `max_new_tokens`  | `Generator.generate()`      | `256`               | Maximum tokens in generated answer               |

### Environment variables

| Variable            | Default   | Description                                              |
| ------------------- | --------- | -------------------------------------------------------- |
| `GENERATOR_BACKEND` | `flan-t5` | Free model backend: `flan-t5`, `phi2`, or `gpt2`         |
| `CUSTOM_MODEL`      | _(unset)_ | Any HuggingFace model ID — overrides `GENERATOR_BACKEND` |

---

## GPU vs CPU

No configuration is needed. Every component detects hardware automatically at startup via `torch.cuda.is_available()`.

| Component                  | CPU behaviour         | GPU behaviour                               |
| -------------------------- | --------------------- | ------------------------------------------- |
| SentenceTransformer        | Runs on CPU           | Moves to GPU automatically                  |
| CLIP (image + cross-modal) | Runs on CPU           | Moves to GPU automatically                  |
| HuggingFace generator      | Runs on CPU (float32) | Moves to GPU (float16 — ~half VRAM)         |
| FAISS index                | Always on CPU         | Vectors transferred from GPU after encoding |
| Ingestion batch size       | Auto: 32              | Auto: 256                                   |

To verify which device is being used, check the startup logs:

```
INFO  Generator ready. model=google/flan-t5-base, device=GPU (CUDA).
```

---

## Extending the System

### Use a larger / better free model

```bash
# Flan-T5 large (800 MB, noticeably better than base)
python -m scripts.generation.generate --custom_model google/flan-t5-large

# Flan-T5 XL (3 GB, strong instruction following)
python -m scripts.generation.generate --custom_model google/flan-t5-xl
```

Or register it permanently in the `FREE_MODELS` dict in `generate.py`:

```python
FREE_MODELS["flan-t5-large"] = {
    "model_id": "google/flan-t5-large",
    "task": "text2text-generation",
    "description": "Flan-T5 large (~800 MB, better quality)",
}
```

### Swap the text encoder

Change `text_model_name` in `EmbeddingModel.__init__` and update the `VectorStore`
dimension to match (e.g. `dim=768` for `all-mpnet-base-v2`):

```python
# embeddings.py
text_model_name = "sentence-transformers/all-mpnet-base-v2"

# ingest.py
self.text_store = VectorStore(dim=768)
```

### Switch to approximate search (millions of vectors)

Replace `faiss.IndexFlatIP` in `VectorStore.__init__` with an IVF index:

```python
quantiser = faiss.IndexFlatIP(dim)
self.index = faiss.IndexIVFFlat(quantiser, dim, 100)  # 100 Voronoi cells
self.index.train(training_vectors)                     # must train before adding
```

### Add BM25 sparse retrieval

Install `rank_bm25`, build a BM25 index over the same chunks, retrieve sparse
results, and pass them alongside the dense results to `MultiModalFusion.fuse()`.
The RRF strategy handles heterogeneous score distributions automatically.

### Add a new fusion strategy

Subclass or extend `MultiModalFusion` and register it in the `fuse()` method:

```python
elif self.strategy == "my_strategy":
    return self._my_strategy(text_results, image_results)
```
