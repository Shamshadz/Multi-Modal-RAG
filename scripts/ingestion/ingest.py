"""
ingest.py
=========
Data ingestion pipeline: reads raw text files and images from disk, encodes
them into dense vectors, and writes two FAISS stores to disk.

Pipeline overview
-----------------
Text files
~~~~~~~~~~
1. Read ``.txt`` files from a folder.
2. Split each file into overlapping chunks (:class:`~core.chunking.TextChunker`).
3. Batch-encode chunks with SentenceTransformer (384-d).
4. Insert vectors + chunk metadata into the text ``VectorStore``.

Images
~~~~~~
1. Discover ``.png / .jpg / .jpeg`` files in a folder.
2. Encode each image with CLIP (512-d).
3. Insert vector + image path metadata into the image ``VectorStore``.

CSV files
~~~~~~~~~
Optionally ingest a CSV with ``text`` and ``source_url`` columns via
:meth:`DataIngestor.ingest_csv`.  Useful for pre-scraped document corpora.

Running
-------
.. code-block:: bash

    python -m scripts.ingestion.ingest \\
        --text_folder data/text \\
        --image_folder data/images \\
        --csv_file data/documents.csv

Stores are written to ``stores/text_store`` and ``stores/image_store``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import List

import torch
from tqdm import tqdm

from core.chunking import TextChunker
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _optimal_batch_size() -> int:
    """
    Return a sensible embedding batch size based on available hardware.

    * GPU present  → 256  (GPU memory bandwidth makes large batches fast)
    * CPU only     → 32   (avoids excessive RAM usage per batch)

    Override by passing ``batch_size`` explicitly to :class:`DataIngestor`.
    """
    if torch.cuda.is_available():
        logger.info("GPU detected — using batch_size=256 for embedding.")
        return 256
    logger.info("No GPU detected — using batch_size=32 for embedding (CPU).")
    return 32


class DataIngestor:
    """
    Orchestrates the ingestion of text files, image files, and CSV corpora.

    Parameters
    ----------
    text_store_path : str
        Directory where the text ``VectorStore`` will be saved.
    image_store_path : str
        Directory where the image ``VectorStore`` will be saved.
    chunk_size : int
        Maximum characters per text chunk.
    chunk_overlap : int
        Character overlap between consecutive chunks.
    batch_size : int or None
        Number of text chunks to encode per ``embed_text`` call.
        Defaults to ``None``, which auto-selects 256 on GPU or 32 on CPU.
    """

    def __init__(
        self,
        text_store_path: str = "stores/text_store",
        image_store_path: str = "stores/image_store",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        batch_size: int = None,
    ):
        self.text_store_path = text_store_path
        self.image_store_path = image_store_path
        self.batch_size = batch_size if batch_size is not None else _optimal_batch_size()

        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingModel()   # auto-selects GPU/CPU internally

        # 384-d for MiniLM, 512-d for CLIP
        self.text_store = VectorStore(dim=384)
        self.image_store = VectorStore(dim=512)

    # ------------------------------------------------------------------
    # Text ingestion
    # ------------------------------------------------------------------

    def ingest_text_files(self, folder_path: str) -> None:
        """
        Ingest all ``.txt`` files in *folder_path*.

        Parameters
        ----------
        folder_path : str
            Directory containing ``.txt`` files.
        """
        files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        if not files:
            logger.warning("No .txt files found in '%s'.", folder_path)
            return

        for file in tqdm(files, desc="Text files"):
            full_path = os.path.join(folder_path, file)
            with open(full_path, "r", encoding="utf-8") as fh:
                content = fh.read()

            self._ingest_text(content, source=file)

    def ingest_csv(self, csv_path: str, text_col: str = "text", source_col: str = "source_url") -> None:
        """
        Ingest text rows from a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.
        text_col : str
            Column name containing document text.
        source_col : str
            Column name used as the ``source`` metadata field.
        """
        if not os.path.isfile(csv_path):
            logger.error("CSV file not found: %s", csv_path)
            return

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        logger.info("Ingesting %d rows from '%s'.", len(rows), csv_path)

        for row in tqdm(rows, desc="CSV rows"):
            text = row.get(text_col, "").strip()
            source = row.get(source_col, "unknown")
            if text:
                self._ingest_text(text, source=source)

    def _ingest_text(self, content: str, source: str) -> None:
        """Chunk *content*, embed in batches, and add to text_store."""
        chunks = self.chunker.chunk_text(content, metadata={"source": source})
        if not chunks:
            return

        # Batch encode for efficiency
        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start : batch_start + self.batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedder.embed_text(texts)

            # Store the chunk text inside metadata for context building
            metadatas = []
            for c in batch:
                m = c["metadata"].copy()
                m["text"] = c["text"]
                metadatas.append(m)

            ids = [
                f"{source}_{batch_start + i}"
                for i in range(len(batch))
            ]

            self.text_store.add(embeddings, metadatas=metadatas, ids=ids)

    # ------------------------------------------------------------------
    # Image ingestion
    # ------------------------------------------------------------------

    def ingest_images(self, folder_path: str) -> None:
        """
        Ingest all ``.png / .jpg / .jpeg`` images in *folder_path*.

        Parameters
        ----------
        folder_path : str
            Directory containing image files.
        """
        exts = (".png", ".jpg", ".jpeg", ".webp")
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(exts)]
        if not files:
            logger.warning("No image files found in '%s'.", folder_path)
            return

        for file in tqdm(files, desc="Images"):
            full_path = os.path.join(folder_path, file)
            try:
                embedding = self.embedder.embed_image(full_path)[0]
            except Exception as exc:
                logger.warning("Skipping '%s': %s", file, exc)
                continue

            metadata = {
                "image_path": full_path,
                "source": file,
                "text": f"[image: {file}]",  # placeholder for context building
            }
            self.image_store.add([embedding], metadatas=[metadata], ids=[file])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save both stores to disk."""
        os.makedirs("stores", exist_ok=True)
        self.text_store.save(self.text_store_path)
        self.image_store.save(self.image_store_path)
        logger.info(
            "Saved text_store (%d vectors) and image_store (%d vectors).",
            len(self.text_store),
            len(self.image_store),
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Multi-Modal RAG – data ingestion.")
    parser.add_argument("--text_folder", type=str, default="data/text")
    parser.add_argument("--image_folder", type=str, default="data/images")
    parser.add_argument("--csv_file", type=str, default="", help="Optional CSV corpus.")
    args = parser.parse_args()

    ingestor = DataIngestor()

    if os.path.isdir(args.text_folder):
        ingestor.ingest_text_files(args.text_folder)
    else:
        logger.warning("Text folder '%s' does not exist – skipping.", args.text_folder)

    if os.path.isdir(args.image_folder):
        ingestor.ingest_images(args.image_folder)
    else:
        logger.warning("Image folder '%s' does not exist – skipping.", args.image_folder)

    if args.csv_file:
        ingestor.ingest_csv(args.csv_file)

    ingestor.save()
    print("✅ Ingestion complete.")


if __name__ == "__main__":
    main()