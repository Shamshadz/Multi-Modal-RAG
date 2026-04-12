"""
ingest.py
=========
Data ingestion pipeline: reads raw text/CSV files and images from disk,
encodes them into dense vectors, and writes two FAISS stores to disk.

Supported folder layouts
------------------------
Text folder (``data/text/``)
    ├── report.txt          ← plain text files
    ├── documents.csv       ← CSV files are auto-detected and ingested
    └── notes.txt

Image folder (``data/images/``)
    ├── photo.jpg           ← top-level images
    └── bike/               ← sub-folders are walked recursively
        ├── 1.bmp
        ├── 2.png
        └── racing/
            └── 3.jpg

Supported image formats
-----------------------
``.bmp``, ``.png``, ``.jpg``, ``.jpeg``, ``.webp``, ``.tiff``, ``.tif``

Supported text formats
----------------------
``.txt``  — read as plain text, chunked, then embedded.
``.csv``  — every row's ``text`` column is chunked and embedded.
           The ``source_url`` column is used as the source label if present.

Running
-------
.. code-block:: bash

    python -m scripts.ingestion.ingest \\
        --text_folder data/text \\
        --image_folder data/images

Stores are written to ``stores/text_store`` and ``stores/image_store``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from typing import List, Tuple

import torch
from tqdm import tqdm

from core.chunking import TextChunker
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

#: All image extensions that PIL / CLIP can handle.
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".bmp",   # your format — added
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tiff",
    ".tif",
)

#: Text-based extensions handled in the text folder.
TEXT_EXTENSIONS: Tuple[str, ...] = (".txt",)
CSV_EXTENSIONS: Tuple[str, ...] = (".csv",)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optimal_batch_size() -> int:
    """
    Return a sensible embedding batch size based on available hardware.

    * GPU present  → 256  (GPU parallelism makes large batches fast)
    * CPU only     → 32   (avoids excessive RAM usage per batch)

    Override by passing ``batch_size`` explicitly to :class:`DataIngestor`.
    """
    if torch.cuda.is_available():
        logger.info("GPU detected — using batch_size=256 for embedding.")
        return 256
    logger.info("No GPU detected — using batch_size=32 for embedding (CPU).")
    return 32


def _walk_files(root: str, extensions: Tuple[str, ...]) -> List[str]:
    """
    Recursively collect all files under *root* whose extension is in
    *extensions* (case-insensitive).

    Parameters
    ----------
    root : str
        Top-level directory to walk.
    extensions : tuple[str, ...]
        Lower-case extensions to match, e.g. ``(".jpg", ".bmp")``.

    Returns
    -------
    list[str]
        Sorted list of absolute file paths.

    Example
    -------
    For the tree::

        data/images/
        ├── cat.jpg
        └── bike/
            └── 1.bmp

    ``_walk_files("data/images", (".jpg", ".bmp"))`` returns::

        ["data/images/bike/1.bmp", "data/images/cat.jpg"]
    """
    matched: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            if os.path.splitext(fname.lower())[1] in extensions:
                matched.append(os.path.join(dirpath, fname))
    return sorted(matched)


def _relative_id(base_folder: str, full_path: str) -> str:
    """
    Return a stable, human-readable ID for a file relative to *base_folder*.

    Example: ``data/images/bike/1.bmp`` → ``bike/1.bmp``

    This preserves sub-folder context in the stored IDs so results are
    traceable back to their origin.
    """
    return os.path.relpath(full_path, base_folder)


# ---------------------------------------------------------------------------
# DataIngestor
# ---------------------------------------------------------------------------

class DataIngestor:
    """
    Orchestrates ingestion of text files, CSV files, and image files
    (including nested sub-folders and .bmp format).

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
        Embedding batch size. ``None`` auto-selects 256 (GPU) or 32 (CPU).
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

        # 384-d for MiniLM text encoder, 512-d for CLIP image encoder
        self.text_store = VectorStore(dim=384)
        self.image_store = VectorStore(dim=512)

    # ------------------------------------------------------------------
    # Public: text folder ingestion
    # ------------------------------------------------------------------

    def ingest_text_folder(self, folder_path: str) -> None:
        """
        Ingest all text-like files in *folder_path*.

        Handles both ``.txt`` and ``.csv`` files found anywhere in the
        folder (non-recursive for text — all files expected at top level).

        Parameters
        ----------
        folder_path : str
            Path to the text data folder (e.g. ``data/text``).
        """
        if not os.path.isdir(folder_path):
            logger.warning("Text folder '%s' does not exist — skipping.", folder_path)
            return

        # Collect .txt and .csv files (flat scan — text files are not nested)
        all_files = os.listdir(folder_path)
        txt_files = [f for f in all_files if f.lower().endswith(TEXT_EXTENSIONS)]
        csv_files = [f for f in all_files if f.lower().endswith(CSV_EXTENSIONS)]

        if not txt_files and not csv_files:
            logger.warning(
                "No .txt or .csv files found in '%s'.", folder_path
            )
            return

        logger.info(
            "Found %d .txt file(s) and %d .csv file(s) in '%s'.",
            len(txt_files), len(csv_files), folder_path,
        )

        # --- Ingest plain text files ---
        for fname in tqdm(txt_files, desc=f"TXT [{folder_path}]"):
            full_path = os.path.join(folder_path, fname)
            try:
                with open(full_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                self._ingest_text(content, source=fname)
            except Exception as exc:
                logger.warning("Skipping '%s': %s", fname, exc)

        # --- Ingest CSV files ---
        for fname in tqdm(csv_files, desc=f"CSV [{folder_path}]"):
            full_path = os.path.join(folder_path, fname)
            self._ingest_csv_file(full_path)

    # ------------------------------------------------------------------
    # Public: image folder ingestion (recursive)
    # ------------------------------------------------------------------

    def ingest_images(self, folder_path: str) -> None:
        """
        Recursively ingest all images under *folder_path*.

        Walks the full directory tree, so sub-folders like
        ``data/images/bike/`` are included automatically.

        Supported formats: ``.bmp``, ``.png``, ``.jpg``, ``.jpeg``,
        ``.webp``, ``.tiff``, ``.tif``.

        Parameters
        ----------
        folder_path : str
            Root image directory (e.g. ``data/images``).
        """
        if not os.path.isdir(folder_path):
            logger.warning("Image folder '%s' does not exist — skipping.", folder_path)
            return

        files = _walk_files(folder_path, IMAGE_EXTENSIONS)

        if not files:
            logger.warning(
                "No image files found under '%s' (checked extensions: %s).",
                folder_path, IMAGE_EXTENSIONS,
            )
            return

        logger.info(
            "Found %d image(s) under '%s' (recursive).", len(files), folder_path
        )

        for full_path in tqdm(files, desc=f"Images [{folder_path}]"):
            # Use relative path as ID so sub-folder info is preserved
            # e.g.  "bike/1.bmp"  instead of just "1.bmp"
            rel_id = _relative_id(folder_path, full_path)

            try:
                embedding = self.embedder.embed_image(full_path)[0]
            except Exception as exc:
                logger.warning("Skipping image '%s': %s", rel_id, exc)
                continue

            metadata = {
                "image_path": full_path,
                "source": rel_id,
                # Placeholder text so build_context() can reference images
                "text": f"[image: {rel_id}]",
            }
            self.image_store.add([embedding], metadatas=[metadata], ids=[rel_id])

    # ------------------------------------------------------------------
    # Public: standalone CSV ingestion (path given explicitly)
    # ------------------------------------------------------------------

    def ingest_csv(
        self,
        csv_path: str,
        text_col: str = "text",
        source_col: str = "source_url",
    ) -> None:
        """
        Ingest a single CSV file given its explicit path.

        Use this when your CSV lives outside the text folder
        (e.g. ``--csv_file data/text/documents.csv`` from the CLI).

        Parameters
        ----------
        csv_path : str
            Absolute or relative path to the CSV file.
        text_col : str
            Column name containing document text.
        source_col : str
            Column name used as the source metadata label.
        """
        if not os.path.isfile(csv_path):
            logger.error("CSV file not found: '%s'", csv_path)
            return
        self._ingest_csv_file(csv_path, text_col=text_col, source_col=source_col)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save both FAISS stores to disk."""
        os.makedirs("stores", exist_ok=True)
        self.text_store.save(self.text_store_path)
        self.image_store.save(self.image_store_path)
        logger.info(
            "Saved text_store (%d vectors) and image_store (%d vectors).",
            len(self.text_store),
            len(self.image_store),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ingest_csv_file(
        self,
        csv_path: str,
        text_col: str = "text",
        source_col: str = "source_url",
    ) -> None:
        """Read every row of *csv_path* and ingest the text column."""
        try:
            with open(csv_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except Exception as exc:
            logger.error("Could not read CSV '%s': %s", csv_path, exc)
            return

        if not rows:
            logger.warning("CSV '%s' is empty — skipping.", csv_path)
            return

        # Validate column presence
        sample = rows[0]
        if text_col not in sample:
            available = list(sample.keys())
            logger.error(
                "CSV '%s' has no column '%s'. Available columns: %s",
                csv_path, text_col, available,
            )
            return

        csv_name = os.path.basename(csv_path)
        logger.info("Ingesting %d rows from '%s'.", len(rows), csv_name)

        for row in tqdm(rows, desc=f"CSV [{csv_name}]", leave=False):
            text = row.get(text_col, "").strip()
            source = row.get(source_col, csv_name)   # fall back to filename
            if text:
                self._ingest_text(text, source=source)

    def _ingest_text(self, content: str, source: str) -> None:
        """Chunk *content*, batch-embed, and add to the text store."""
        chunks = self.chunker.chunk_text(content, metadata={"source": source})
        if not chunks:
            return

        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start : batch_start + self.batch_size]
            texts = [c["text"] for c in batch]
            embeddings = self.embedder.embed_text(texts)

            metadatas = []
            for c in batch:
                m = c["metadata"].copy()
                m["text"] = c["text"]   # store chunk text for context building
                metadatas.append(m)

            ids = [
                f"{source}_{batch_start + i}"
                for i in range(len(batch))
            ]

            self.text_store.add(embeddings, metadatas=metadatas, ids=ids)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description=(
            "Multi-Modal RAG — data ingestion.\n\n"
            "Folder layout expected:\n"
            "  data/text/            ← .txt and .csv files (flat)\n"
            "  data/images/          ← images in any sub-folder (recursive)\n"
            "  data/images/bike/     ← e.g. bike/1.bmp is found automatically"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text_folder",
        type=str,
        default="data/text",
        help="Folder containing .txt and/or .csv files (default: data/text).",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="data/images",
        help="Root image folder — sub-folders are walked recursively (default: data/images).",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="",
        help="Explicit path to a CSV file (optional, used in addition to --text_folder).",
    )
    args = parser.parse_args()

    ingestor = DataIngestor()

    # Ingest text folder (.txt + .csv files discovered automatically)
    ingestor.ingest_text_folder(args.text_folder)

    # Ingest images (recursive, includes .bmp and sub-folders)
    ingestor.ingest_images(args.image_folder)

    # Optionally ingest an additional explicit CSV path
    if args.csv_file:
        ingestor.ingest_csv(args.csv_file)

    ingestor.save()

    print("\n✅ Ingestion complete.")
    print(f"   Text vectors  : {len(ingestor.text_store)}")
    print(f"   Image vectors : {len(ingestor.image_store)}")


if __name__ == "__main__":
    main()