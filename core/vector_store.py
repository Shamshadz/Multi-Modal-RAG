"""
vector_store.py
===============
Wraps a FAISS flat index with metadata storage, persistence helpers, and a
clean search interface.

Why FAISS flat index?
---------------------
``IndexFlatIP`` (inner-product / cosine when vectors are L2-normalised) and
``IndexFlatL2`` are *exact* nearest-neighbour indexes. They are simple,
deterministic, and fast enough for collections up to ~1 M vectors on CPU.
For larger corpora, swap to ``IndexIVFFlat`` or ``IndexHNSWFlat``.

Storage layout (on disk)
------------------------
``<path>/index.faiss``   – the FAISS index binary
``<path>/meta.pkl``      – Python dict with metadata list, id map, dim, index_type

Usage
-----
>>> store = VectorStore(dim=384)
>>> store.add([[0.1]*384], metadatas=[{"text": "hello"}], ids=["doc_0"])
>>> results = store.search([0.1]*384, top_k=3)
>>> store.save("stores/text_store")
>>> store2 = VectorStore.load("stores/text_store")
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin wrapper around a FAISS flat index.

    Parameters
    ----------
    dim : int
        Dimensionality of the vectors to store.
    index_type : str
        ``"cosine"`` uses ``IndexFlatIP`` with L2-normalisation before
        insertion/search (equivalent to cosine similarity).
        ``"l2"`` uses ``IndexFlatL2``.
    """

    def __init__(self, dim: int, index_type: str = "cosine"):
        if index_type not in ("cosine", "l2"):
            raise ValueError("index_type must be 'cosine' or 'l2'")

        self.dim = dim
        self.index_type = index_type

        if index_type == "cosine":
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = faiss.IndexFlatL2(dim)

        self.metadata: List[Dict[str, Any]] = []
        self.id_map: List[str] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Insert vectors (and optional metadata/IDs) into the index.

        Parameters
        ----------
        vectors : list[list[float]]
            Dense embeddings to insert.  Shape: ``(n, dim)``.
        metadatas : list[dict], optional
            One metadata dict per vector. Missing entries default to ``{}``.
        ids : list[str], optional
            String identifiers for each vector.  Auto-generated (sequential)
            when omitted.

        Notes
        -----
        Vectors are L2-normalised *in place* before insertion when
        ``index_type == "cosine"``.  Pass already-normalised vectors to avoid
        double-normalisation artefacts.
        """
        if not vectors:
            return

        arr = np.array(vectors, dtype="float32")

        if self.index_type == "cosine":
            faiss.normalize_L2(arr)

        self.index.add(arr)

        n = len(vectors)
        base = len(self.id_map)

        self.metadata.extend(metadatas if metadatas else [{}] * n)
        self.id_map.extend(ids if ids else [str(base + i) for i in range(n)])

        logger.debug("Added %d vectors. Total: %d", n, len(self.id_map))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return the *top_k* most similar vectors.

        Parameters
        ----------
        query_vector : list[float]
            A single query embedding.
        top_k : int
            Number of neighbours to return.

        Returns
        -------
        list[dict]
            Each dict has keys ``"id"``, ``"score"``, ``"metadata"``::

                [
                    {"id": "doc_0", "score": 0.97, "metadata": {...}},
                    ...
                ]
        """
        if self.index.ntotal == 0:
            logger.warning("VectorStore is empty – returning no results.")
            return []

        query = np.array([query_vector], dtype="float32")

        if self.index_type == "cosine":
            faiss.normalize_L2(query)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "id": self.id_map[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
            })

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Persist the FAISS index and metadata to *path*.

        Parameters
        ----------
        path : str
            Directory path (created if it does not exist).
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        payload = {
            "metadata": self.metadata,
            "id_map": self.id_map,
            "dim": self.dim,
            "index_type": self.index_type,
        }
        with open(os.path.join(path, "meta.pkl"), "wb") as fh:
            pickle.dump(payload, fh)

        logger.info("VectorStore saved to '%s' (%d vectors).", path, len(self.id_map))

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """
        Restore a previously saved VectorStore.

        Parameters
        ----------
        path : str
            Directory path written by :meth:`save`.

        Returns
        -------
        VectorStore
        """
        index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "meta.pkl"), "rb") as fh:
            data = pickle.load(fh)

        obj = cls(dim=data["dim"], index_type=data["index_type"])
        obj.index = index
        obj.metadata = data["metadata"]
        obj.id_map = data["id_map"]

        logger.info("VectorStore loaded from '%s' (%d vectors).", path, len(obj.id_map))
        return obj

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.index.ntotal

    def __repr__(self) -> str:
        return (
            f"VectorStore(dim={self.dim}, index_type={self.index_type!r}, "
            f"n_vectors={len(self)})"
        )