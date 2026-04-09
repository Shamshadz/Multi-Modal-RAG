import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any


class VectorStore:
    def __init__(self, dim: int, index_type: str = "cosine"):
        """
        dim: embedding dimension
        index_type: 'cosine' or 'l2'
        """
        self.dim = dim
        self.index_type = index_type

        if index_type == "cosine":
            self.index = faiss.IndexFlatIP(dim)  # inner product
        elif index_type == "l2":
            self.index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("index_type must be 'cosine' or 'l2'")

        self.metadata = []
        self.id_map = []

    # -----------------------------
    # ADD VECTORS
    # -----------------------------
    def add(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ):
        """
        Add vectors with optional metadata and ids
        """
        vectors = np.array(vectors).astype("float32")

        if self.index_type == "cosine":
            faiss.normalize_L2(vectors)

        self.index.add(vectors)

        # Store metadata
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{}] * len(vectors))

        # Store IDs
        if ids:
            self.id_map.extend(ids)
        else:
            self.id_map.extend([str(len(self.id_map) + i) for i in range(len(vectors))])

    # -----------------------------
    # SEARCH
    # -----------------------------
    def search(self, query_vector: List[float], top_k: int = 5):
        """
        Search for similar vectors
        """
        query = np.array([query_vector]).astype("float32")

        if self.index_type == "cosine":
            faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "id": self.id_map[idx],
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results

    # -----------------------------
    # SAVE
    # -----------------------------
    def save(self, path: str):
        """
        Save FAISS index + metadata
        """
        os.makedirs(path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "meta.pkl"), "wb") as f:
            pickle.dump({
                "metadata": self.metadata,
                "id_map": self.id_map,
                "dim": self.dim,
                "index_type": self.index_type
            }, f)

    # -----------------------------
    # LOAD
    # -----------------------------
    @classmethod
    def load(cls, path: str):
        """
        Load FAISS index + metadata
        """
        index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "meta.pkl"), "rb") as f:
            data = pickle.load(f)

        obj = cls(dim=data["dim"], index_type=data["index_type"])
        obj.index = index
        obj.metadata = data["metadata"]
        obj.id_map = data["id_map"]

        return obj