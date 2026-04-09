from typing import List, Dict, Any, Optional

from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        text_store: Optional[VectorStore] = None,
        image_store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingModel] = None,
    ):
        self.text_store = text_store
        self.image_store = image_store
        self.embedder = embedder or EmbeddingModel()

    # -----------------------------
    # TEXT QUERY → TEXT SEARCH
    # -----------------------------
    def retrieve_text(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        query_vec = self.embedder.embed_text(query)[0]
        return self.text_store.search(query_vec, top_k=top_k)

    # -----------------------------
    # IMAGE QUERY → IMAGE SEARCH
    # -----------------------------
    def retrieve_image(
        self,
        image,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        query_vec = self.embedder.embed_image(image)[0]
        return self.image_store.search(query_vec, top_k=top_k)

    # -----------------------------
    # TEXT → IMAGE (CLIP)
    # -----------------------------
    def retrieve_text_to_image(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        query_vec = self.embedder.embed_text_clip(query)[0]
        return self.image_store.search(query_vec, top_k=top_k)

    # -----------------------------
    # HYBRID RETRIEVAL (TEXT + IMAGE)
    # -----------------------------
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combines text + image retrieval
        alpha: weight for text vs image
        """
        text_results = self.retrieve_text(query, top_k=top_k)
        image_results = self.retrieve_text_to_image(query, top_k=top_k)

        combined_scores = {}

        # Merge text results
        for r in text_results:
            combined_scores[r["id"]] = {
                "score": alpha * r["score"],
                "metadata": r["metadata"]
            }

        # Merge image results
        for r in image_results:
            if r["id"] in combined_scores:
                combined_scores[r["id"]]["score"] += (1 - alpha) * r["score"]
            else:
                combined_scores[r["id"]] = {
                    "score": (1 - alpha) * r["score"],
                    "metadata": r["metadata"]
                }

        # Sort results
        results = [
            {"id": k, "score": v["score"], "metadata": v["metadata"]}
            for k, v in combined_scores.items()
        ]

        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    # -----------------------------
    # BUILD CONTEXT FOR LLM
    # -----------------------------
    def build_context(
        self,
        results: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> str:
        """
        Combine retrieved chunks into context
        """
        context_parts = []
        current_length = 0

        for r in results:
            text = r["metadata"].get("text", "")

            if not text:
                continue

            if current_length + len(text) > max_tokens:
                break

            context_parts.append(text)
            current_length += len(text)

        return "\n\n".join(context_parts)