"""
retriever.py
============
Orchestrates retrieval across text and image modalities and builds the context
string passed to the LLM.

Retrieval modes
---------------
``retrieve_text``
    Encodes the query with SentenceTransformer → searches the text FAISS store.
    Pure text-to-text retrieval.

``retrieve_image``
    Encodes an image query with CLIP → searches the image FAISS store.
    Returns visually similar images.

``retrieve_text_to_image``
    Encodes the *text* query with CLIP's text encoder → searches the image
    FAISS store.  Cross-modal: words describe what images to find.

``retrieve_hybrid``
    Runs ``retrieve_text`` + ``retrieve_text_to_image``, then fuses the ranked
    lists with :class:`~core.multimodal_fusion.MultiModalFusion`.

Context building
----------------
``build_context`` concatenates the ``text`` field from each result's metadata
up to a character budget (``max_tokens``).  The resulting string is injected
directly into the LLM prompt.

Usage
-----
>>> retriever = Retriever(text_store, image_store, embedder)
>>> results   = retriever.retrieve_hybrid("What is neural scaling?", top_k=5)
>>> context   = retriever.build_context(results)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.embeddings import EmbeddingModel
from core.multimodal_fusion import MultiModalFusion
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Multi-modal retriever combining text and image vector stores.

    Parameters
    ----------
    text_store : VectorStore, optional
        FAISS store holding text chunk embeddings (384-d SentenceTransformer).
    image_store : VectorStore, optional
        FAISS store holding image embeddings (512-d CLIP).
    embedder : EmbeddingModel, optional
        Shared embedding model.  A new one is created if *None*.
    fusion_strategy : str
        Strategy passed to :class:`~core.multimodal_fusion.MultiModalFusion`.
        One of ``"weighted_sum"``, ``"max_score"``, ``"reciprocal_rank"``.
    """

    def __init__(
        self,
        text_store: Optional[VectorStore] = None,
        image_store: Optional[VectorStore] = None,
        embedder: Optional[EmbeddingModel] = None,
        fusion_strategy: str = "reciprocal_rank",
    ):
        self.text_store = text_store
        self.image_store = image_store
        self.embedder = embedder or EmbeddingModel()
        self.fusion = MultiModalFusion(strategy=fusion_strategy)

    # ------------------------------------------------------------------
    # Text → text
    # ------------------------------------------------------------------

    def retrieve_text(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Encode *query* with SentenceTransformer and search the text store.

        Parameters
        ----------
        query : str
            Natural-language question or keyword.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Ranked results, each with ``id``, ``score``, and ``metadata``.

        Raises
        ------
        RuntimeError
            If ``text_store`` was not provided at construction time.
        """
        if self.text_store is None:
            raise RuntimeError("text_store is not initialised.")

        query_vec = self.embedder.embed_text(query)[0]
        return self.text_store.search(query_vec, top_k=top_k)

    # ------------------------------------------------------------------
    # Image → image
    # ------------------------------------------------------------------

    def retrieve_image(
        self,
        image,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Encode an *image* with CLIP and search the image store.

        Parameters
        ----------
        image : str or PIL.Image
            File path or PIL Image to use as the query.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Ranked visually-similar images.

        Raises
        ------
        RuntimeError
            If ``image_store`` was not provided at construction time.
        """
        if self.image_store is None:
            raise RuntimeError("image_store is not initialised.")

        query_vec = self.embedder.embed_image(image)[0]
        return self.image_store.search(query_vec, top_k=top_k)

    # ------------------------------------------------------------------
    # Text → image (cross-modal)
    # ------------------------------------------------------------------

    def retrieve_text_to_image(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Encode *query* with the CLIP text encoder and search the image store.

        This enables zero-shot cross-modal retrieval: describe what you want
        in words and retrieve visually matching images.

        Parameters
        ----------
        query : str
            Description of the desired image content.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            Ranked image results.
        """
        if self.image_store is None:
            raise RuntimeError("image_store is not initialised.")

        query_vec = self.embedder.embed_text_clip(query)[0]
        return self.image_store.search(query_vec, top_k=top_k)

    # ------------------------------------------------------------------
    # Hybrid (text + image → fused)
    # ------------------------------------------------------------------

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Run text and cross-modal image retrieval, then fuse the results.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int
            Number of results *after* fusion.
        alpha : float
            Weight given to text results in the ``weighted_sum`` strategy.
            Ignored by ``max_score`` and ``reciprocal_rank``.

        Returns
        -------
        list[dict]
            Fused and re-ranked results, length ≤ *top_k*.
        """
        text_results: List[Dict[str, Any]] = []
        image_results: List[Dict[str, Any]] = []

        if self.text_store is not None:
            text_results = self.retrieve_text(query, top_k=top_k)

        if self.image_store is not None:
            image_results = self.retrieve_text_to_image(query, top_k=top_k)

        fused = self.fusion.fuse(text_results, image_results, alpha=alpha)
        return fused[:top_k]

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def build_context(
        self,
        results: List[Dict[str, Any]],
        max_chars: int = 4000,
    ) -> str:
        """
        Concatenate retrieved chunk texts into a single context string.

        Iterates results in ranked order, appending ``metadata["text"]`` until
        the character budget is exhausted.

        Parameters
        ----------
        results : list[dict]
            Output of any ``retrieve_*`` method.
        max_chars : int
            Soft character cap on the returned context string.

        Returns
        -------
        str
            Newline-separated chunk texts, ready for prompt injection.
        """
        parts: List[str] = []
        used = 0

        for r in results:
            text = r.get("metadata", {}).get("text", "")
            if not text:
                continue
            if used + len(text) > max_chars:
                # Include as much of this chunk as fits
                remaining = max_chars - used
                if remaining > 50:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            used += len(text)

        return "\n\n".join(parts)