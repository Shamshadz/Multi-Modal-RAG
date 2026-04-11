"""
evaluate.py
===========
Offline evaluation of retrieval quality.

Metrics implemented
-------------------
``Recall@K``
    Fraction of relevant documents found in the top-K results.
    Good when you want to know "did we surface everything important?".

``Precision@K``
    Fraction of top-K results that are relevant.
    Good when you care about the quality of the first K shown results.

``MRR`` (Mean Reciprocal Rank)
    Average of 1 / (rank of first relevant result).
    Good when the user typically looks at just the first hit.

Providing ground truth
----------------------
Replace ``EVAL_DATA`` with real query/relevant-ID pairs, or load from a JSON
file.  Each entry must have the shape::

    {
        "query": "some question",
        "relevant_ids": ["doc_id_1", "doc_id_2"]
    }

CLI usage
---------
.. code-block:: bash

    python -m scripts.evaluation.evaluate --mode text --k 5
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List, Optional

from core.embeddings import EmbeddingModel
from core.retriever import Retriever
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default evaluation dataset (replace with your own)
# ---------------------------------------------------------------------------

EVAL_DATA: List[Dict[str, Any]] = [
    {"query": "cat", "relevant_ids": ["cat_1", "cat_2"]},
    {"query": "car", "relevant_ids": ["car_1"]},
]


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def recall_at_k(
    results: List[Dict[str, Any]],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Compute Recall@K.

    Parameters
    ----------
    results : list[dict]
        Ranked retrieval results (``id`` key required).
    relevant_ids : list[str]
        Ground-truth document IDs.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Recall in [0, 1].
    """
    if not relevant_ids:
        return 0.0
    retrieved = {r["id"] for r in results[:k]}
    hits = sum(1 for rid in relevant_ids if rid in retrieved)
    return hits / len(relevant_ids)


def precision_at_k(
    results: List[Dict[str, Any]],
    relevant_ids: List[str],
    k: int,
) -> float:
    """
    Compute Precision@K.

    Parameters
    ----------
    results : list[dict]
        Ranked retrieval results.
    relevant_ids : list[str]
        Ground-truth document IDs.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Precision in [0, 1].
    """
    if not results or k == 0:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = results[:k]
    hits = sum(1 for r in top_k if r["id"] in relevant_set)
    return hits / k


def reciprocal_rank(
    results: List[Dict[str, Any]],
    relevant_ids: List[str],
) -> float:
    """
    Compute the Reciprocal Rank for a single query.

    Parameters
    ----------
    results : list[dict]
        Ranked retrieval results.
    relevant_ids : list[str]
        Ground-truth document IDs.

    Returns
    -------
    float
        1 / rank of the first relevant result, or 0 if none found.
    """
    relevant_set = set(relevant_ids)
    for rank, r in enumerate(results, start=1):
        if r["id"] in relevant_set:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(
    eval_data: Optional[List[Dict[str, Any]]] = None,
    mode: str = "text",
    k: int = 5,
) -> Dict[str, float]:
    """
    Run evaluation across all queries in *eval_data*.

    Parameters
    ----------
    eval_data : list[dict], optional
        List of ``{"query": ..., "relevant_ids": [...]}`` entries.
        Defaults to the module-level ``EVAL_DATA``.
    mode : str
        Retrieval mode: ``"text"``, ``"hybrid"``, or ``"text_to_image"``.
    k : int
        Rank cutoff for Recall@K and Precision@K.

    Returns
    -------
    dict
        ``{"recall": float, "precision": float, "mrr": float}``
    """
    if eval_data is None:
        eval_data = EVAL_DATA

    text_store = VectorStore.load("stores/text_store")
    image_store = VectorStore.load("stores/image_store")

    retriever = Retriever(
        text_store=text_store,
        image_store=image_store,
        embedder=EmbeddingModel(),
    )

    recalls, precisions, rrs = [], [], []

    for item in eval_data:
        query = item["query"]
        relevant_ids = item["relevant_ids"]

        if mode == "text":
            results = retriever.retrieve_text(query, top_k=k)
        elif mode == "hybrid":
            results = retriever.retrieve_hybrid(query, top_k=k)
        elif mode == "text_to_image":
            results = retriever.retrieve_text_to_image(query, top_k=k)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        r_k = recall_at_k(results, relevant_ids, k)
        p_k = precision_at_k(results, relevant_ids, k)
        rr = reciprocal_rank(results, relevant_ids)

        recalls.append(r_k)
        precisions.append(p_k)
        rrs.append(rr)

        logger.info(
            "Query: %-30s | R@%d=%.2f | P@%d=%.2f | RR=%.2f",
            query, k, r_k, k, p_k, rr,
        )

    aggregated = {
        "recall": sum(recalls) / len(recalls),
        "precision": sum(precisions) / len(precisions),
        "mrr": sum(rrs) / len(rrs),
    }

    print(f"\n{'='*50}")
    print(f"Evaluation results (mode={mode}, k={k})")
    print(f"{'='*50}")
    print(f"  Recall@{k}:    {aggregated['recall']:.4f}")
    print(f"  Precision@{k}: {aggregated['precision']:.4f}")
    print(f"  MRR:          {aggregated['mrr']:.4f}")

    return aggregated


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Multi-Modal RAG – evaluation.")
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["text", "hybrid", "text_to_image"],
    )
    parser.add_argument("--k", type=int, default=5, help="Rank cutoff.")
    parser.add_argument(
        "--eval_json",
        type=str,
        default="",
        help="Path to a JSON file with eval data (overrides built-in EVAL_DATA).",
    )
    args = parser.parse_args()

    eval_data = None
    if args.eval_json:
        with open(args.eval_json) as fh:
            eval_data = json.load(fh)

    evaluate(eval_data=eval_data, mode=args.mode, k=args.k)


if __name__ == "__main__":
    main()