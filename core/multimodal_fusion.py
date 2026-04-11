"""
multimodal_fusion.py
====================
Merges ranked result lists from heterogeneous retrieval sources (text, image)
into a single unified ranking.

Three fusion strategies are provided:

``weighted_sum``
    Normalises each list's scores to [0, 1] and combines them as a
    weighted average.  Controlled by *alpha* (weight of text results).
    Best when scores from both modalities are calibrated on a similar scale.

``max_score``
    Takes the maximum score across modalities for each document.
    Simple and robust; preserves high-confidence signals from either source.

``reciprocal_rank`` (RRF)
    Score = Σ 1 / (k + rank).  Rank-based: ignores raw score magnitudes.
    State-of-the-art for heterogeneous fusion; insensitive to score scale
    differences.  Default ``k=60`` follows the original RRF paper.

Usage
-----
>>> fusion = MultiModalFusion(strategy="reciprocal_rank")
>>> merged = fusion.fuse(text_results, image_results)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_STRATEGIES = ("weighted_sum", "max_score", "reciprocal_rank")


class MultiModalFusion:
    """
    Fuses ranked result lists from text and image retrievers.

    Parameters
    ----------
    strategy : str
        One of ``"weighted_sum"``, ``"max_score"``, ``"reciprocal_rank"``.
    """

    def __init__(self, strategy: str = "weighted_sum"):
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {_STRATEGIES}."
            )
        self.strategy = strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Merge two ranked lists into one.

        Parameters
        ----------
        text_results : list[dict]
            Results from text retrieval.  Each dict must have ``"id"``,
            ``"score"``, and ``"metadata"`` keys.
        image_results : list[dict]
            Results from image retrieval.  Same schema.
        alpha : float
            Used only by ``weighted_sum``.  Weight given to text scores;
            image scores receive weight ``(1 - alpha)``.  Range: [0, 1].

        Returns
        -------
        list[dict]
            Merged list sorted by descending fused score.
        """
        if not text_results and not image_results:
            return []

        if self.strategy == "weighted_sum":
            return self._weighted_sum(text_results, image_results, alpha)
        elif self.strategy == "max_score":
            return self._max_score(text_results, image_results)
        else:  # reciprocal_rank
            return self._rrf(text_results, image_results)

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _weighted_sum(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
        alpha: float,
    ) -> List[Dict[str, Any]]:
        """Min-max normalise each list then blend with weight *alpha*."""
        text_results = self._normalize_scores(text_results)
        image_results = self._normalize_scores(image_results)

        fused: Dict[str, Dict[str, Any]] = {}

        for r in text_results:
            fused[r["id"]] = {
                "score": alpha * r.get("norm_score", r["score"]),
                "metadata": r["metadata"],
            }

        for r in image_results:
            contrib = (1 - alpha) * r.get("norm_score", r["score"])
            if r["id"] in fused:
                fused[r["id"]]["score"] += contrib
            else:
                fused[r["id"]] = {"score": contrib, "metadata": r["metadata"]}

        return self._sort_fused(fused)

    def _max_score(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keep the highest score a document received in either modality."""
        fused: Dict[str, Dict[str, Any]] = {}

        for r in text_results + image_results:
            if r["id"] not in fused:
                fused[r["id"]] = {"score": r["score"], "metadata": r["metadata"]}
            else:
                fused[r["id"]]["score"] = max(fused[r["id"]]["score"], r["score"])

        return self._sort_fused(fused)

    def _rrf(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion.

        Score contribution for a document at rank *r* is ``1 / (k + r + 1)``.
        """
        fused: Dict[str, Dict[str, Any]] = {}

        def _update(results: List[Dict[str, Any]]) -> None:
            for rank, r in enumerate(results):
                contrib = 1.0 / (k + rank + 1)
                if r["id"] not in fused:
                    fused[r["id"]] = {"score": contrib, "metadata": r["metadata"]}
                else:
                    fused[r["id"]]["score"] += contrib

        _update(text_results)
        _update(image_results)

        return self._sort_fused(fused)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add a ``norm_score`` key using min-max normalisation."""
        if not results:
            return results

        scores = [r["score"] for r in results]
        lo, hi = min(scores), max(scores)

        for r in results:
            r["norm_score"] = (r["score"] - lo) / (hi - lo) if hi != lo else 1.0

        return results

    @staticmethod
    def _sort_fused(
        fused: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert the fused dict to a list sorted by descending score."""
        items = [
            {"id": k, "score": v["score"], "metadata": v["metadata"]}
            for k, v in fused.items()
        ]
        items.sort(key=lambda x: x["score"], reverse=True)
        return items