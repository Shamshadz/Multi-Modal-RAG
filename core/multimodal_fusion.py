from typing import List, Dict, Any
import math


class MultiModalFusion:
    def __init__(self, strategy: str = "weighted_sum"):
        """
        strategy:
            - weighted_sum
            - max_score
            - reciprocal_rank
        """
        self.strategy = strategy

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    def _normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return results

        for r in results:
            r["norm_score"] = (r["score"] - min_s) / (max_s - min_s)

        return results

    # -----------------------------
    # WEIGHTED SUM FUSION
    # -----------------------------
    def _weighted_sum(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
        alpha: float
    ) -> List[Dict[str, Any]]:

        text_results = self._normalize_scores(text_results)
        image_results = self._normalize_scores(image_results)

        fused = {}

        for r in text_results:
            fused[r["id"]] = {
                "score": alpha * r.get("norm_score", r["score"]),
                "metadata": r["metadata"]
            }

        for r in image_results:
            if r["id"] in fused:
                fused[r["id"]]["score"] += (1 - alpha) * r.get("norm_score", r["score"])
            else:
                fused[r["id"]] = {
                    "score": (1 - alpha) * r.get("norm_score", r["score"]),
                    "metadata": r["metadata"]
                }

        return self._sort_results(fused)

    # -----------------------------
    # MAX SCORE FUSION
    # -----------------------------
    def _max_score(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        fused = {}

        for r in text_results + image_results:
            if r["id"] not in fused:
                fused[r["id"]] = {
                    "score": r["score"],
                    "metadata": r["metadata"]
                }
            else:
                fused[r["id"]]["score"] = max(fused[r["id"]]["score"], r["score"])

        return self._sort_results(fused)

    # -----------------------------
    # RECIPROCAL RANK FUSION (RRF)
    # -----------------------------
    def _rrf(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:

        fused = {}

        def update(results):
            for rank, r in enumerate(results):
                score = 1 / (k + rank + 1)
                if r["id"] not in fused:
                    fused[r["id"]] = {
                        "score": score,
                        "metadata": r["metadata"]
                    }
                else:
                    fused[r["id"]]["score"] += score

        update(text_results)
        update(image_results)

        return self._sort_results(fused)

    # -----------------------------
    # SORT RESULTS
    # -----------------------------
    def _sort_results(self, fused_dict: Dict[str, Dict]) -> List[Dict[str, Any]]:
        results = [
            {"id": k, "score": v["score"], "metadata": v["metadata"]}
            for k, v in fused_dict.items()
        ]

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # -----------------------------
    # PUBLIC FUSION METHOD
    # -----------------------------
    def fuse(
        self,
        text_results: List[Dict[str, Any]],
        image_results: List[Dict[str, Any]],
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combine results based on selected strategy
        """

        if self.strategy == "weighted_sum":
            return self._weighted_sum(text_results, image_results, alpha)

        elif self.strategy == "max_score":
            return self._max_score(text_results, image_results)

        elif self.strategy == "reciprocal_rank":
            return self._rrf(text_results, image_results)

        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")