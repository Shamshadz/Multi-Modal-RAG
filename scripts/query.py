"""
query.py
========
Command-line interface for raw retrieval (no LLM generation).

Useful for debugging retrieval quality independently of the generator.

CLI usage
---------
.. code-block:: bash

    # Text retrieval
    python -m scripts.query --query "What is CLIP?" --mode text

    # Image query
    python -m scripts.query --image path/to/photo.jpg --mode image

    # Hybrid
    python -m scripts.query --query "cat sitting on a mat" --mode hybrid --top_k 8
"""

from __future__ import annotations

import argparse
import json
import logging

from core.embeddings import EmbeddingModel
from core.retriever import Retriever
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Multi-Modal RAG – retrieval CLI.")
    parser.add_argument("--query", type=str, help="Text query string.")
    parser.add_argument("--image", type=str, help="Path to a query image.")
    parser.add_argument(
        "--mode",
        type=str,
        default="text",
        choices=["text", "image", "hybrid", "text_to_image"],
        help="Retrieval mode.",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of results.")
    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON."
    )
    args = parser.parse_args()

    # Validate arguments
    if args.mode in ("text", "hybrid", "text_to_image") and not args.query:
        parser.error(f"--query is required for mode '{args.mode}'.")
    if args.mode == "image" and not args.image:
        parser.error("--image is required for mode 'image'.")

    # Load stores
    text_store = VectorStore.load("stores/text_store")
    image_store = VectorStore.load("stores/image_store")

    retriever = Retriever(
        text_store=text_store,
        image_store=image_store,
        embedder=EmbeddingModel(),
    )

    # Run retrieval
    if args.mode == "text":
        results = retriever.retrieve_text(args.query, top_k=args.top_k)
    elif args.mode == "image":
        results = retriever.retrieve_image(args.image, top_k=args.top_k)
    elif args.mode == "text_to_image":
        results = retriever.retrieve_text_to_image(args.query, top_k=args.top_k)
    elif args.mode == "hybrid":
        results = retriever.retrieve_hybrid(args.query, top_k=args.top_k)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Output
    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    print(f"\n=== RESULTS (mode={args.mode}, top_k={args.top_k}) ===\n")
    for i, r in enumerate(results, start=1):
        print(f"Rank {i}")
        print(f"  ID:     {r['id']}")
        print(f"  Score:  {r['score']:.4f}")
        meta_preview = {k: v for k, v in r["metadata"].items() if k != "text"}
        print(f"  Meta:   {meta_preview}")
        text_snippet = r["metadata"].get("text", "")[:120]
        if text_snippet:
            print(f"  Text:   {text_snippet!r}")
        print()


if __name__ == "__main__":
    main()