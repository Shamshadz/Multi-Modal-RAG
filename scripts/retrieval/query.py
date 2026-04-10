import argparse

from core.retriever import Retriever
from core.vector_store import VectorStore
from core.embeddings import EmbeddingModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, help="Text query")
    parser.add_argument("--image", type=str, help="Image path")
    parser.add_argument("--mode", type=str, default="text",
                        choices=["text", "image", "hybrid", "text_to_image"])

    args = parser.parse_args()

    # Load stores
    text_store = VectorStore.load("stores/text_store")
    image_store = VectorStore.load("stores/image_store")

    retriever = Retriever(
        text_store=text_store,
        image_store=image_store,
        embedder=EmbeddingModel()
    )

    # -----------------------------
    # RUN QUERY
    # -----------------------------
    if args.mode == "text":
        results = retriever.retrieve_text(args.query)

    elif args.mode == "image":
        results = retriever.retrieve_image(args.image)

    elif args.mode == "text_to_image":
        results = retriever.retrieve_text_to_image(args.query)

    elif args.mode == "hybrid":
        results = retriever.retrieve_hybrid(args.query)

    else:
        raise ValueError("Invalid mode")

    # -----------------------------
    # PRINT RESULTS
    # -----------------------------
    print("\n=== RESULTS ===\n")

    for i, r in enumerate(results):
        print(f"Rank {i+1}")
        print(f"ID: {r['id']}")
        print(f"Score: {r['score']:.4f}")
        print(f"Metadata: {r['metadata']}")
        print("-" * 40)


if __name__ == "__main__":
    main()