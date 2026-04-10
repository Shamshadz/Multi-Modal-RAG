from typing import List, Dict
from core.retriever import Retriever
from core.vector_store import VectorStore
from core.embeddings import EmbeddingModel


# Dummy ground truth (you replace this later)
EVAL_DATA = [
    {
        "query": "cat",
        "relevant_ids": ["cat_1", "cat_2"]
    },
    {
        "query": "car",
        "relevant_ids": ["car_1"]
    }
]


def recall_at_k(results: List[Dict], relevant_ids: List[str], k: int):
    retrieved_ids = [r["id"] for r in results[:k]]
    hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return hits / len(relevant_ids) if relevant_ids else 0


def evaluate():
    text_store = VectorStore.load("stores/text_store")
    image_store = VectorStore.load("stores/image_store")

    retriever = Retriever(
        text_store=text_store,
        image_store=image_store,
        embedder=EmbeddingModel()
    )

    k = 5
    scores = []

    for item in EVAL_DATA:
        results = retriever.retrieve_text(item["query"], top_k=k)
        score = recall_at_k(results, item["relevant_ids"], k)
        scores.append(score)

        print(f"Query: {item['query']} | Recall@{k}: {score:.2f}")

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Recall@{k}: {avg_score:.2f}")


if __name__ == "__main__":
    evaluate()