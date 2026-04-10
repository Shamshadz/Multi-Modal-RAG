import os
from typing import List
from tqdm import tqdm

from core.chunking import TextChunker
from core.embeddings import EmbeddingModel
from core.vector_store import VectorStore


class DataIngestor:
    def __init__(
        self,
        text_store_path: str = "stores/text_store",
        image_store_path: str = "stores/image_store",
    ):
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=100)
        self.embedder = EmbeddingModel()

        self.text_store = VectorStore(dim=384)   # MiniLM
        self.image_store = VectorStore(dim=512)  # CLIP

        self.text_store_path = text_store_path
        self.image_store_path = image_store_path

    # -----------------------------
    # TEXT INGESTION
    # -----------------------------
    def ingest_text_files(self, folder_path: str):
        files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

        for file in tqdm(files, desc="Processing text files"):
            full_path = os.path.join(folder_path, file)

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self.chunker.chunk_text(
                content,
                metadata={"source": file}
            )

            texts = [c["text"] for c in chunks]
            embeddings = self.embedder.embed_text(texts)

            self.text_store.add(
                embeddings,
                metadatas=chunks,
                ids=[f"{file}_{i}" for i in range(len(chunks))]
            )

    # -----------------------------
    # IMAGE INGESTION
    # -----------------------------
    def ingest_images(self, folder_path: str):
        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for file in tqdm(files, desc="Processing images"):
            full_path = os.path.join(folder_path, file)

            embedding = self.embedder.embed_image(full_path)[0]

            metadata = {
                "image_path": full_path,
                "source": file
            }

            self.image_store.add(
                [embedding],
                metadatas=[metadata],
                ids=[file]
            )

    # -----------------------------
    # SAVE STORES
    # -----------------------------
    def save(self):
        os.makedirs("stores", exist_ok=True)

        self.text_store.save(self.text_store_path)
        self.image_store.save(self.image_store_path)


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    ingestor = DataIngestor()

    # Update paths as needed
    ingestor.ingest_text_files("data/text")
    ingestor.ingest_images("data/images")

    ingestor.save()

    print("Ingestion complete.")