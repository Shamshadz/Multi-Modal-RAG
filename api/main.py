from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
import uuid

from core.retriever import Retriever
from core.vector_store import VectorStore
from core.embeddings import EmbeddingModel
from scripts.generation.generate import Generator, build_prompt


app = FastAPI(title="Multi-Modal RAG API")


# -----------------------------
# LOAD MODELS ON STARTUP
# -----------------------------
print("Loading models and stores...")

text_store = VectorStore.load("stores/text_store")
image_store = VectorStore.load("stores/image_store")

embedder = EmbeddingModel()

retriever = Retriever(
    text_store=text_store,
    image_store=image_store,
    embedder=embedder
)

generator = Generator(model_type="openai")

print("System ready.")


# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class QueryRequest(BaseModel):
    query: str
    mode: str = "text"  # text | hybrid | text_to_image
    top_k: int = 5


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def root():
    return {"status": "running"}


# -----------------------------
# TEXT / HYBRID QUERY
# -----------------------------
@app.post("/query")
def query_api(request: QueryRequest):
    if request.mode == "text":
        results = retriever.retrieve_text(request.query, top_k=request.top_k)

    elif request.mode == "hybrid":
        results = retriever.retrieve_hybrid(request.query, top_k=request.top_k)

    elif request.mode == "text_to_image":
        results = retriever.retrieve_text_to_image(request.query, top_k=request.top_k)

    else:
        return {"error": "Invalid mode"}

    context = retriever.build_context(results)

    prompt = build_prompt(request.query, context)
    answer = generator.generate(prompt)

    return {
        "query": request.query,
        "answer": answer,
        "sources": results,
        "context_preview": context[:500]
    }


# -----------------------------
# IMAGE QUERY
# -----------------------------
@app.post("/image-query")
def image_query(file: UploadFile = File(...)):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = retriever.retrieve_image(file_path)

    context = retriever.build_context(results)

    prompt = build_prompt("Describe this image context", context)
    answer = generator.generate(prompt)

    return {
        "answer": answer,
        "sources": results
    }